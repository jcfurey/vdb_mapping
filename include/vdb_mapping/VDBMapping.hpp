// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-

// -- BEGIN LICENSE BLOCK ----------------------------------------------
// Copyright 2021 FZI Forschungszentrum Informatik
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// -- END LICENSE BLOCK ------------------------------------------------

//----------------------------------------------------------------------
/*!\file
 *
 * \author  Marvin Große Besselmann grosse@fzi.de
 * \author  Lennart Puck puck@fzi.de
 * \date    2020-12-23
 *
 */
//----------------------------------------------------------------------
#ifndef VDB_MAPPING_VDB_MAPPING_H_INCLUDED
#define VDB_MAPPING_VDB_MAPPING_H_INCLUDED


#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <eigen3/Eigen/Geometry>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <openvdb/Types.h>
#include <openvdb/io/Stream.h>
#include <openvdb/math/DDA.h>
#include <openvdb/math/Ray.h>
#include <openvdb/openvdb.h>
#include <openvdb/tools/Clip.h>
#include <openvdb/tools/Morphology.h>
#include <openvdb/tools/RayIntersector.h>

#include <zstd.h>


namespace vdb_mapping {


/*!
 * \brief Accumulation of configuration parameters
 */
struct BaseConfig
{
  // When redesigning split into static and changable config
  double max_range           = 10.0;
  bool fast_mode             = false;
  double accumulation_period = 1.0;
  std::string map_directory_path;
  // Applies to both compressed input and the size declared by a Zstd frame.
  // Network-facing section transport must not be able to request an
  // unbounded allocation before OpenVDB has a chance to validate the stream.
  std::size_t max_serialized_grid_bytes = 512ULL * 1024ULL * 1024ULL;
};


/*!
 * \brief Main Mapping class which handles all data integration
 */
template <typename TData, typename TConfig = BaseConfig, typename PointType = pcl::PointXYZ>
class VDBMapping
{
public:
  using PointT      = PointType;
  using PointCloudT = pcl::PointCloud<PointT>;

  using RayT  = openvdb::math::Ray<double>;
  using Vec3T = RayT::Vec3Type;
  using DDAT  = openvdb::math::DDA<RayT, 0>;

  using GridT       = openvdb::Grid<typename openvdb::tree::Tree4<TData, 5, 4, 3>::Type>;
  using UpdateGridT = openvdb::Grid<openvdb::tree::Tree4<bool, 1, 4, 3>::Type>;

  struct InputSource
  {
    std::string source_id;
    double max_range;
    std::shared_ptr<openvdb::tools::VolumeRayIntersector<openvdb::FloatGrid> >
      volume_ray_intersector;
    UpdateGridT::Ptr update_grid;
    std::mutex update_grid_mutex;
    std::mutex input_data_mutex;
    std::optional<std::pair<typename PointCloudT::ConstPtr, Eigen::Matrix<double, 3, 1> > > input_data;
    std::chrono::milliseconds max_input_period;
    std::condition_variable data_available_cv;
    // Per-source behavior (defaults preserve the upstream lidar-style
    // semantics: every point clears along its ray and hits its endpoint).
    // Decoupling the two lets e.g. a sonar stack fill from confident hit
    // points only (no self-erosion by grazing rays) while a dedicated
    // clearing cloud carves geometry-aware free space.
    bool ray_clearing  = true;  // rays carve free space toward each point
    bool endpoint_hits = true;  // endpoints are marked occupied
    double prob_hit  = -1.0;    // <= 0: use the map-wide probability
    double prob_miss = -1.0;
  };

  /*!
   * \brief Severity of a library log message
   */
  enum class LogLevel
  {
    Info,
    Warning,
    Error
  };

  /*!
   * \brief Callback type for routing library log messages into a host
   * logging framework (e.g. rclcpp logging)
   */
  using LogCallbackT = std::function<void(LogLevel, const std::string&)>;

  /*!
   * \brief Callback type for providing custom time (e.g. ROS time in nanoseconds)
   */
  using TimeCallbackT = std::function<uint64_t()>;

  /*!
   * \brief Routes all library log output through the given callback instead
   * of stdout/stderr. Pass an empty callback to restore the default
   * behavior. The callback is invoked from the calling thread, including the
   * internal worker threads, but never concurrently.
   *
   * \param callback Log callback
   */
  void setLogCallback(LogCallbackT callback)
  {
    std::lock_guard<std::mutex> lock(m_log_mutex);
    m_log_callback = std::move(callback);
  }

  /*!
   * \brief Routes all library time requests through the given callback instead
   * of std::chrono. Useful for providing simulated time from a host framework.
   *
   * \param callback Time callback returning nanoseconds since epoch
   */
  void setTimeCallback(TimeCallbackT callback)
  {
    std::lock_guard<std::mutex> lock(m_time_mutex);
    m_time_callback = std::move(callback);
  }

  /*!
   * \brief Gets the current time in nanoseconds
   */
  uint64_t getTimeNow() const
  {
    std::lock_guard<std::mutex> lock(m_time_mutex);
    if (m_time_callback)
    {
      return m_time_callback();
    }
    // Standalone (non-ROS) callers need epoch time for timestamped map
    // filenames. ROS hosts install a callback backed by the node clock, so
    // replay scheduling and filenames follow /clock there.
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::system_clock::now().time_since_epoch()).count();
  }

  VDBMapping()                  = delete;
  VDBMapping(const VDBMapping&) = delete;

  VDBMapping& operator=(const VDBMapping&) = delete;

  /*!
   * \brief Constructor creates a new VDBMapping object with parametrizable grid resolution
   *
   * \param resolution Resolution of the VDB Grid
   */
  explicit VDBMapping(const double resolution)
    : m_resolution(resolution)
    , m_config_set(false)
    , m_artificial_areas_present(false)
  {
    if (!std::isfinite(resolution) || resolution <= 0.0)
    {
      throw std::invalid_argument("VDB map resolution must be finite and positive");
    }
    m_map_mutex = std::make_shared<std::shared_mutex>();
    //  Initialize Grid
    openvdb::initialize();
    if (!GridT::isRegistered())
    {
      GridT::registerGrid();
    }
    if (!UpdateGridT::isRegistered())
    {
      UpdateGridT::registerGrid();
    }
    m_vdb_grid             = createVDBMap();
    m_artificial_area_grid = UpdateGridT::create(false);
    m_integration_thread   = std::thread(&VDBMapping::integrationThread, this);
  }

  /*!
   * \brief Stop all background worker threads explicitly
   */
  void stop()
  {
    m_thread_stop_signal = true;
    for (auto& [source_id, worker_thread] : m_worker_threads)
    {
      auto& source = m_input_sources[source_id];
      {
        // Hold the mutex the worker waits on while notifying so the stop
        // signal cannot fall into the gap between the worker's predicate
        // check and its blocking wait (lost wakeup -> join() hangs).
        std::lock_guard<std::mutex> lock(source->input_data_mutex);
        source->data_available_cv.notify_all();
      }
      if (worker_thread.joinable())
      {
        worker_thread.join();
      }
    }
    if (m_integration_thread.joinable())
    {
      m_integration_thread.join();
    }
  }

  /*!
   * \brief Destructor that tears down all threads
   */
  virtual ~VDBMapping()
  {
    stop();
  }

  /*!
   * \brief Creates a new VDB Grid
   *
   * \returns Grid shared pointer
   */
  typename GridT::Ptr createVDBMap()
  {
    typename GridT::Ptr new_map = GridT::create(TData());
    new_map->setTransform(openvdb::math::Transform::createLinearTransform(m_resolution));
    // The grid stores occupancy data, not a signed distance field; labelling
    // it GRID_LEVEL_SET makes OpenVDB tools and viewers interpret the values
    // as an SDF.
    new_map->setGridClass(openvdb::GRID_UNKNOWN);
    return new_map;
  }

  /*!
   * \brief Reset the current map
   */
  void resetMap()
  {
    // The map lock also guards m_input_sources against concurrent
    // registration in addInputSource, and ordering map lock -> update grid
    // lock matches accumulateUpdate/integrateUpdate.
    std::unique_lock map_lock(*m_map_mutex);
    m_vdb_grid->clear();
    m_vdb_grid = createVDBMap();
    // Artificial areas are session state too: updateMap re-activates every
    // coordinate in this grid each integration cycle, so leaving it populated
    // resurrects pre-reset artificial walls in the "clean" map (active voxels
    // at background value) that only remove_artificial_areas could purge.
    m_artificial_area_grid = UpdateGridT::create(false);
    m_artificial_areas_present = false;
    // The volume ray intersectors reference the replaced grid (they hold raw
    // pointers and a topology copy, not shared ownership) and must not
    // outlive it.
    resetVolumeRayIntersectors();

    clearPendingUpdatesLocked();
  }

  /*!
   * \brief Creates a timestamped file path inside the configured map directory
   *
   * \param suffix Filename suffix including the extension
   *
   * \returns Full file path
   */
  std::string timestampedMapPath(const std::string& suffix) const
  {
    std::time_t now_tt = static_cast<std::time_t>(getTimeNow() / 1000000000ULL);
    std::tm tm{};
    localtime_r(&now_tt, &tm);
    std::stringstream sstime;
    sstime << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S");

    std::string prefix = m_map_directory_path;
    if (!prefix.empty() && prefix.back() != '/')
    {
      prefix += '/';
    }
    return prefix + sstime.str() + suffix;
  }

  /*!
   * \brief Saves the current map as vdb file
   *
   * \returns Saving map successful
   */
  bool saveMap() const
  {
    std::string map_name = timestampedMapPath("_map.vdb");
    logMessage(LogLevel::Info, "Saving map to " + map_name);
    try
    {
      openvdb::io::File file_handle(map_name);
      openvdb::GridPtrVec grids;
      std::shared_lock map_lock(*m_map_mutex);
      grids.push_back(m_vdb_grid);
      file_handle.write(grids);
      file_handle.close();
    }
    catch (const std::exception& e)
    {
      logMessage(LogLevel::Error, "Could not save map to " + map_name + ": " + e.what());
      return false;
    }
    return true;
  }

  /*!
   * \brief Saves the active values of the current map as PCD file
   *
   * \returns Saving pcd successful
   */
  bool saveMapToPCD()
  {
    std::string pcd_path = timestampedMapPath("_active_values_map.pcd");

    typename PointCloudT::Ptr cloud(new PointCloudT);

    std::shared_lock map_lock(*m_map_mutex);
    cloud->points.reserve(m_vdb_grid->activeVoxelCount());

    const auto append_point = [&](const openvdb::Coord& coord) {
      // indexToWorld already yields the voxel center under this library's
      // rounding convention (worldToIndex = floor(index + 0.5)); adding an
      // additional half-voxel offset here would shift a save/load round trip
      // by half a voxel diagonally.
      openvdb::Vec3d world_coord = m_vdb_grid->indexToWorld(coord);
      PointT point;
      point.x = static_cast<float>(world_coord.x());
      point.y = static_cast<float>(world_coord.y());
      point.z = static_cast<float>(world_coord.z());
      cloud->points.push_back(point);
    };

    for (typename GridT::ValueOnCIter iter = m_vdb_grid->cbeginValueOn(); iter; ++iter)
    {
      if (iter.isVoxelValue())
      {
        append_point(iter.getCoord());
        continue;
      }

      // pruneGrid() represents a uniform occupied region as one active tile.
      // A PCD has no tile primitive, so preserving the map requires emitting
      // every voxel covered by that iterator item.
      openvdb::CoordBBox tile_bbox;
      iter.getBoundingBox(tile_bbox);
      for (auto coord_iter = tile_bbox.begin(); coord_iter != tile_bbox.end(); ++coord_iter)
      {
        append_point(*coord_iter);
      }
    }
    map_lock.unlock();

    cloud->points.shrink_to_fit();
    cloud->width  = cloud->points.size();
    cloud->height = 1;

    // savePCDFile throws (rather than returning nonzero) on an empty cloud,
    // which would otherwise escape a ROS service callback uncaught.
    if (cloud->points.empty())
    {
      logMessage(LogLevel::Error, "Map contains no active voxels, not writing " + pcd_path);
      return false;
    }
    try
    {
      if (pcl::io::savePCDFile(pcd_path, *cloud) != 0)
      {
        logMessage(LogLevel::Error, "Could not write PCD file to " + pcd_path);
        return false;
      }
    }
    catch (const std::exception& e)
    {
      logMessage(LogLevel::Error, "Could not write PCD file to " + pcd_path + ": " + e.what());
      return false;
    }
    logMessage(LogLevel::Info, "Wrote pcd to: " + pcd_path);
    return true;
  }

  /*!
   * \brief Loads a stored map from a vdb file
   *
   * \returns Loading map from vdb file successful
   */
  bool loadMap(const std::string& file_path)
  {
    typename GridT::Ptr loaded_grid;
    try
    {
      openvdb::io::File file_handle(file_path);
      file_handle.open();
      for (openvdb::io::File::NameIterator name_iter = file_handle.beginName();
           name_iter != file_handle.endName();
           ++name_iter)
      {
        openvdb::GridBase::Ptr base_grid = file_handle.readGrid(name_iter.gridName());
        loaded_grid                      = openvdb::gridPtrCast<GridT>(base_grid);
        if (loaded_grid)
        {
          break;
        }
        logMessage(LogLevel::Warning,
                   "Skipping grid '" + name_iter.gridName() +
                     "': not compatible with this map's grid type");
      }
      file_handle.close();
    }
    catch (const std::exception& e)
    {
      logMessage(LogLevel::Error, "Could not load map from " + file_path + ": " + e.what());
      return false;
    }

    if (!loaded_grid)
    {
      logMessage(LogLevel::Error, "File " + file_path + " contains no compatible grid");
      return false;
    }

    std::unique_lock map_lock(*m_map_mutex);
    clearPendingUpdatesLocked();
    m_vdb_grid = loaded_grid;
    // The volume ray intersectors reference the replaced grid (they hold raw
    // pointers and a topology copy, not shared ownership) and must not
    // outlive it. They are rebuilt on the next integration cycle.
    resetVolumeRayIntersectors();
    // Keep the index<->world math consistent with the loaded grid. Otherwise
    // a map saved at a different resolution silently corrupts every
    // worldToIndex computation afterwards.
    const openvdb::Vec3d voxel_size = m_vdb_grid->voxelSize();
    if (std::abs(voxel_size.x() - m_resolution) > 1e-9)
    {
      std::ostringstream msg;
      msg << "Loaded map resolution " << voxel_size.x() << " differs from configured resolution "
          << m_resolution << ". Adopting the loaded resolution.";
      logMessage(LogLevel::Warning, msg.str());
      m_resolution = voxel_size.x();
    }
    return true;
  }

  /*!
   * \brief Loads a stored map from a pcd file
   *
   * \param file_path Path to pcd file
   * \param set_background Specifies if the background should be set
   * \param clear_map Specifies if the map has to be cleared before inserting data
   *
   * \returns Loading of map successful
   */
  bool loadMapFromPCD(const std::string& file_path, const bool set_background, const bool clear_map)
  {
    typename PointCloudT::Ptr cloud(new PointCloudT);
    if (pcl::io::loadPCDFile<PointT>(file_path, *cloud) == -1)
    {
      logMessage(LogLevel::Error, "Could not open PCD file " + file_path);
      return false;
    }
    std::unique_lock map_lock(*m_map_mutex);
    const bool success = createMapFromPointCloud(cloud, set_background, clear_map);
    if (success && clear_map)
    {
      clearPendingUpdatesLocked();
    }
    // The grid contents were rewritten in place; drop any VolumeRayIntersector
    // built from the previous topology (mirrors loadMap/resetMap) so the next
    // fast-mode query does not run against a stale snapshot.
    resetVolumeRayIntersectors();
    return success;
  }

  /*!
   * \brief Accumulates a new sensor point cloud to the update grid
   *
   * \param cloud Input cloud in map coordinates
   * \param origin Sensor position in map coordinates
   * \param source_id Specifies the input source
   */
  bool accumulateUpdate(const typename PointCloudT::ConstPtr& cloud,
                        const Eigen::Matrix<double, 3, 1>& origin,
                        const std::string source_id)
  {
    // The shared map lock also guards m_input_sources against concurrent
    // registration in addInputSource.
    std::shared_lock map_lock(*m_map_mutex);
    auto source = m_input_sources.find(source_id);
    if (source == m_input_sources.end())
    {
      logMessage(LogLevel::Warning,
                 "Tried to accumulate update for " + source_id + ". Source not available");
      return false;
    }
    return accumulateUpdateLocked(cloud, origin, source->second);
  }

protected:
  // Replacing a map invalidates samples and index-space updates from the
  // previous map. Workers hold the map lock while consuming and accumulating
  // input, so none can reappear after this transaction completes.
  void clearPendingUpdatesLocked()
  {
    for (auto& [source_id, source] : m_input_sources)
    {
      std::unique_lock input_lock(source->input_data_mutex);
      source->input_data.reset();
      std::unique_lock update_grid_lock(source->update_grid_mutex);
      source->update_grid = UpdateGridT::create(false);
    }
  }

  // Caller holds the map lock through input consumption and raycasting.
  bool accumulateUpdateLocked(const typename PointCloudT::ConstPtr& cloud,
                              const Eigen::Matrix<double, 3, 1>& origin,
                              const std::shared_ptr<InputSource>& source)
  {
    std::unique_lock update_grid_lock(source->update_grid_mutex);
    UpdateGridT::Accessor update_grid_acc = source->update_grid->getAccessor();

    // Resolve the map-wide fallback at USE time as well as at registration:
    // a source registered before setConfig() resolved its fallback against
    // m_max_range == 0 and stayed dead forever, even after a valid config
    // arrived.
    const double map_max_range      = m_max_range;
    const double effective_max_range = source->max_range > 0 ? source->max_range : map_max_range;
    if (effective_max_range > 0)
    {
      // A source registered after the map became non-empty has no
      // intersector until the next integration cycle. Fall back to the
      // standard DDA raycast instead of dereferencing a null intersector.
      if (m_fast_mode && source->volume_ray_intersector)
      {
        return raycastPointCloud(cloud,
                                 origin,
                                 effective_max_range,
                                 update_grid_acc,
                                 source->volume_ray_intersector,
                                 source->ray_clearing,
                                 source->endpoint_hits);
      }
      else
      {
        return raycastPointCloud(cloud,
                                 origin,
                                 effective_max_range,
                                 update_grid_acc,
                                 std::nullopt,
                                 source->ray_clearing,
                                 source->endpoint_hits);
      }
    }
    logMessage(LogLevel::Warning,
               "Input source " + source->source_id + " has no positive effective max range");
    return false;
  }

public:
  /*!
   * \brief Adds a new sensor point cloud to the accumulation pipeline
   *
   * \param cloud Input cloud in map coordinates
   * \param origin Sensor position in map coordinates
   * \param source_id Specifies the input source
   */
  void addDataToAccumulate(const typename PointCloudT::ConstPtr& cloud,
                           const Eigen::Matrix<double, 3, 1>& origin,
                           const std::string source_id)
  {
    std::shared_lock map_lock(*m_map_mutex);
    auto source = m_input_sources.find(source_id);
    if (source == m_input_sources.end())
    {
      logMessage(LogLevel::Warning,
                 "Tried to add data for accumulation of " + source_id + ". Source not available");
      return;
    }

    std::unique_lock lock(source->second->input_data_mutex);
    source->second->input_data = std::make_pair(cloud, origin);
    source->second->data_available_cv.notify_all();
  }

  /*!
   * \brief Integrates the accumulated updates into the map
   */
  void integrateUpdate(const bool finalize_map = true)
  {
    m_map_mutex_requested = true;
    std::unique_lock map_lock(*m_map_mutex);
    m_map_mutex_requested = false;
    for (auto& [source_id, source] : m_input_sources)
    {
      // The update grid pointer is swapped here while accumulateUpdate and
      // resetMap touch it under the same mutex; the map lock alone does not
      // serialise against resetMap's swap.
      std::unique_lock update_grid_lock(source->update_grid_mutex);
      // Per-source probability overrides apply for this source's grid only;
      // safe because the exclusive map lock is held for the whole loop.
      applySourceProbabilityOverride(source->prob_hit, source->prob_miss);
      updateMap(source->update_grid);
      clearSourceProbabilityOverride();

      source->update_grid = UpdateGridT::create(false);
    }
    // The clamping update policy drives stable regions to uniform log-odds
    // values, which is precisely what enables pruning to merge them into
    // tiles (Hornung et al., "OctoMap", Auton. Robots 2013, Sect. 3.4).
    // Without this the tree only ever grows.
    if (finalize_map)
    {
      m_vdb_grid->pruneGrid();
      updateVolumeRayIntersectors();
    }
    else
    {
      // The existing fast-mode intersectors describe the pre-integration
      // topology. Drop them so subsequent batch members fall back to exact
      // DDA instead of ray-marching through stale occupancy; finalizeUpdates
      // rebuilds them once after the batch.
      resetVolumeRayIntersectors();
    }
  }

  /*!
   * \brief Finalizes a batch of already-integrated updates.
   *
   * Batch re-renderers can preserve one probabilistic update per observation
   * while avoiding a full growing-tree prune after every individual cloud.
   */
  void finalizeUpdates()
  {
    m_map_mutex_requested = true;
    std::unique_lock map_lock(*m_map_mutex);
    m_map_mutex_requested = false;
    m_vdb_grid->pruneGrid();
    updateVolumeRayIntersectors();
  }

  /*!
   * \brief Handles the integration of new PointCloud data into the VDB data structure.
   * All datapoints are raycasted starting from the origin position
   *
   * \param cloud Input cloud in map coordinates
   * \param origin Sensor position in map coordinates
   * \param source_id Specifies the input source
   *
   * \returns Was the insertion of the new pointcloud successful
   */
  bool insertPointCloud(const typename PointCloudT::ConstPtr& cloud,
                        const Eigen::Matrix<double, 3, 1>& origin,
                        const std::string source_id,
                        const bool finalize_map = true)
  {
    if (!accumulateUpdate(cloud, origin, source_id))
    {
      return false;
    }
    integrateUpdate(finalize_map);
    return true;
  }

  /*!
   * \brief Removes points directly from the map
   *
   * \param cloud Input point cloud that should be removed
   */
  bool removePointsFromGrid(const typename PointCloudT::ConstPtr& cloud)
  {
    if (!cloud)
    {
      return false;
    }
    // setNodeToFree writes m_min_logodds, which is uninitialized until
    // setConfig has run; reject the call instead of storing garbage.
    if (!m_config_set)
    {
      logMessage(LogLevel::Error, "Map not properly configured. Did you call setConfig method?");
      return false;
    }
    std::unique_lock map_lock(*m_map_mutex);
    typename GridT::Accessor acc = m_vdb_grid->getAccessor();

    auto set_node = [&](TData& voxel_value, bool& active) { setNodeToFree(voxel_value, active); };


    for (const PointT& pt : *cloud)
    {
      openvdb::Coord index_pt;
      if (!worldToIndexChecked(openvdb::Vec3d(pt.x, pt.y, pt.z), index_pt))
      {
        continue;
      }
      acc.modifyValueAndActiveState(index_pt, set_node);
    }
    // In-place mutation leaves any VolumeRayIntersector built from the previous
    // topology stale; drop them so the next fast-mode query falls back to an
    // exact DDA until the next integration rebuilds them.
    resetVolumeRayIntersectors();
    return true;
  }

  /*!
   * \brief Adds points directly to the map
   *
   * \param cloud Input point cloud that should be added
   */
  bool addPointsToGrid(const typename PointCloudT::ConstPtr& cloud)
  {
    if (!cloud)
    {
      return false;
    }
    // setNodeToOccupied writes m_max_logodds, which is uninitialized until
    // setConfig has run; reject the call instead of storing garbage.
    if (!m_config_set)
    {
      logMessage(LogLevel::Error, "Map not properly configured. Did you call setConfig method?");
      return false;
    }
    std::unique_lock map_lock(*m_map_mutex);
    typename GridT::Accessor acc = m_vdb_grid->getAccessor();
    auto set_node                = [&](TData& voxel_value, bool& active) {
      setNodeToOccupied(voxel_value, active);
    };
    for (const PointT& pt : *cloud)
    {
      openvdb::Coord index_pt;
      if (!worldToIndexChecked(openvdb::Vec3d(pt.x, pt.y, pt.z), index_pt))
      {
        continue;
      }
      acc.modifyValueAndActiveState(index_pt, set_node);
    }
    // In-place mutation leaves any VolumeRayIntersector built from the previous
    // topology stale; drop them so the next fast-mode query falls back to an
    // exact DDA until the next integration rebuilds them.
    resetVolumeRayIntersectors();
    return true;
  }

  /*!
   * \brief  Raycasts a Pointcloud into an update Grid
   * For each point in the input pointcloud, a raycast is performed from the origin.
   * All cells along these rays are marked as active.
   *
   * All points are clipped according to the config's max_range parameter. If a point is within
   * this range, its corresponding cell value in the update grid is set to true and will be handled
   * as a sensor hit in the map update.
   *
   * \param cloud Input sensor point cloud
   * \param origin Origin of the sensor measurement
   * \param raycast_range Maximum raycasting range
   * \param update_grid_acc Accessor to the grid in which the raycasting takes place
   * \param intersector Optional volume ray intersector to determine collisions with the map
   *
   * \returns Raycasting successful
   */
  bool raycastPointCloud(
    const typename PointCloudT::ConstPtr& cloud,
    const Eigen::Matrix<double, 3, 1>& origin,
    const double raycast_range,
    UpdateGridT::Accessor& update_grid_acc,
    std::optional<std::shared_ptr<openvdb::tools::VolumeRayIntersector<openvdb::FloatGrid> > >
      intersector = std::nullopt,
    const bool ray_clearing  = true,
    const bool endpoint_hits = true)
  {
    if (!cloud)
    {
      return false;
    }
    // Creating a temporary grid in which the new data is casted. This way we prevent the
    // computation of redundant probability updates in the actual map

    // Check if a valid configuration was loaded
    if (!m_config_set)
    {
      logMessage(LogLevel::Error, "Map not properly configured. Did you call setConfig method?");
      return false;
    }

    // Ray origin in world coordinates
    openvdb::Vec3d ray_origin_world(origin.x(), origin.y(), origin.z());

    // Finite world coordinates can still overflow OpenVDB's int32 indices.
    openvdb::Coord ray_origin_index;
    if (!worldToIndexChecked(ray_origin_world, ray_origin_index))
    {
      logMessage(LogLevel::Error, "Ray origin is outside the finite grid coordinate range");
      return false;
    }

    // Ray end point in world coordinates
    openvdb::Vec3d ray_end_world;

    bool grid_empty              = m_vdb_grid->empty();
    typename GridT::Accessor acc = m_vdb_grid->getAccessor();

    // Raycasting of every point in the input cloud
    for (const PointT& pt : *cloud)
    {
      ray_end_world      = openvdb::Vec3d(pt.x, pt.y, pt.z);
      bool max_range_ray = false;

      // isfinite, not just isnan: lidar drivers commonly encode no-return
      // points as +/-inf. An inf point passes a NaN check, but the max-range
      // clipping below turns it into NaN ((inf - origin).unit() = NaN), which
      // Coord::floor converts to INT_MIN — and the DDA then walks ~2^31 voxels
      // toward that coordinate, allocating leaves the whole way.
      if (!std::isfinite(ray_end_world.x()) || !std::isfinite(ray_end_world.y()) ||
          !std::isfinite(ray_end_world.z()))
      {
        continue;
      }

      if (raycast_range > 0.0 && (ray_end_world - ray_origin_world).length() > raycast_range)
      {
        ray_end_world =
          ray_origin_world + (ray_end_world - ray_origin_world).unit() * raycast_range;
        max_range_ray = true;
      }

      openvdb::Coord ray_end_index;
      if (!worldToIndexChecked(ray_end_world, ray_end_index))
      {
        continue;
      }
      if (ray_clearing)
      {
        // Decide based on the provided intersector rather than m_fast_mode so a
        // caller can never reach intersector.value() with an empty optional.
        if (intersector.has_value() && intersector.value())
        {
          if (!grid_empty)
          {
            castRayIntoGridFast(
              ray_origin_index, ray_end_index, acc, update_grid_acc, intersector.value());
          }
        }
        else
        {
          castRayIntoGrid(ray_origin_index, ray_end_index, update_grid_acc);
        }
      }

      if (!max_range_ray && endpoint_hits)
      {
        update_grid_acc.setValueOn(ray_end_index, true);
      }
    }
    return true;
  }

  /*!
   * \brief Casts a single ray into an update grid structure using raycasting
   *
   * Each cell along the ray is marked as active.
   *
   * \param ray_origin_index Ray origin in index coordinates
   * \param ray_end_index Ray endpoint in index coordinates
   * \param update_grid_acc Accessor to the update grid
   */
  void castRayIntoGrid(const openvdb::Coord& ray_origin_index,
                       const openvdb::Coord& ray_end_index,
                       UpdateGridT::Accessor& update_grid_acc) const
  {
    openvdb::Vec3d ray_direction = (ray_end_index.asVec3d() - ray_origin_index);

    // Starting the ray from the center of the voxel
    RayT ray(ray_origin_index.asVec3d() + 0.5, ray_direction, 0, 1);
    DDAT dda(ray, 0);
    if (ray_end_index != ray_origin_index)
    {
      do
      {
        update_grid_acc.setActiveState(dda.voxel(), true);
      } while (dda.step());
    }
  }

  /*!
   * \brief Casts a single ray into an update grid structure using ray marching
   *
   * \param ray_origin_index Ray origin in index coordinates
   * \param ray_end_index Ray endpoint in index coordinates
   * \param grid_acc Accessor to the map grid
   * \param update_grid_acc Accessor to the update grid
   * \param intersector Volume ray intersector to determine collisions with the grid
   */
  void castRayIntoGridFast(
    const openvdb::Coord& ray_origin_index,
    const openvdb::Coord& ray_end_index,
    typename GridT::Accessor& grid_acc,
    UpdateGridT::Accessor& update_grid_acc,
    const std::shared_ptr<openvdb::tools::VolumeRayIntersector<openvdb::FloatGrid> >& intersector)
    const
  {
    openvdb::Vec3d ray_direction = (ray_end_index.asVec3d() - ray_origin_index);
    // Ray times must be strictly positive: a t0 of exactly 0 trips
    // Ray::setTimes' assertion inside the intersector when the origin lies
    // within an occupied leaf node.
    RayT ray(
      ray_origin_index.asVec3d() + 0.5, ray_direction, openvdb::math::Delta<double>::value(), 1);
    if (!intersector->setIndexRay(ray))
    {
      return;
    }
    std::vector<RayT::TimeSpan> hits;
    intersector->hits(hits);
    for (auto& hit : hits)
    {
      RayT fine_ray(ray_origin_index.asVec3d() + 0.5, ray_direction, hit.t0, hit.t1);
      DDAT dda(fine_ray);
      do
      {
        if (grid_acc.isValueOn(dda.voxel()))
        {
          update_grid_acc.setActiveState(dda.voxel(), true);
        }
      } while (dda.step());
    }
  }

  /*!
   * \brief Transforms world to index coordinates by rounding to the nearest voxel center
   *
   * \param world_coordinate Coordinate in world space
   *
   * \returns Coordinate in index space
   */
  openvdb::Coord worldToIndex(const openvdb::Vec3d& world_coordinate) const
  {
    openvdb::Vec3d index_coord = m_vdb_grid->worldToIndex(world_coordinate);
    return openvdb::Coord::floor(
      openvdb::Vec3d(index_coord.x() + 0.5, index_coord.y() + 0.5, index_coord.z() + 0.5));
  }

  // Validate external points before converting a floating-point coordinate
  // to OpenVDB's signed integer lattice. The caller holds the map lock.
  bool worldToIndexChecked(const openvdb::Vec3d& world_coordinate, openvdb::Coord& coordinate) const
  {
    const openvdb::Vec3d index = m_vdb_grid->worldToIndex(world_coordinate);
    for (int axis = 0; axis < 3; ++axis)
    {
      const double rounded = std::floor(index[axis] + 0.5);
      if (!std::isfinite(rounded) || rounded < std::numeric_limits<std::int32_t>::min() ||
          rounded > std::numeric_limits<std::int32_t>::max())
      {
        return false;
      }
      coordinate[axis] = static_cast<std::int32_t>(rounded);
    }
    return true;
  }

  /*!
   * \brief Raytraces a single ray through the map
   *
   * \param ray_origin_world Origin of the beam in world coordinates
   * \param ray_direction Direction of the beam
   * \param max_ray_length Maximum raycasting length
   * \param success Did the raycast hit an obstacle
   * \param end_point End point of the raycasting
   *
   */
  void raytrace(const openvdb::Vec3d& ray_origin_world,
                const openvdb::Vec3d& ray_direction,
                const double max_ray_lengths,
                bool& success,
                openvdb::Vec3d& end_point)
  {
    // In this instance, the single raytrace is not the base method
    // This is because if the batch raytrace would call the single raytrace it would try to always
    // get the shared lock again, which might decrease performance or lead to unexpected waits
    std::vector<openvdb::Vec3d> ray_origins_world = {ray_origin_world};
    std::vector<openvdb::Vec3d> ray_directions    = {ray_direction};
    std::vector<double> max_ray_length            = {max_ray_lengths};
    std::vector<bool> successes;
    std::vector<openvdb::Vec3d> end_points;

    raytrace(ray_origins_world, ray_directions, max_ray_length, successes, end_points);

    success   = successes[0];
    end_point = end_points[0];
  }


  /*!
   * \brief Raytraces a single ray through the map
   *
   * \param ray_origins_world Array of origins of the beams in world coordinates
   * \param ray_directions Array of directions of the beams
   * \param max_ray_lengths Array of maximum raycasting lengths
   * \param successes Array specifying if the corresponding raycast hit an obstacle
   * \param end_points Array of end point of the raycasting
   *
   */
  void raytrace(const std::vector<openvdb::Vec3d>& ray_origins_world,
                const std::vector<openvdb::Vec3d>& ray_directions,
                const std::vector<double>& max_ray_lengths,
                std::vector<bool>& successes,
                std::vector<openvdb::Vec3d>& end_points)
  {
    std::shared_lock map_lock(*m_map_mutex);

    // The volume ray intersector only exists in fast mode after an
    // integration on a non-empty grid. When available it accelerates the
    // query by restricting the fine DDA walk to leaf spans that contain
    // active voxels; without it (fast_mode off, or nothing integrated yet)
    // fall back to an exact DDA walk over the full ray instead of failing
    // the query.
    //
    // Shallow-copy the shared intersector so concurrent raytrace callers do
    // not race on setIndexRay/march, which both mutate intersector state.
    std::optional<openvdb::tools::VolumeRayIntersector<openvdb::FloatGrid> > local_intersector;
    if (m_volume_ray_intersector)
    {
      local_intersector.emplace(*m_volume_ray_intersector);
    }

    typename GridT::Accessor acc = m_vdb_grid->getAccessor();
    successes.assign(ray_origins_world.size(), false);
    end_points = ray_origins_world;

    // The three input arrays are indexed in lockstep; mismatched sizes would
    // read out of bounds below.
    if (ray_directions.size() != ray_origins_world.size() ||
        max_ray_lengths.size() != ray_origins_world.size())
    {
      logMessage(LogLevel::Error,
                 "Raytrace input arrays differ in size (origins=" +
                   std::to_string(ray_origins_world.size()) +
                   ", directions=" + std::to_string(ray_directions.size()) +
                   ", max_lengths=" + std::to_string(max_ray_lengths.size()) + ")");
      return;
    }

    for (size_t i = 0; i < ray_origins_world.size(); i++)
    {
      // Guard against degenerate queries: a non-finite origin, a zero-length
      // or non-finite direction, or a non-positive ray length would yield NaN
      // after normalize() (or a zero-direction ray) and feed NaN
      // times/coordinates into the DDA and intersector (raycastPointCloud
      // guards its inputs the same way). Report a miss anchored at the origin
      // instead of corrupting state.
      const openvdb::Vec3d& raw_origin    = ray_origins_world[i];
      const openvdb::Vec3d& raw_direction = ray_directions[i];
      if (!std::isfinite(raw_origin.x()) || !std::isfinite(raw_origin.y()) ||
          !std::isfinite(raw_origin.z()) || !std::isfinite(raw_direction.x()) ||
          !std::isfinite(raw_direction.y()) || !std::isfinite(raw_direction.z()) ||
          raw_direction.lengthSqr() <= 0.0 || !std::isfinite(max_ray_lengths[i]) ||
          max_ray_lengths[i] <= 0.0)
      {
        continue;
      }

      // Normalize direction vector just to be sure
      openvdb::Vec3d direction_norm = raw_direction;
      direction_norm.normalize();
      direction_norm *= max_ray_lengths[i];

      openvdb::Vec3d ray_origin_index    = m_vdb_grid->worldToIndex(ray_origins_world[i]);
      // Directions are vectors, not points. worldToIndex() includes the map's
      // translation and corrupts loaded grids whose transform is not anchored
      // at the origin; the inverse Jacobian applies only the linear part.
      openvdb::Vec3d ray_direction_index =
        m_vdb_grid->transform().baseMap()->applyInverseJacobian(direction_norm);

      end_points[i] = m_vdb_grid->indexToWorld(ray_origin_index + ray_direction_index);

      if (!local_intersector)
      {
        // Exact DDA over the full ray. Voxels are cell-centered
        // (worldToIndex rounds to the nearest lattice point) while the DDA
        // cell convention is [i, i+1): shift by half a voxel so dda.voxel()
        // yields cell-centered indices.
        RayT ray(ray_origin_index + openvdb::Vec3d(0.5),
                 ray_direction_index,
                 openvdb::math::Delta<double>::value(),
                 1);
        DDAT dda(ray);
        do
        {
          if (acc.isValueOn(dda.voxel()))
          {
            end_points[i] = m_vdb_grid->indexToWorld(dda.voxel());
            successes[i]  = true;
            break;
          }
        } while (dda.step());
        continue;
      }

      // Ray times must be strictly positive: a t0 of exactly 0 trips
      // Ray::setTimes' assertion inside the intersector when the origin lies
      // within an occupied leaf node. The +0.5 shift matches the cell-centered
      // voxel convention used by every other DDA in this file (worldToIndex
      // rounds to the nearest lattice point); without it this branch would
      // traverse the lattice half a voxel off and miss obstacles on the
      // geometric ray for non-axis-aligned queries.
      RayT ray(ray_origin_index + openvdb::Vec3d(0.5),
               ray_direction_index,
               openvdb::math::Delta<double>::value(),
               1);

      if (!local_intersector->setIndexRay(ray))
      {
        // Ray misses the active bounding box of the map entirely
        continue;
      }

      // march() yields one span per run of intersected leaf nodes. A span only
      // guarantees its leaves contain active voxels somewhere, not on the ray
      // itself, so keep marching until a hit is found or the ray is exhausted.
      double t0;
      double t1;
      while (!successes[i] && local_intersector->march(t0, t1))
      {
        // Same +0.5 cell-centered shift as the coarse ray above; march()'s
        // t-values are in that ray's parameterization so the fine ray must use
        // the identical shifted origin.
        RayT fine_ray(ray_origin_index + openvdb::Vec3d(0.5), ray_direction_index, t0, t1);
        DDAT dda(fine_ray);
        // Check the current voxel before stepping: the first voxel of a span
        // is a valid candidate (leaf-aligned obstacles start exactly there).
        do
        {
          if (acc.isValueOn(dda.voxel()))
          {
            end_points[i] = m_vdb_grid->indexToWorld(dda.voxel());
            successes[i]  = true;
            break;
          }
        } while (dda.step());
      }
    }
  }

  /*!
   * \brief Incorporates the information of an update grid to the internal map. This will update the
   * probabilities of all cells specified by the update grid.
   *
   * \param temp_grid Grid containing all cells which shall be updated
   *
   * \returns Returns a grid containing all changed voxel
   */
  UpdateGridT::Ptr updateMap(const UpdateGridT::Ptr& temp_grid)
  {
    UpdateGridT::Ptr change          = UpdateGridT::create(false);
    UpdateGridT::Accessor change_acc = change->getAccessor();
    if (temp_grid->empty())
    {
      return change;
    }

    bool state_changed           = false;
    typename GridT::Accessor acc = m_vdb_grid->getAccessor();
    // Probability update lambda for free space grid elements
    auto miss = [&](TData& voxel_value, bool& active) {
      bool last_state = active;
      updateFreeNode(voxel_value, active);
      if (last_state != active)
      {
        state_changed = true;
      }
    };

    // Probability update lambda for occupied grid elements
    auto hit = [&](TData& voxel_value, bool& active) {
      bool last_state = active;
      updateOccupiedNode(voxel_value, active);
      if (last_state != active)
      {
        state_changed = true;
      }
    };

    // Integrating the data of the temporary grid into the map using the probability update
    // functions
    for (UpdateGridT::ValueOnCIter iter = temp_grid->cbeginValueOn(); iter; ++iter)
    {
      state_changed = false;
      if (*iter)
      {
        acc.modifyValueAndActiveState(iter.getCoord(), hit);
        if (state_changed)
        {
          change_acc.setValueOn(iter.getCoord(), true);
        }
      }
      else
      {
        acc.modifyValueAndActiveState(iter.getCoord(), miss);
        if (state_changed)
        {
          change_acc.setActiveState(iter.getCoord(), true);
        }
      }
    }

    // Set all artificial values to active
    for (UpdateGridT::ValueOnCIter iter = m_artificial_area_grid->cbeginValueOn(); iter; ++iter)
    {
      acc.setActiveState(iter.getCoord(), true);
    }

    return change;
  }

  /*!
   * \brief Returns a pointer to the VDB map structure
   *
   * \returns Map pointer
   */
  typename GridT::Ptr getGrid() const
  {
    // Synchronise the shared_ptr read against resetMap/loadMap, which rebind
    // m_vdb_grid under the unique lock (an unguarded copy concurrent with the
    // rebind is a data race on the control block). NOTE: this only makes
    // fetching the handle safe; the integration thread keeps mutating the
    // returned grid's tree, so callers reading it concurrently must fetch the
    // handle FIRST and then hold getMapMutex() in shared mode while reading.
    // (Do not lock before calling getGrid(): that would recursively acquire
    // the non-recursive shared_mutex from the same thread.)
    std::shared_lock map_lock(*m_map_mutex);
    return m_vdb_grid;
  }

  /*!
   * \brief Returns the resolution of the map
   *
   * \returns Resolution
   */
  double getResolution() const { return m_resolution; }

  /*!
   * \brief Creates a world coordinate bounding box around a transform
   *
   * \param min_boundary Minimum boundary of box
   * \param max_boundary Maximum boundary of box
   * \param map_to_reference_tf Transform from map to reference frame
   *
   * \returns World coordinate bounding box
   */
  openvdb::BBoxd
  createWorldBoundingBox(const Eigen::Matrix<double, 3, 1>& min_boundary,
                         const Eigen::Matrix<double, 3, 1>& max_boundary,
                         const Eigen::Matrix<double, 4, 4>& map_to_reference_tf) const
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr corners(new pcl::PointCloud<pcl::PointXYZ>());
    corners->points.emplace_back(static_cast<float>(min_boundary.x()),
                                 static_cast<float>(min_boundary.y()),
                                 static_cast<float>(min_boundary.z()));
    corners->points.emplace_back(static_cast<float>(min_boundary.x()),
                                 static_cast<float>(min_boundary.y()),
                                 static_cast<float>(max_boundary.z()));
    corners->points.emplace_back(static_cast<float>(min_boundary.x()),
                                 static_cast<float>(max_boundary.y()),
                                 static_cast<float>(min_boundary.z()));
    corners->points.emplace_back(static_cast<float>(min_boundary.x()),
                                 static_cast<float>(max_boundary.y()),
                                 static_cast<float>(max_boundary.z()));
    corners->points.emplace_back(static_cast<float>(max_boundary.x()),
                                 static_cast<float>(min_boundary.y()),
                                 static_cast<float>(min_boundary.z()));
    corners->points.emplace_back(static_cast<float>(max_boundary.x()),
                                 static_cast<float>(min_boundary.y()),
                                 static_cast<float>(max_boundary.z()));
    corners->points.emplace_back(static_cast<float>(max_boundary.x()),
                                 static_cast<float>(max_boundary.y()),
                                 static_cast<float>(min_boundary.z()));
    corners->points.emplace_back(static_cast<float>(max_boundary.x()),
                                 static_cast<float>(max_boundary.y()),
                                 static_cast<float>(max_boundary.z()));
    pcl::transformPointCloud(*corners, *corners, map_to_reference_tf);
    pcl::PointXYZ min_pt;
    pcl::PointXYZ max_pt;
    pcl::getMinMax3D(*corners, min_pt, max_pt);

    return openvdb::BBoxd(openvdb::Vec3d(min_pt.x, min_pt.y, min_pt.z),
                          openvdb::Vec3d(max_pt.x, max_pt.y, max_pt.z));
  }
  /*!
   * \brief Creates an index coordinate bounding box around a transform
   *
   * \param min_boundary Minimum boundary of box
   * \param max_boundary Maximum boundary of box
   * \param map_to_reference_tf Transform from map to reference frame
   *
   * \returns Index coordinate bounding box
   */
  openvdb::CoordBBox
  createIndexBoundingBox(const Eigen::Matrix<double, 3, 1>& min_boundary,
                         const Eigen::Matrix<double, 3, 1>& max_boundary,
                         const Eigen::Matrix<double, 4, 4>& map_to_reference_tf) const
  {
    openvdb::BBoxd world_bb =
      createWorldBoundingBox(min_boundary, max_boundary, map_to_reference_tf);

    std::shared_lock map_lock(*m_map_mutex);
    openvdb::Vec3d min_index = m_vdb_grid->worldToIndex(world_bb.min());
    openvdb::Vec3d max_index = m_vdb_grid->worldToIndex(world_bb.max());
    map_lock.unlock();

    return {openvdb::Coord::floor(min_index), openvdb::Coord::floor(max_index)};
  }

  /*!
   * \brief Generates an update grid from a bounding box and a reference frame
   *
   * \param min_boundary Minimum boundary of the box
   * \param max_boundary Maximum boundary of the box
   * \param map_to_reference_tf Transform from map to reference frame
   * \param full_grid Specifies whether the entire grid or just the active values should be return
   *
   * \returns Update Grid containing the information within the bounding box
   */
  typename UpdateGridT::Ptr
  getMapSectionUpdateGrid(const Eigen::Matrix<double, 3, 1>& min_boundary,
                          const Eigen::Matrix<double, 3, 1>& max_boundary,
                          const Eigen::Matrix<double, 4, 4>& map_to_reference_tf,
                          const bool full_grid = false) const
  {
    return getMapSection<typename VDBMapping<TData, TConfig, PointT>::UpdateGridT>(
      min_boundary, max_boundary, map_to_reference_tf, full_grid);
  }
  /*!
   * \brief Generates a grid from a bounding box and a reference frame
   *
   * \param min_boundary Minimum boundary of the box
   * \param max_boundary Maximum boundary of the box
   * \param map_to_reference_tf Transform from map to reference frame
   * \param full_grid Specifies whether the entire grid or just the active values should be return
   *
   * \returns Grid containing the information within the bounding box
   */
  typename GridT::Ptr getMapSectionGrid(const Eigen::Matrix<double, 3, 1>& min_boundary,
                                        const Eigen::Matrix<double, 3, 1>& max_boundary,
                                        const Eigen::Matrix<double, 4, 4>& map_to_reference_tf,
                                        const bool full_grid = false) const
  {
    return getMapSection<typename VDBMapping<TData, TConfig, PointT>::GridT>(
      min_boundary, max_boundary, map_to_reference_tf, full_grid);
  }
  /*!
   * \brief Generates a grid or update grid from a bounding box and a reference frame
   *
   * @tparam TResultGrid Resulting Grid Type
   * \param min_boundary Minimum boundary of the box
   * \param max_boundary Maximum boundary of the box
   * \param map_to_reference_tf Transform from map to reference frame
   * \param full_grid Specifies whether the entire grid or just the active values should be return
   *
   * \returns Grid/UpdateGrid containing the information within the bounding box
   */
  template <typename TResultGrid>
  typename TResultGrid::Ptr getMapSection(const Eigen::Matrix<double, 3, 1>& min_boundary,
                                          const Eigen::Matrix<double, 3, 1>& max_boundary,
                                          const Eigen::Matrix<double, 4, 4>& map_to_reference_tf,
                                          const bool full_grid = false) const
  {
    typename TResultGrid::Ptr temp_grid = TResultGrid::create(0);
    temp_grid->setTransform(openvdb::math::Transform::createLinearTransform(m_resolution));

    typename TResultGrid::Accessor temp_acc = temp_grid->getAccessor();

    openvdb::CoordBBox bounding_box(
      createIndexBoundingBox(min_boundary, max_boundary, map_to_reference_tf));

    std::shared_lock map_lock(*m_map_mutex);
    for (auto leaf_iter = m_vdb_grid->tree().cbeginLeaf(); leaf_iter; ++leaf_iter)
    {
      openvdb::CoordBBox bbox;
      bbox = leaf_iter.getLeaf()->getNodeBoundingBox();

      if (bbox.hasOverlap(bounding_box))
      {
        if (full_grid)
        {
          extractFullLeaf<TResultGrid>(temp_acc, bounding_box, leaf_iter);
        }
        else
        {
          extractSparseLeaf<TResultGrid>(temp_acc, bounding_box, leaf_iter);
        }
      }
    }

    // Pruned constant regions live as TILES at internal nodes (integrateUpdate
    // calls pruneGrid every cycle), which the leaf iteration above misses.
    // Iterate observed (non-background) tiles overlapping the region and fill
    // their bbox-clamped extent with the same full/sparse semantics as the leaf
    // helpers — otherwise large free/occupied volumes are dropped from the
    // serialized section and a remote peer applying it gets holes.
    {
      const typename GridT::ValueType background = m_vdb_grid->background();
      auto tile_iter = m_vdb_grid->tree().cbeginValueAll();
      tile_iter.setMaxDepth(GridT::TreeType::DEPTH - 2);  // tiles, not voxels
      for (; tile_iter; ++tile_iter)
      {
        if (tile_iter.getValue() == background)
        {
          continue;  // unobserved background — nothing to serialize
        }
        const bool tile_on = tile_iter.isValueOn();
        if (!full_grid && !tile_on)
        {
          continue;  // sparse mode extracts active values only (extractSparseLeaf)
        }
        openvdb::CoordBBox tile_bbox;
        tile_iter.getBoundingBox(tile_bbox);
        tile_bbox.intersect(bounding_box);
        if (tile_bbox.empty())
        {
          continue;
        }
        const typename GridT::ValueType val = tile_iter.getValue();
        for (int z = tile_bbox.min().z(); z <= tile_bbox.max().z(); ++z)
        {
          for (int y = tile_bbox.min().y(); y <= tile_bbox.max().y(); ++y)
          {
            for (int x = tile_bbox.min().x(); x <= tile_bbox.max().x(); ++x)
            {
              const openvdb::Coord c(x, y, z);
              if (full_grid)
              {
                tile_on ? temp_acc.setValueOn(c, val) : temp_acc.setValueOff(c, val);
              }
              else if constexpr (std::is_same_v<typename TResultGrid::ValueType, bool>)
              {
                temp_acc.setValueOn(c, true);
              }
              else
              {
                temp_acc.setValueOn(c, static_cast<typename TResultGrid::ValueType>(val));
              }
            }
          }
        }
      }
    }
    map_lock.unlock();

    openvdb::Vec3d min(bounding_box.min().x(), bounding_box.min().y(), bounding_box.min().z());
    openvdb::Vec3d max(bounding_box.max().x(), bounding_box.max().y(), bounding_box.max().z());
    temp_grid->insertMeta("bb_min", openvdb::Vec3DMetadata(min));
    temp_grid->insertMeta("bb_max", openvdb::Vec3DMetadata(max));
    return temp_grid;
  }

  /*!
   * \brief Extracts all voxel inside of a bounding box from a leaf
   *
   * @tparam TResultGrid Resulting Grid Type
   * \param temp_acc Accessor to the resulting grid
   * \param bounding_box Bounding box that should be extracted
   * \param leaf Reference to the leaf iterator
   */
  template <typename TResultGrid>
  void extractFullLeaf(typename TResultGrid::Accessor& temp_acc,
                       const openvdb::CoordBBox& bounding_box,
                       const typename GridT::TreeType::LeafCIter& leaf) const
  {
    for (auto iter = leaf->cbeginValueAll(); iter; ++iter)
    {
      if (bounding_box.isInside(iter.getCoord()))
      {
        if (iter.isValueOn())
        {
          temp_acc.setValueOn(iter.getCoord(), iter.getValue());
        }
        else
        {
          temp_acc.setValueOff(iter.getCoord(), iter.getValue());
        }
      }
    }
  }

  /*!
   * \brief Extracts all active voxel inside of a bounding box from a leaf
   *
   * @tparam TResultGrid Resulting Grid Type
   * \param temp_acc Accessor to the resulting grid
   * \param bounding_box Bounding box that should be extracted
   * \param leaf Reference to the leaf iterator
   */
  template <typename TResultGrid>
  void extractSparseLeaf(typename TResultGrid::Accessor& temp_acc,
                         const openvdb::CoordBBox& bounding_box,
                         const typename GridT::TreeType::LeafCIter& leaf) const
  {
    for (auto iter = leaf->cbeginValueOn(); iter; ++iter)
    {
      if (bounding_box.isInside(iter.getCoord()))
      {
        if constexpr (std::is_same_v<typename TResultGrid::ValueType, bool>)
        {
          temp_acc.setValueOn(iter.getCoord(), true);
        }
        else
        {
          // Preserve the stored value (e.g. log-odds) instead of flattening
          // every active voxel to 1.0
          temp_acc.setValueOn(iter.getCoord(),
                              static_cast<typename TResultGrid::ValueType>(iter.getValue()));
        }
      }
    }
  }

  /*!
   * \brief Applies a map section grid to the map
   *
   * \param section Section grid containing the information about part of the map. The boundary box
   * of the section is encoded in the grids meta information
   * \param smooth_map Specifies if the section should be morphologically smoothed
   * \param smoothing_iterations Amount of morphological smoothing iterations
   *
   */
  void applyMapSectionGrid(const typename GridT::Ptr section,
                           bool smooth_map          = false,
                           int smoothing_iterations = 2)
  {
    if (smooth_map)
    {
      morphologicalCloseMap<GridT>(section, smoothing_iterations);
    }

    typename GridT::Accessor section_acc = section->getAccessor();
    std::unique_lock map_lock(*m_map_mutex);
    typename GridT::Accessor acc = m_vdb_grid->getAccessor();
    for (auto iter = section->cbeginValueAll(); iter; ++iter)
    {
      // Allocated leaves/internal nodes can expose inactive background tiles
      // through ValueAll. They carry no section information and their node
      // bounding boxes may span billions of implicit voxels, so never expand
      // them. Explicit observed-free values are non-background and remain.
      if (!iter.isValueOn() && iter.getValue() == section->background())
      {
        continue;
      }
      const auto apply_coord = [&](const openvdb::Coord& coord) {
        if (section_acc.isValueOn(coord))
        {
          acc.setValueOn(coord, section_acc.getValue(coord));
        }
        else
        {
          acc.setValueOff(coord, section_acc.getValue(coord));
        }
      };
      if (iter.isVoxelValue())
      {
        apply_coord(iter.getCoord());
      }
      else
      {
        openvdb::CoordBBox tile_bbox;
        iter.getBoundingBox(tile_bbox);
        for (auto coord_iter = tile_bbox.begin(); coord_iter != tile_bbox.end(); ++coord_iter)
        {
          apply_coord(*coord_iter);
        }
      }
    }
    // The map was mutated in place; drop intersectors built from the stale
    // topology (the next integration rebuilds them).
    resetVolumeRayIntersectors();
    map_lock.unlock();
  }

  void transformAndApplyMapSectionGrid(const typename GridT::Ptr section,
                                       const Eigen::Matrix<double, 4, 4>& transform,
                                       bool smooth_map          = false,
                                       int smoothing_iterations = 2)
  {
    if (!transform.allFinite() ||
        std::fabs(transform.template block<3, 3>(0, 0).determinant()) < 1e-12)
    {
      logMessage(LogLevel::Error,
                 "Cannot apply map section with a non-finite or singular transform");
      return;
    }
    if (smooth_map)
    {
      morphologicalCloseMap<GridT>(section, smoothing_iterations);
    }

    typename GridT::Accessor section_acc = section->getAccessor();
    std::unique_lock map_lock(*m_map_mutex);
    typename GridT::Accessor acc = m_vdb_grid->getAccessor();
    for (auto iter = section->cbeginValueAll(); iter; ++iter)
    {
      if (!iter.isValueOn() && iter.getValue() == section->background())
      {
        continue;
      }
      const auto apply_coord = [&](const openvdb::Coord& source_coord) {
        openvdb::Vec3d buffer = section->indexToWorld(source_coord);
        Eigen::Matrix<double, 4, 1> eigen_world(buffer.x(), buffer.y(), buffer.z(), 1.0);
        eigen_world = transform * eigen_world;
        buffer      = openvdb::Vec3d(eigen_world.x(), eigen_world.y(), eigen_world.z());
        // The transformed point is in the destination map frame. Converting
        // it with section->worldToIndex() applied the source transform twice
        // and also failed whenever source and destination resolutions differ.
        buffer                                 = m_vdb_grid->worldToIndex(buffer);
        const openvdb::Coord destination_coord = openvdb::Coord::floor(
          openvdb::Vec3d(buffer.x() + 0.5, buffer.y() + 0.5, buffer.z() + 0.5));
        // Read state/value at the original source coordinate, not at the
        // transformed destination coordinate.
        if (section_acc.isValueOn(source_coord))
        {
          acc.setValueOn(destination_coord, section_acc.getValue(source_coord));
        }
        else
        {
          acc.setValueOff(destination_coord, section_acc.getValue(source_coord));
        }
      };
      if (iter.isVoxelValue())
      {
        apply_coord(iter.getCoord());
      }
      else
      {
        openvdb::CoordBBox tile_bbox;
        iter.getBoundingBox(tile_bbox);
        for (auto coord_iter = tile_bbox.begin(); coord_iter != tile_bbox.end(); ++coord_iter)
        {
          apply_coord(*coord_iter);
        }
      }
    }
    // The map was mutated in place; drop intersectors built from the stale
    // topology (the next integration rebuilds them).
    resetVolumeRayIntersectors();
    map_lock.unlock();
  }

  /*!
   * \brief Applies a map section update grid to the map
   *
   * \param section Section grid containing the information about part of the map. The boundary box
   * of the section is encoded in the grids meta information
   * \param smooth_map Specifies if the section should be morphologically smoothed
   * \param smoothing_iterations Amount of morphological smoothing iterations
   *
   */
  void applyMapSectionUpdateGrid(const typename UpdateGridT::Ptr section,
                                 bool smooth_map          = false,
                                 int smoothing_iterations = 2)
  {
    if (smooth_map)
    {
      morphologicalCloseMap<UpdateGridT>(section, smoothing_iterations);
    }
    typename UpdateGridT::Accessor section_acc = section->getAccessor();
    std::unique_lock map_lock(*m_map_mutex);
    typename GridT::Accessor acc = m_vdb_grid->getAccessor();

    // bb_min/bb_max are written by getMapSection. A hand-built section, or one
    // that lost its metadata in a transport round trip, will not carry them;
    // metaValue throws on a missing key, which would otherwise abort this call
    // (with the map lock held) on an opaque OpenVDB error. Fall back to the
    // section's own active bounding box for the cleared region instead.
    auto bb_min_meta = section->template getMetadata<openvdb::Vec3DMetadata>("bb_min");
    auto bb_max_meta = section->template getMetadata<openvdb::Vec3DMetadata>("bb_max");
    openvdb::CoordBBox bbox;
    if (bb_min_meta && bb_max_meta)
    {
      bbox = openvdb::CoordBBox(openvdb::Coord::floor(bb_min_meta->value()),
                                openvdb::Coord::floor(bb_max_meta->value()));
    }
    else
    {
      logMessage(LogLevel::Warning,
                 "Map section update grid is missing bb_min/bb_max metadata; falling back to its "
                 "active bounding box for the cleared region");
      bbox = section->evalActiveVoxelBoundingBox();
    }

    // Walk only leaves that overlap the section bbox instead of every active
    // voxel in the entire map.
    for (auto leaf_iter = m_vdb_grid->tree().cbeginLeaf(); leaf_iter; ++leaf_iter)
    {
      if (!leaf_iter.getLeaf()->getNodeBoundingBox().hasOverlap(bbox))
      {
        continue;
      }
      for (auto iter = leaf_iter->cbeginValueOn(); iter; ++iter)
      {
        if (bbox.isInside(iter.getCoord()))
        {
          acc.setActiveState(iter.getCoord(), false);
        }
      }
    }
    // Integration prunes uniform leaves into active TILES every cycle, and
    // tiles are invisible to the leaf walk above — without this pass a
    // pruned occupied block inside the section could never be cleared by a
    // peer update. Deactivating through the accessor densifies the tile
    // back into voxels for exactly the overlap region.
    auto tile_iter = m_vdb_grid->tree().cbeginValueOn();
    tile_iter.setMaxDepth(GridT::TreeType::DEPTH - 2);
    for (; tile_iter; ++tile_iter)
    {
      if (!tile_iter.isTileValue())
      {
        continue;
      }
      openvdb::CoordBBox tile_bbox;
      tile_iter.getBoundingBox(tile_bbox);
      if (!tile_bbox.hasOverlap(bbox))
      {
        continue;
      }
      tile_bbox.intersect(bbox);
      for (auto coord_iter = tile_bbox.begin(); coord_iter != tile_bbox.end(); ++coord_iter)
      {
        acc.setActiveState(*coord_iter, false);
      }
    }
    for (auto iter = section->cbeginValueOn(); iter; ++iter)
    {
      if (iter.isVoxelValue())
      {
        acc.setActiveState(iter.getCoord(), true);
        continue;
      }
      openvdb::CoordBBox tile_bbox;
      iter.getBoundingBox(tile_bbox);
      for (auto coord_iter = tile_bbox.begin(); coord_iter != tile_bbox.end(); ++coord_iter)
      {
        acc.setActiveState(*coord_iter, true);
      }
    }
    // The map was mutated in place; drop intersectors built from the stale
    // topology (the next integration rebuilds them).
    resetVolumeRayIntersectors();
    map_lock.unlock();
  }

  /*!
   * \brief Applies a map section update grid to the map after transforming it
   *
   * This variant has the same complete-section semantics as
   * applyMapSectionUpdateGrid: destination voxels covered by the transformed
   * source bounding box are cleared before active source voxels are applied.
   *
   * \param section Section update grid to apply
   * \param transform Transform applied to the section before merging
   * \param smooth_map Specifies if the section should be morphologically smoothed
   * \param smoothing_iterations Amount of morphological smoothing iterations
   */
  void transformAndApplyMapSectionUpdateGrid(const typename UpdateGridT::Ptr section,
                                             const Eigen::Matrix<double, 4, 4>& transform,
                                             bool smooth_map          = false,
                                             int smoothing_iterations = 2)
  {
    if (!transform.allFinite() ||
        std::fabs(transform.template block<3, 3>(0, 0).determinant()) < 1e-12)
    {
      logMessage(LogLevel::Error,
                 "Cannot apply map section update with a non-finite or singular transform");
      return;
    }
    if (smooth_map)
    {
      morphologicalCloseMap<UpdateGridT>(section, smoothing_iterations);
    }
    std::unique_lock map_lock(*m_map_mutex);
    typename GridT::Accessor acc = m_vdb_grid->getAccessor();

    auto bb_min_meta = section->template getMetadata<openvdb::Vec3DMetadata>("bb_min");
    auto bb_max_meta = section->template getMetadata<openvdb::Vec3DMetadata>("bb_max");
    openvdb::CoordBBox source_bbox;
    if (bb_min_meta && bb_max_meta)
    {
      source_bbox = openvdb::CoordBBox(openvdb::Coord::floor(bb_min_meta->value()),
                                       openvdb::Coord::floor(bb_max_meta->value()));
    }
    else
    {
      logMessage(LogLevel::Warning,
                 "Transformed map section update is missing bb_min/bb_max metadata; "
                 "falling back to its active bounding box for the cleared region");
      source_bbox = section->evalActiveVoxelBoundingBox();
    }

    if (!source_bbox.empty())
    {
      // Find an index-space AABB containing the transformed section, then
      // inverse-sample its voxel centres. The inside test avoids clearing the
      // unused corners of that AABB when the section is rotated.
      openvdb::Vec3d destination_min(std::numeric_limits<double>::infinity(),
                                     std::numeric_limits<double>::infinity(),
                                     std::numeric_limits<double>::infinity());
      openvdb::Vec3d destination_max(-std::numeric_limits<double>::infinity(),
                                     -std::numeric_limits<double>::infinity(),
                                     -std::numeric_limits<double>::infinity());
      for (int x_side = 0; x_side < 2; ++x_side)
      {
        for (int y_side = 0; y_side < 2; ++y_side)
        {
          for (int z_side = 0; z_side < 2; ++z_side)
          {
            const openvdb::Coord source_corner(
              x_side ? source_bbox.max().x() : source_bbox.min().x(),
              y_side ? source_bbox.max().y() : source_bbox.min().y(),
              z_side ? source_bbox.max().z() : source_bbox.min().z());
            const openvdb::Vec3d source_world = section->indexToWorld(source_corner);
            const Eigen::Matrix<double, 4, 1> destination_world =
              transform * Eigen::Matrix<double, 4, 1>(
                            source_world.x(), source_world.y(), source_world.z(), 1.0);
            const openvdb::Vec3d destination_index = m_vdb_grid->worldToIndex(
              openvdb::Vec3d(destination_world.x(), destination_world.y(), destination_world.z()));
            for (int axis = 0; axis < 3; ++axis)
            {
              destination_min[axis] = std::min(destination_min[axis], destination_index[axis]);
              destination_max[axis] = std::max(destination_max[axis], destination_index[axis]);
            }
          }
        }
      }
      const openvdb::CoordBBox destination_bbox(
        openvdb::Coord::floor(destination_min - openvdb::Vec3d(1.0)),
        openvdb::Coord::ceil(destination_max + openvdb::Vec3d(1.0)));
      const Eigen::Matrix<double, 4, 4> inverse_transform = transform.inverse();
      for (auto coord_iter = destination_bbox.begin(); coord_iter != destination_bbox.end();
           ++coord_iter)
      {
        const openvdb::Vec3d destination_world = m_vdb_grid->indexToWorld(*coord_iter);
        const Eigen::Matrix<double, 4, 1> source_world =
          inverse_transform *
          Eigen::Matrix<double, 4, 1>(
            destination_world.x(), destination_world.y(), destination_world.z(), 1.0);
        const openvdb::Vec3d source_index = section->worldToIndex(
          openvdb::Vec3d(source_world.x(), source_world.y(), source_world.z()));
        const openvdb::Coord source_coord = openvdb::Coord::floor(
          openvdb::Vec3d(source_index.x() + 0.5, source_index.y() + 0.5, source_index.z() + 0.5));
        if (source_bbox.isInside(source_coord))
        {
          acc.setActiveState(*coord_iter, false);
        }
      }
    }

    for (auto iter = section->cbeginValueOn(); iter; ++iter)
    {
      const auto activate_coord = [&](const openvdb::Coord& source_coord) {
        openvdb::Vec3d buffer = section->indexToWorld(source_coord);
        Eigen::Matrix<double, 4, 1> eigen_world(buffer.x(), buffer.y(), buffer.z(), 1.0);
        eigen_world = transform * eigen_world;
        buffer      = m_vdb_grid->worldToIndex(
          openvdb::Vec3d(eigen_world.x(), eigen_world.y(), eigen_world.z()));
        acc.setActiveState(openvdb::Coord::floor(
                             openvdb::Vec3d(buffer.x() + 0.5, buffer.y() + 0.5, buffer.z() + 0.5)),
                           true);
      };
      if (iter.isVoxelValue())
      {
        activate_coord(iter.getCoord());
      }
      else
      {
        openvdb::CoordBBox tile_bbox;
        iter.getBoundingBox(tile_bbox);
        for (auto coord_iter = tile_bbox.begin(); coord_iter != tile_bbox.end(); ++coord_iter)
        {
          activate_coord(*coord_iter);
        }
      }
    }
    // The map was mutated in place; drop intersectors built from the stale
    // topology (the next integration rebuilds them).
    resetVolumeRayIntersectors();
    map_lock.unlock();
  }

  /*!
   * \brief Morphological closing operator
   *
   * @tparam TGrid Grid type
   * \param grid Grid that should be morphologically processed
   * \param iterations Amount of morphological iterations
   */
  template <typename TGrid>
  void morphologicalCloseMap(typename TGrid::Ptr grid, int iterations)
  {
    morphologicalDilateMap<TGrid>(grid, iterations);
    morphologicalErodeMap<TGrid>(grid, iterations);
  }

  /*!
   * \brief Morphological opening operator
   *
   * @tparam TGrid Grid type
   * \param grid Grid that should be morphologically processed
   * \param iterations Amount of morphological iterations
   */
  template <typename TGrid>
  void morphologicalOpenMap(typename TGrid::Ptr grid, int iterations)
  {
    morphologicalErodeMap<TGrid>(grid, iterations);
    morphologicalDilateMap<TGrid>(grid, iterations);
  }

  /*!
   * \brief Morphological dilation operator
   *
   * @tparam TGrid Grid type
   * \param grid Grid that should be morphologically processed
   * \param iterations Amount of morphological iterations
   */
  template <typename TGrid>
  void morphologicalDilateMap(typename TGrid::Ptr grid, int iterations)
  {
    openvdb::tools::dilateActiveValues(grid->tree(),
                                       iterations,
                                       openvdb::tools::NN_FACE_EDGE_VERTEX,
                                       openvdb::tools::EXPAND_TILES,
                                       true);
  }

  /*!
   * \brief Morphological erosion operator
   *
   * @tparam TGrid Grid type
   * \param grid Grid that should be morphologically processed
   * \param iterations Amount of morphological iterations
   */
  template <typename TGrid>
  void morphologicalErodeMap(typename TGrid::Ptr grid, int iterations)
  {
    openvdb::tools::erodeActiveValues(grid->tree(),
                                      iterations,
                                      openvdb::tools::NN_FACE_EDGE_VERTEX,
                                      openvdb::tools::EXPAND_TILES,
                                      true);
  }

  /*!
   * \brief Helper function to restore the map integrity after artificial data has been inserted
   */
  void restoreMapIntegrity()
  {
    std::unique_lock map_lock(*m_map_mutex);
    restoreMapIntegrityLocked();
  }

  /*!
   * \brief Adds a set of artificial Areas to the map
   *
   * \param artificial_areas Array of area polygons
   * \param negative_height Specifies how low the polygon should be projected
   * \param positive_height Specifies how high the polygon should be projected
   */
  void addArtificialAreas(
    const std::vector<std::vector<Eigen::Matrix<double, 4, 1> > >& artificial_areas,
    const double negative_height,
    const double positive_height)
  {
    // The integration thread reads m_artificial_area_grid under the unique map
    // lock; service callers must serialise with it through the same lock.
    std::unique_lock map_lock(*m_map_mutex);
    restoreMapIntegrityLocked();
    m_artificial_areas_present = true;

    UpdateGridT::Accessor artificial_acc = m_artificial_area_grid->getAccessor();
    for (const auto& artificial_area : artificial_areas)
    {
      for (size_t i = 0; i < artificial_area.size(); i++)
      {
        addArtificialWallLocked(artificial_area[i],
                                artificial_area[(i + 1) % artificial_area.size()],
                                negative_height,
                                positive_height,
                                artificial_acc);
      }
    }
  }

protected:
  void restoreMapIntegrityLocked()
  {
    auto restore_state = [&](TData& voxel_value, bool& active) {
      setNodeState(voxel_value, active);
    };
    typename GridT::Accessor acc = m_vdb_grid->getAccessor();
    for (auto iter = m_artificial_area_grid->cbeginValueOn(); iter; ++iter)
    {
      acc.modifyValueAndActiveState(iter.getCoord(), restore_state);
    }
    m_artificial_area_grid->clear();
    m_artificial_areas_present = false;
    // The map was mutated in place; drop intersectors built from the stale
    // topology (the next integration rebuilds them).
    resetVolumeRayIntersectors();
  }

  void addArtificialWallLocked(const Eigen::Matrix<double, 4, 1>& start,
                               const Eigen::Matrix<double, 4, 1>& end,
                               const double negative_height,
                               const double positive_height,
                               UpdateGridT::Accessor& artificial_area_grid_acc)
  {
    openvdb::Coord start_index =
      this->worldToIndex(openvdb::Vec3d(start.x(), start.y(), start.z()));
    openvdb::Coord end_index = this->worldToIndex(openvdb::Vec3d(end.x(), end.y(), end.z()));

    // floor/ceil instead of truncation so negative heights round downwards
    // and the requested positive height is fully covered
    int negative_index = static_cast<int>(std::floor(negative_height / m_resolution));
    int positive_index = static_cast<int>(std::ceil(positive_height / m_resolution));

    for (int i = negative_index; i < positive_index; i++)
    {
      castRayIntoGrid(start_index + openvdb::Coord(0, 0, i),
                      end_index + openvdb::Coord(0, 0, i),
                      artificial_area_grid_acc);
    }
  }

public:
  /*!
   * \brief Compresses a string as a byte array
   *
   * \param string Input string
   *
   * \returns Compressed byte array
   */
  std::vector<uint8_t> compressString(const std::string& string) const
  {
    const std::size_t max_bytes = m_max_serialized_grid_bytes.load(std::memory_order_acquire);
    if (string.size() > max_bytes)
    {
      logMessage(LogLevel::Error,
                 "Serialized grid exceeds max_serialized_grid_bytes; refusing to encode it");
      return {};
    }
    // Cap the destination allocation too. An incompressible input at exactly
    // the configured limit can require a few bytes more than its source;
    // that payload is rejected instead of quietly exceeding the contract.
    const size_t len = std::min(ZSTD_compressBound(string.size()), max_bytes);
    std::vector<uint8_t> compressed(len);

    // Compress directly from the string. Materializing a second uncompressed
    // vector doubled peak memory during large section serialization.
    size_t ret =
      ZSTD_compress(compressed.data(), len, string.data(), string.size(), m_compression_level);


    if (ZSTD_isError(ret))
    {
      logMessage(LogLevel::Error,
                 std::string("Compression using ZSTD failed: ") + ZSTD_getErrorName(ret) +
                   " , sending uncompressed byte array");
      return std::vector<uint8_t>(string.begin(), string.end());
    }

    // Resize compressed buffer to actual compressed size
    compressed.resize(ret);
    return compressed;
  }

  /*!
   * \brief Decompresses a byte array into a string
   *
   * \param byte_array Input byte array
   *
   * returns Decompressed string
   */
  std::string decompressByteArray(const std::vector<uint8_t>& byte_array) const
  {
    const std::size_t max_bytes = m_max_serialized_grid_bytes.load(std::memory_order_acquire);
    if (byte_array.empty())
    {
      return {};
    }
    if (byte_array.size() > max_bytes)
    {
      logMessage(LogLevel::Error,
                 "Serialized grid input exceeds max_serialized_grid_bytes; rejecting it");
      return {};
    }
    // ZSTD_getDecompressedSize was deprecated in favour of
    // ZSTD_getFrameContentSize, which signals "unknown" / "error" via
    // sentinel values instead of returning 0. Without the check below a
    // corrupted or non-zstd buffer would allocate (size_t)-1 bytes and OOM
    // the process.
    const unsigned long long frame_len =
      ZSTD_getFrameContentSize(byte_array.data(), byte_array.size());

    std::string map_str;
    if (frame_len == ZSTD_CONTENTSIZE_ERROR || frame_len == ZSTD_CONTENTSIZE_UNKNOWN)
    {
      logMessage(LogLevel::Error,
                 "Could not determine decompressed size (frame not zstd or missing "
                 "size header); returning raw data");
      return std::string(byte_array.begin(), byte_array.end());
    }

    if (frame_len > static_cast<unsigned long long>(max_bytes) ||
        frame_len > static_cast<unsigned long long>(std::numeric_limits<std::size_t>::max()))
    {
      logMessage(LogLevel::Error,
                 "Zstd frame declares a grid larger than max_serialized_grid_bytes; rejecting it");
      return {};
    }

    const std::size_t len = static_cast<std::size_t>(frame_len);
    std::vector<uint8_t> uncompressed(len);

    std::size_t size =
      ZSTD_decompress(uncompressed.data(), len, byte_array.data(), byte_array.size());

    if (ZSTD_isError(size))
    {
      logMessage(LogLevel::Error,
                 std::string("Could not decompress map using ZSTD failed: ") +
                   ZSTD_getErrorName(size) + " , returning raw data");
      map_str = std::string(byte_array.begin(), byte_array.end());
    }
    else
    {
      map_str = std::string(uncompressed.begin(), uncompressed.begin() + size);
    }

    return map_str;
  }

  /*!
   * \brief Converts a grid into a byte array
   *
   * @tparam TGrid Grid Type
   * \param grid Input grid
   *
   * \returns Compressed byte array representation of the grid
   */
  template <typename TGrid>
  std::vector<uint8_t> gridToByteArray(typename TGrid::Ptr grid)
  {
    openvdb::GridPtrVec grids;
    grids.push_back(grid);
    std::ostringstream oss(std::ios_base::binary);
    openvdb::io::Stream(oss).write(grids);
    return compressString(oss.str());
  }

  /*!
   * \brief Unpacks a compressed byte array into a grid
   *
   * @tparam TGrid Grid Type
   * \param byte_array Input byte array
   *
   * \returns Pointer to the unpacked grid, nullptr if the data could not be parsed
   */
  template <typename TGrid>
  typename TGrid::Ptr byteArrayToGrid(const std::vector<uint8_t>& byte_array)
  {
    try
    {
      std::istringstream iss(decompressByteArray(byte_array));
      openvdb::io::Stream strm(iss);
      openvdb::GridPtrVecPtr grids = strm.getGrids();
      if (!grids || grids->empty())
      {
        logMessage(LogLevel::Error, "Byte array contains no grids");
        return nullptr;
      }
      // Scan for the first compatible grid instead of only trying the front
      // one, matching loadMap's behavior for multi-grid payloads.
      // The cast might also fail if different VDB versions are used.
      // Corresponding error messages are generated by VDB directly
      for (const openvdb::GridBase::Ptr& grid : *grids)
      {
        if (typename TGrid::Ptr cast_grid = openvdb::gridPtrCast<TGrid>(grid))
        {
          return cast_grid;
        }
      }
      logMessage(LogLevel::Error, "Byte array contains no compatible grid");
      return nullptr;
    }
    catch (const std::exception& e)
    {
      logMessage(LogLevel::Error, std::string("Could not parse grid from byte array: ") + e.what());
      return nullptr;
    }
  }

  /*!
   * \brief Returns the map mutex
   */
  std::shared_ptr<std::shared_mutex> getMapMutex() { return m_map_mutex; }

  /*!
   * \brief Adds a sensor input for a sensor source
   *
   * \param source_id Unique identifier of input source
   * \param max_range Maximum raycasting range
   * \param max_rate Maximum integration rate
   */
  void addInputSource(std::string source_id,
                      double max_range,
                      double max_rate,
                      bool ray_clearing  = true,
                      bool endpoint_hits = true,
                      double prob_hit    = -1.0,
                      double prob_miss   = -1.0)
  {
    if (source_id.empty())
    {
      logMessage(LogLevel::Error, "Input source ID must not be empty");
      return;
    }
    if (!std::isfinite(max_range))
    {
      logMessage(LogLevel::Error, "Input source " + source_id + " has a non-finite max range");
      return;
    }
    if (!std::isfinite(max_rate) || max_rate < 0.0)
    {
      logMessage(LogLevel::Error,
                 "Input source " + source_id + " max rate must be finite and non-negative");
      return;
    }
    if (!std::isfinite(prob_hit) || !std::isfinite(prob_miss))
    {
      logMessage(LogLevel::Error,
                 "Input source " + source_id + " probability overrides must be finite");
      return;
    }
    auto s           = std::make_shared<InputSource>();
    s->source_id     = source_id;
    s->ray_clearing  = ray_clearing;
    s->endpoint_hits = endpoint_hits;
    s->prob_hit      = prob_hit;
    s->prob_miss     = prob_miss;
    // Keep the non-positive sentinel instead of resolving it now. The
    // map-wide range may be configured after this source is registered or
    // changed later; accumulateUpdate resolves the fallback at use time.
    s->max_range   = max_range;
    s->update_grid = UpdateGridT::create(false);
    if (max_rate <= 0)
    {
      s->max_input_period = std::chrono::milliseconds(0);
    }
    else
    {
      const long double period_ms = std::ceil(1000.0L / static_cast<long double>(max_rate));
      const long double max_period_ms =
        static_cast<long double>(std::numeric_limits<std::int64_t>::max()) / 1000000.0L;
      if (period_ms > max_period_ms)
      {
        logMessage(LogLevel::Error,
                   "Input source " + source_id + " max rate is too small to represent safely");
        return;
      }
      s->max_input_period = std::chrono::milliseconds(
        static_cast<std::chrono::milliseconds::rep>(std::max(1.0L, period_ms)));
    }
    if (s->max_range <= 0 && m_max_range <= 0)
    {
      // This source will become usable once the map-wide fallback is set.
      logMessage(LogLevel::Warning,
                 "Input source " + source_id +
                   " has no positive range yet; it will use the map-wide max range at "
                   "integration time");
    }
    {
      // Register under the map lock: other threads read m_input_sources under
      // the shared lock. m_worker_threads is guarded by the same lock so two
      // concurrent registrations do not race on the map insert; starting the
      // worker inside the lock is safe because its first action is to block on
      // the shared lock until registration completes.
      std::unique_lock map_lock(*m_map_mutex);
      if (m_input_sources.find(source_id) != m_input_sources.end())
      {
        logMessage(LogLevel::Warning, "Input source " + source_id + " already registered");
        return;
      }
      m_input_sources[s->source_id] = s;
      m_worker_threads[source_id]   = std::thread(&VDBMapping::accumulationThread, this, source_id);
    }
  }


  /*!
   * \brief Handles function of all accumulation worker threads
   *
   * \param source_id Identifier of input source
   */
  void accumulationThread(std::string source_id)
  {
    // Capture the source pointer once instead of hitting the shared container
    // on every iteration. addInputSource guarantees the entry exists before
    // this thread is started.
    std::shared_ptr<InputSource> source;
    {
      std::shared_lock map_lock(*m_map_mutex);
      source = m_input_sources.at(source_id);
    }
    while (!m_config_set && !m_thread_stop_signal)
    {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    while (!m_thread_stop_signal)
    {
      const uint64_t period_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(source->max_input_period).count());
      const uint64_t start_time = getTimeNow();
      const uint64_t effective_period =
        std::min(period_ns, std::numeric_limits<uint64_t>::max() - start_time);
      const uint64_t sleep_time = start_time + effective_period;
      std::unique_lock lock(source->input_data_mutex);
      source->data_available_cv.wait(lock,
                                     [&] { return source->input_data || m_thread_stop_signal; });
      if (m_thread_stop_signal)
      {
        break;
      }
      if (m_map_mutex_requested)
      {
        // The integration thread wants the map; yield briefly instead of
        // busy-spinning on the still-pending input data.
        lock.unlock();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        continue;
      }
      lock.unlock();
      {
        // Match resetMap/addDataToAccumulate's map -> input lock order.
        // Keep the map lock until accumulation completes: otherwise reset
        // can clear the map after we pop a sample but before we insert it.
        std::shared_lock map_lock(*m_map_mutex);
        lock.lock();
        if (!source->input_data || m_thread_stop_signal)
        {
          continue;
        }
        auto measurement = std::move(*source->input_data);
        source->input_data.reset();
        lock.unlock();
        accumulateUpdateLocked(measurement.first, measurement.second, source);
      }
      sleepUntilOrStop(sleep_time, effective_period);
    }
    logMessage(LogLevel::Info, "Thread for source " + source_id + " received stop signal.");
  }

  /*!
   * \brief Handles function of integration worker thread
   */
  void integrationThread()
  {
    while (!m_config_set && !m_thread_stop_signal)
    {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    while (!m_thread_stop_signal)
    {
      const uint64_t period_ns =
        static_cast<uint64_t>(m_accumulation_period.load()) * 1000000ULL;
      const uint64_t start_time = getTimeNow();
      const uint64_t effective_period =
        std::min(period_ns, std::numeric_limits<uint64_t>::max() - start_time);
      const uint64_t sleeping_time = start_time + effective_period;
      integrateUpdate();
      sleepUntilOrStop(sleeping_time, effective_period);
    }
    logMessage(LogLevel::Info, "Integration thread received stop signal");
  }

  /*!
   * \brief Sleeps until the given time point, waking up early when the stop
   * signal is set so object destruction is not delayed by a full period
   *
   * \param until_ns Time point to sleep until (in nanoseconds)
   * \param max_expected_ns Longest sleep the caller intended (0 = no bound).
   * A remaining sleep beyond this means the deadline and the current clock
   * come from different epochs (e.g. the time callback was installed after
   * the deadline was computed) — bail out instead of freezing until the new
   * epoch catches up to the old one.
   */
  void sleepUntilOrStop(uint64_t until_ns, uint64_t max_expected_ns = 0) const
  {
    uint64_t previous_now = getTimeNow();
    while (!m_thread_stop_signal)
    {
      const uint64_t current_now = getTimeNow();
      if (current_now >= until_ns)
      {
        break;
      }
      // Epoch shift BEFORE entry: the backwards-jump guard below only sees
      // jumps between its own iterations.
      if (max_expected_ns > 0 && until_ns - current_now > max_expected_ns)
      {
        break;
      }
      // ROS time can jump backwards during bag seek/loop. A deadline from the
      // old epoch must not freeze this worker until replay catches back up.
      if (current_now < previous_now)
      {
        break;
      }
      previous_now = current_now;
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }

  /*!
   * \brief Updates the volume ray intersectors of each worker thread with respect to the current
   * map
   */
  void updateVolumeRayIntersectors()
  {
    // NOTE: fast mode requires GridT == openvdb::FloatGrid. VolumeRayIntersector
    // is hardcoded to FloatGrid throughout this class, so a non-float TData
    // specialization will fail to compile here once fast mode is exercised.
    // Always drop the previous topology first; when the map has just become
    // empty (or fast mode was disabled), retaining it would leave workers
    // ray-marching against stale occupancy indefinitely.
    resetVolumeRayIntersectors();
    if (m_fast_mode && !m_vdb_grid->empty())
    {
      m_volume_ray_intersector =
        std::make_shared<openvdb::tools::VolumeRayIntersector<openvdb::FloatGrid> >(*m_vdb_grid);
      for (auto& [source_id, source] : m_input_sources)
      {
        source->volume_ray_intersector =
          std::make_shared<openvdb::tools::VolumeRayIntersector<openvdb::FloatGrid> >(
            *m_volume_ray_intersector);
      }
    }
  }

  /*!
   * \brief Drops all volume ray intersectors. Must be called whenever the
   * map grid object is replaced, since intersectors reference the grid they
   * were constructed from without keeping it alive.
   * Expects the unique map lock to be held by the caller.
   */
  void resetVolumeRayIntersectors()
  {
    m_volume_ray_intersector.reset();
    for (auto& [source_id, source] : m_input_sources)
    {
      source->volume_ray_intersector.reset();
    }
  }

  /*!
   * \brief Handles changing the mapping config
   *
   * \param config Configuration structure
   * \returns false when the config was rejected and NOTHING was applied.
   * Returning void here let a derived class apply its own half of the config
   * after a base reject — a silently half-applied configuration.
   */
  virtual bool setConfig(const TConfig& config)
  {
    if (!validateBaseConfig(config))
    {
      return false;
    }
    std::unique_lock map_lock(*m_map_mutex);
    applyBaseConfigLocked(config);
    // Publish only after every field protected by the map lock is coherent.
    // Derived maps use the same helpers and set this after their own fields.
    m_config_set.store(true, std::memory_order_release);
    return true;
  }

protected:
  bool validateBaseConfig(const TConfig& config) const
  {
    if (!std::isfinite(config.max_range) || config.max_range < 0.0)
    {
      logMessage(LogLevel::Error,
                 "Max range of " + std::to_string(config.max_range) +
                   " invalid. Range must be finite and non-negative.");
      return false;
    }
    if (!std::isfinite(config.accumulation_period) || config.accumulation_period <= 0.0)
    {
      // Without this guard, a zero or negative period silently casts to 0 ms
      // (busy-spin) or wraps to a huge unsigned sleep duration on the
      // integration thread.
      logMessage(LogLevel::Error,
                 "Accumulation period of " + std::to_string(config.accumulation_period) +
                   " invalid. Must be a finite positive number of seconds.");
      return false;
    }
    if (config.accumulation_period * 1000.0 > static_cast<double>(std::numeric_limits<int>::max()))
    {
      logMessage(LogLevel::Error, "Accumulation period is too large to represent in milliseconds");
      return false;
    }
    if (config.max_serialized_grid_bytes == 0)
    {
      logMessage(LogLevel::Error, "max_serialized_grid_bytes must be positive");
      return false;
    }
    return true;
  }

  void applyBaseConfigLocked(const TConfig& config)
  {
    m_max_range                 = config.max_range;
    m_map_directory_path        = config.map_directory_path;
    m_fast_mode                 = config.fast_mode;
    m_max_serialized_grid_bytes = config.max_serialized_grid_bytes;
    // Floor at 1 ms: (0, 1ms) truncated to 0 and busy-spun the integration
    // thread while it held the unique map lock.
    m_accumulation_period = std::max(1, static_cast<int>(config.accumulation_period * 1000.0));
  }

  /*!
   * \brief Emits a log message through the configured callback, falling back
   * to stdout (Info) / stderr (Warning, Error) when no callback is set
   *
   * \param level Message severity
   * \param message Message text
   */
  void logMessage(const LogLevel level, const std::string& message) const
  {
    std::lock_guard<std::mutex> lock(m_log_mutex);
    if (m_log_callback)
    {
      m_log_callback(level, message);
      return;
    }
    if (level == LogLevel::Info)
    {
      std::cout << message << std::endl;
    }
    else
    {
      std::cerr << message << std::endl;
    }
  }

  // Default no-op implementations — OccupancyVDBMapping (and any other
  // subclass) overrides these with concrete log-odds behaviour. The params
  // are named for documentation / IDE completion; [[maybe_unused]] silences
  // -Wunused-parameter for the default bodies without removing the names.
  /*!
   * \brief Applies a per-source hit/miss probability override for the
   * duration of one updateMap call (values <= 0 keep the map-wide config).
   * Called under the exclusive map lock. Base implementation ignores the
   * override; probability-based subclasses translate it into their update
   * weights.
   */
  virtual void applySourceProbabilityOverride([[maybe_unused]] double prob_hit,
                                              [[maybe_unused]] double prob_miss)
  {
  }

  /*!
   * \brief Restores the map-wide probabilities after
   * applySourceProbabilityOverride
   */
  virtual void clearSourceProbabilityOverride() {}

  virtual bool updateFreeNode([[maybe_unused]] TData& voxel_value, [[maybe_unused]] bool& active)
  {
    return false;
  }
  virtual bool updateOccupiedNode([[maybe_unused]] TData& voxel_value,
                                  [[maybe_unused]] bool& active)
  {
    return false;
  }
  virtual bool setNodeToFree([[maybe_unused]] TData& voxel_value, [[maybe_unused]] bool& active)
  {
    return false;
  }
  virtual bool setNodeToOccupied([[maybe_unused]] TData& voxel_value, [[maybe_unused]] bool& active)
  {
    return false;
  }
  virtual bool setNodeState([[maybe_unused]] TData& voxel_value, [[maybe_unused]] bool& active)
  {
    return false;
  }

  virtual bool createMapFromPointCloud([[maybe_unused]] const typename PointCloudT::Ptr& cloud,
                                       [[maybe_unused]] const bool set_background,
                                       [[maybe_unused]] const bool clear_map)
  {
    logMessage(LogLevel::Error, "Not implemented for data type");
    return false;
  }

  /*!
   * \brief VDB grid pointer
   */
  typename GridT::Ptr m_vdb_grid;

  /*!
   * \brief Grid containing all artificial areas of the map
   */
  typename UpdateGridT::Ptr m_artificial_area_grid;
  /*!
   * \brief Maximum raycasting distance.
   * Atomic because it is read by background worker threads while setConfig
   * may run on the main thread.
   */
  std::atomic<double> m_max_range{0.0};
  /*!
   * \brief Grid resolution of the map.
   * Atomic because loadMap may overwrite it (when a loaded map's resolution
   * differs) while getResolution / getMapSection read it from other threads.
   */
  std::atomic<double> m_resolution;

  /*!
   * \brief Should vdb_mapping operate in the fast raycasting method.
   * Atomic because it is read by background worker threads while setConfig
   * may run on the main thread.
   */
  std::atomic<bool> m_fast_mode{false};

  /*!
   * \brief Timeframe in milliseconds over which updates are accumulated before
   * inserting them into the map.
   * Atomic because it is read by the integration thread while setConfig may
   * run on the main thread.
   */
  std::atomic<int> m_accumulation_period{1000};
  /*!
   * \brief path where the maps will be stored
   */
  std::string m_map_directory_path;
  /*!
   * \brief Flag checking whether a valid config was already loaded.
   * Atomic because it is read by background threads (accumulation/integration)
   * and written by setConfig from the main thread.
   */
  std::atomic<bool> m_config_set;

  /*!
   * \brief Compression level for grid compression
   */
  unsigned int m_compression_level = 1;

  /*!
   * \brief Maximum accepted compressed or declared decompressed grid size.
   */
  std::atomic<std::size_t> m_max_serialized_grid_bytes{512ULL * 1024ULL * 1024ULL};

  /*!
   * \brief Specifies whether artificial areas are present
   */
  bool m_artificial_areas_present;

  /*!
   * \brief Optional log callback; when unset, messages go to stdout/stderr
   */
  LogCallbackT m_log_callback;

  /*!
   * \brief Serializes log callback invocation and replacement
   */
  mutable std::mutex m_log_mutex;

  /*!
   * \brief Optional time callback; when unset, uses std::chrono
   */
  TimeCallbackT m_time_callback;

  /*!
   * \brief Serializes time callback invocation and replacement
   */
  mutable std::mutex m_time_mutex;

  /*!
   * \brief Map mutex for the map grid
   */
  mutable std::shared_ptr<std::shared_mutex> m_map_mutex;

  /*!
   * \brief Specifies whether a unique lock on the map mutex was requested.
   * Used to implement a priority mutex.
   */
  mutable std::atomic<bool> m_map_mutex_requested{false};

  /*!
   * \brief List of input sources
   */
  std::map<std::string, std::shared_ptr<InputSource> > m_input_sources;

  /*!
   * \brief Stop signal for all worker threads
   */
  std::atomic<bool> m_thread_stop_signal{false};

  /*!
   * \brief List of all accumulation worker threads
   */
  std::map<std::string, std::thread> m_worker_threads;

  /*!
   * \brief Integration worker thread
   */
  std::thread m_integration_thread;

  /*!
   * \brief Global volume ray intersector. This one has to be stores since the volume ray
   * intersectors of each input source is just a shallow copy with a separate grid accessor.
   */
  std::shared_ptr<openvdb::tools::VolumeRayIntersector<openvdb::FloatGrid> >
    m_volume_ray_intersector;
};

} // namespace vdb_mapping

#endif /* VDB_MAPPING_VDB_MAPPING_H_INCLUDED */
