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
 * \date    2021-04-29
 *
 */
//----------------------------------------------------------------------
#ifndef VDB_MAPPING_OCCUPANCY_VDB_MAPPING_H_INCLUDED
#define VDB_MAPPING_OCCUPANCY_VDB_MAPPING_H_INCLUDED

#include "vdb_mapping/VDBMapping.hpp"

namespace vdb_mapping {

/*!
 * \brief Accumulation of configuration parameters
 *
 * The defaults follow the beam-based inverse sensor model parameters
 * published for OctoMap (Hornung et al., Auton. Robots 2013, Sect. 5.1):
 * P(hit) = 0.7, P(miss) = 0.4, with activation thresholds at 0.12 / 0.97.
 */
struct Config : BaseConfig
{
  double prob_hit       = 0.7;
  double prob_miss      = 0.4;
  double prob_thres_min = 0.12;
  double prob_thres_max = 0.97;
  /*!
   * \brief Clamping bounds of the log-odds update (Yguel et al.'s clamping
   * update policy, see OctoMap Eq. 4). Tighter bounds let the map adapt to
   * changes faster and improve pruning compression, at the cost of map
   * confidence; wider bounds behave closer to an unbounded estimator.
   */
  double prob_clamp_min = 0.01;
  double prob_clamp_max = 0.99;
};

class OccupancyVDBMapping : public VDBMapping<float, Config>
{
public:
  OccupancyVDBMapping(const double resolution)
    : VDBMapping<float, Config>(resolution)
  {
  }

  /*!
   * \brief Handles changing the mapping config
   *
   * \param config Configuration structure
   */
  inline void setConfig(const Config& config) override
  {
    // Validate occupancy-specific config before applying base config.
    // All probabilities must lie strictly inside (0, 1); values on or outside
    // the boundary produce NaN/inf log odds that poison every voxel update.
    if (!(config.prob_miss > 0.0 && config.prob_miss <= 0.5))
    {
      logMessage(LogLevel::Error,
                 "Probability for a miss should be in (0, 0.5] but is " +
                   std::to_string(config.prob_miss));
      return;
    }
    if (!(config.prob_hit >= 0.5 && config.prob_hit < 1.0))
    {
      logMessage(LogLevel::Error,
                 "Probability for a hit should be in [0.5, 1) but is " +
                   std::to_string(config.prob_hit));
      return;
    }
    if (!(config.prob_thres_min > 0.0 && config.prob_thres_max < 1.0 &&
          config.prob_thres_min < config.prob_thres_max))
    {
      logMessage(LogLevel::Error,
                 "Probability thresholds must satisfy 0 < min < max < 1 but are min=" +
                   std::to_string(config.prob_thres_min) +
                   " max=" + std::to_string(config.prob_thres_max));
      return;
    }
    // The clamps must strictly enclose the activation thresholds. The
    // activation comparisons are strict (value > thres_max activates, value <
    // thres_min deactivates), so a clamp equal to a threshold would let a voxel
    // saturate exactly AT the threshold and never cross it: a saturated
    // obstacle would stay inactive and the map would never show it.
    if (!(config.prob_clamp_min > 0.0 && config.prob_clamp_min < config.prob_thres_min &&
          config.prob_clamp_max < 1.0 && config.prob_clamp_max > config.prob_thres_max))
    {
      logMessage(LogLevel::Error,
                 "Clamping bounds must satisfy 0 < clamp_min < thres_min and thres_max < "
                 "clamp_max < 1 but are clamp_min=" +
                   std::to_string(config.prob_clamp_min) +
                   " clamp_max=" + std::to_string(config.prob_clamp_max));
      return;
    }

    // call base class function after validation passes
    VDBMapping::setConfig(config);

    // Store probabilities as log odds
    m_logodds_miss = static_cast<float>(log(config.prob_miss) - log(1 - config.prob_miss));
    m_logodds_hit  = static_cast<float>(log(config.prob_hit) - log(1 - config.prob_hit));
    m_logodds_thres_min =
      static_cast<float>(log(config.prob_thres_min) - log(1 - config.prob_thres_min));
    m_logodds_thres_max =
      static_cast<float>(log(config.prob_thres_max) - log(1 - config.prob_thres_max));
    // Clamping update policy bounds (OctoMap Eq. 4): keep the log odds
    // bounded so the map stays adaptive to changes in the environment
    m_max_logodds = static_cast<float>(log(config.prob_clamp_max) - log(1 - config.prob_clamp_max));
    m_min_logodds = static_cast<float>(log(config.prob_clamp_min) - log(1 - config.prob_clamp_min));
  }

protected:
  // Clamping update policy, L = max(min(L + l, l_max), l_min) (OctoMap
  // Eq. 4): the clamp applies on every update so the confidence stays
  // bounded; the activation thresholds then switch the voxel state with
  // hysteresis.
  inline bool updateFreeNode(float& voxel_value, bool& active) override
  {
    voxel_value += m_logodds_miss;
    if (voxel_value < m_min_logodds)
    {
      voxel_value = m_min_logodds;
    }
    if (voxel_value < m_logodds_thres_min)
    {
      active = false;
    }
    return true;
  }
  inline bool updateOccupiedNode(float& voxel_value, bool& active) override
  {
    voxel_value += m_logodds_hit;
    if (voxel_value > m_max_logodds)
    {
      voxel_value = m_max_logodds;
    }
    if (voxel_value > m_logodds_thres_max)
    {
      active = true;
    }
    return true;
  }
  inline bool setNodeToFree(float& voxel_value, bool& active) override
  {
    voxel_value = m_min_logodds;
    active      = false;
    return true;
  }
  inline bool setNodeToOccupied(float& voxel_value, bool& active) override
  {
    voxel_value = m_max_logodds;
    active      = true;
    return true;
  }
  inline bool setNodeState(float& voxel_value, bool& active) override
  {
    active = voxel_value > m_logodds_thres_max;
    return true;
  }

  inline bool createMapFromPointCloud(const PointCloudT::Ptr& cloud,
                                      const bool set_background,
                                      const bool clear_map) override
  {
    if (!m_config_set)
    {
      // The log-odds members are uninitialized until setConfig has run;
      // writing them into the grid would store garbage values.
      logMessage(LogLevel::Error, "Map not properly configured. Did you call setConfig method?");
      return false;
    }
    if (clear_map)
    {
      m_vdb_grid->clear();
    }

    typename GridT::Accessor acc = m_vdb_grid->getAccessor();

    for (const auto& point : cloud->points)
    {
      // A non-finite point would floor to an extreme coordinate and blow up
      // the background sparseFill below (its bbox would span the entire
      // coordinate range).
      if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z))
      {
        continue;
      }
      acc.setValueOn(this->worldToIndex(openvdb::Vec3d(point.x, point.y, point.z)), m_max_logodds);
    }

    if (set_background)
    {
      // Sparse-fill the bounding box with the free-space value instead of
      // visiting every voxel: the previous dense triple loop was O(volume)
      // and allocated leaf nodes for the entire box. sparseFill overwrites
      // everything inside the box, so stash the occupied voxels and restore
      // them afterwards.
      openvdb::CoordBBox bbox = m_vdb_grid->evalActiveVoxelBoundingBox();

      std::vector<std::pair<openvdb::Coord, float> > active_voxels;
      active_voxels.reserve(m_vdb_grid->activeVoxelCount());
      for (auto iter = m_vdb_grid->cbeginValueOn(); iter; ++iter)
      {
        active_voxels.emplace_back(iter.getCoord(), iter.getValue());
      }

      m_vdb_grid->sparseFill(bbox, m_min_logodds, false);

      for (const auto& [coord, value] : active_voxels)
      {
        acc.setValueOn(coord, value);
      }
    }

    m_vdb_grid->pruneGrid();
    return true;
  }

  // In-class initializers (defense in depth): the setConfig guard in the
  // mutators is the primary protection against using these before they are
  // configured, but zero-initializing them avoids indeterminate values should
  // any future code path read them early.
  /*!
   * \brief Probability update value for passing an obstacle
   */
  float m_logodds_hit = 0.0f;
  /*!
   * \brief Probability update value for passing free space
   */
  float m_logodds_miss = 0.0f;
  /*!
   * \brief Lower occupancy probability threshold below which a voxel is deactivated
   */
  float m_logodds_thres_min = 0.0f;
  /*!
   * \brief Upper occupancy probability threshold above which a voxel is activated
   */
  float m_logodds_thres_max = 0.0f;
  /*!
   * \brief Maximum clamping point for logodds
   */
  float m_max_logodds = 0.0f;
  /*!
   * \brief Minimum clamping point for logodds
   */
  float m_min_logodds = 0.0f;
};


} // namespace vdb_mapping

#endif /* VDB_MAPPING_OCCUPANCY_VDB_MAPPING_H_INCLUDED */
