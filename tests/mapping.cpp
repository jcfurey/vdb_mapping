#include "gtest/gtest.h"
#include <array>
#include <cmath>
#include <limits>
#include <vdb_mapping/OccupancyVDBMapping.hpp>
#include <vector>

namespace vdb_mapping {

TEST(Mapping, SetConfig)
{
  OccupancyVDBMapping map(1);
  OccupancyVDBMapping::PointCloudT::Ptr cloud(new OccupancyVDBMapping::PointCloudT);
  cloud->points.emplace_back(0, 0, 1);
  Eigen::Matrix<double, 3, 1> origin(0, 0, 0);
  map.insertPointCloud(cloud, origin, "test");
  OccupancyVDBMapping::GridT::Accessor acc = map.getGrid()->getAccessor();
  openvdb::Coord coord(0, 0, 1);
  EXPECT_EQ(acc.getValue(coord), 0.0);
  Config conf;
  conf.max_range      = 10;
  conf.fast_mode      = false;
  conf.prob_hit       = 0.9;
  conf.prob_miss      = 0.1;
  conf.prob_thres_max = 0.51;
  conf.prob_thres_min = 0.49;
  map.setConfig(conf);
  map.addInputSource("test", conf.max_range, 0);
  auto log_hit  = static_cast<float>(log(conf.prob_hit) - log(1 - conf.prob_hit));
  auto log_miss = static_cast<float>(log(conf.prob_miss) - log(1 - conf.prob_miss));
  map.insertPointCloud(cloud, origin, "test");
  EXPECT_EQ(acc.getValue(openvdb::Coord(0, 0, 0)), log_miss);
  EXPECT_EQ(acc.getValue(openvdb::Coord(0, 0, 1)), log_hit);
}

TEST(Mapping, InsertPositivePoint)
{
  double resolution = 0.1;
  OccupancyVDBMapping map(resolution);
  Config conf;
  conf.max_range = 10;
  conf.fast_mode = false;
  // conf.accumulation_period = 10.0;
  conf.prob_hit       = 0.9;
  conf.prob_miss      = 0.1;
  conf.prob_thres_max = 0.51;
  conf.prob_thres_min = 0.49;
  map.setConfig(conf);
  map.addInputSource("test", conf.max_range, 0);

  auto log_hit  = static_cast<float>(log(conf.prob_hit) - log(1 - conf.prob_hit));
  auto log_miss = static_cast<float>(log(conf.prob_miss) - log(1 - conf.prob_miss));

  OccupancyVDBMapping::PointCloudT::Ptr cloud(new OccupancyVDBMapping::PointCloudT);
  cloud->points.emplace_back(0, 0, 5 * resolution);
  Eigen::Matrix<double, 3, 1> origin(0, 0, 0);
  map.insertPointCloud(cloud, origin, "test");
  OccupancyVDBMapping::GridT::Accessor acc = map.getGrid()->getAccessor();
  for (int i = 0; i < 5; ++i)
  {
    openvdb::Coord coord(0, 0, i);
    EXPECT_EQ(acc.getValue(coord), log_miss);
    EXPECT_FALSE(acc.isValueOn(coord));
  }
  openvdb::Coord coord(0, 0, 5);
  EXPECT_EQ(acc.getValue(coord), log_hit);
  EXPECT_TRUE(acc.isValueOn(coord));
}

TEST(Mapping, InsertNegativePoint)
{
  double resolution = 0.1;
  OccupancyVDBMapping map(resolution);
  Config conf;
  conf.max_range = 10;
  conf.fast_mode = false;
  // conf.accumulation_period = 10.0;
  conf.prob_hit       = 0.9;
  conf.prob_miss      = 0.1;
  conf.prob_thres_max = 0.51;
  conf.prob_thres_min = 0.49;
  map.setConfig(conf);
  map.addInputSource("test", conf.max_range, 0);

  auto log_hit  = static_cast<float>(log(conf.prob_hit) - log(1 - conf.prob_hit));
  auto log_miss = static_cast<float>(log(conf.prob_miss) - log(1 - conf.prob_miss));

  OccupancyVDBMapping::PointCloudT::Ptr cloud(new OccupancyVDBMapping::PointCloudT);
  cloud->points.emplace_back(0, 0, -5 * resolution);
  Eigen::Matrix<double, 3, 1> origin(0, 0, 0);
  map.insertPointCloud(cloud, origin, "test");
  OccupancyVDBMapping::GridT::Accessor acc = map.getGrid()->getAccessor();
  for (int i = 1; i < 5; ++i)
  {
    openvdb::Coord coord(0, 0, -1 * i);
    EXPECT_EQ(acc.getValue(coord), log_miss);
    EXPECT_FALSE(acc.isValueOn(coord));
  }
  openvdb::Coord coord(0, 0, -5);
  EXPECT_EQ(acc.getValue(coord), log_hit);
  EXPECT_TRUE(acc.isValueOn(coord));
}

TEST(Mapping, InsertMaxRangePoint)
{
  double resolution = 0.1;
  OccupancyVDBMapping map(resolution);
  Config conf;
  conf.max_range = 0.5;
  conf.fast_mode = false;
  // conf.accumulation_period = 10.0;
  conf.prob_hit       = 0.9;
  conf.prob_miss      = 0.1;
  conf.prob_thres_max = 0.51;
  conf.prob_thres_min = 0.49;
  map.setConfig(conf);
  map.addInputSource("test", conf.max_range, 0);

  auto log_miss = static_cast<float>(log(conf.prob_miss) - log(1 - conf.prob_miss));

  OccupancyVDBMapping::PointCloudT::Ptr cloud(new OccupancyVDBMapping::PointCloudT);
  cloud->points.emplace_back(0, 0, 7 * resolution);
  Eigen::Matrix<double, 3, 1> origin(0, 0, 0);
  map.insertPointCloud(cloud, origin, "test");
  OccupancyVDBMapping::GridT::Accessor acc = map.getGrid()->getAccessor();
  for (int i = 0; i <= (int)(conf.max_range / resolution); ++i)
  {
    openvdb::Coord coord(0, 0, i);
    EXPECT_EQ(acc.getValue(coord), log_miss);
    EXPECT_FALSE(acc.isValueOn(coord));
  }
  openvdb::Coord coord(0, 0, (int)(conf.max_range / resolution) + 1);
  EXPECT_EQ(acc.getValue(coord), 0);
  EXPECT_FALSE(acc.isValueOn(coord));
}

TEST(Mapping, ResetMap)
{
  OccupancyVDBMapping map(1);
  Config conf;
  conf.max_range = 10;
  conf.fast_mode = false;
  // conf.accumulation_period = 10.0;
  conf.prob_hit       = 0.9;
  conf.prob_miss      = 0.1;
  conf.prob_thres_max = 0.51;
  conf.prob_thres_min = 0.49;
  map.setConfig(conf);
  map.addInputSource("test", conf.max_range, 0);

  auto log_hit = static_cast<float>(log(conf.prob_hit) - log(1 - conf.prob_hit));

  OccupancyVDBMapping::PointCloudT::Ptr cloud(new OccupancyVDBMapping::PointCloudT);
  cloud->points.emplace_back(0, 0, 1);
  Eigen::Matrix<double, 3, 1> origin(0, 0, 0);
  map.insertPointCloud(cloud, origin, "test");
  OccupancyVDBMapping::GridT::Accessor acc = map.getGrid()->getAccessor();
  EXPECT_EQ(acc.getValue(openvdb::Coord(0, 0, 1)), log_hit);
  map.resetMap();
  acc = map.getGrid()->getAccessor();
  EXPECT_EQ(acc.getValue(openvdb::Coord(0, 0, 1)), 0.0);
}

TEST(Mapping, InvalidConfigRejected)
{
  OccupancyVDBMapping map(1);
  Config conf;
  conf.max_range      = 10;
  conf.fast_mode      = false;
  conf.prob_hit       = 0.9;
  conf.prob_miss      = 0.1;
  conf.prob_thres_max = 0.51;
  conf.prob_thres_min = 0.49;
  map.setConfig(conf);
  map.addInputSource("test", conf.max_range, 0);

  auto log_hit = static_cast<float>(log(conf.prob_hit) - log(1 - conf.prob_hit));

  OccupancyVDBMapping::PointCloudT::Ptr cloud(new OccupancyVDBMapping::PointCloudT);
  cloud->points.emplace_back(0, 0, 1);
  Eigen::Matrix<double, 3, 1> origin(0, 0, 0);
  map.insertPointCloud(cloud, origin, "test");
  OccupancyVDBMapping::GridT::Accessor acc = map.getGrid()->getAccessor();
  EXPECT_EQ(acc.getValue(openvdb::Coord(0, 0, 1)), log_hit);

  // Attempt to apply invalid config (prob_miss > 0.5) - should be rejected
  Config bad_conf    = conf;
  bad_conf.prob_miss = 0.8;
  map.setConfig(bad_conf);

  // Map should still use the original valid config
  map.resetMap();
  map.insertPointCloud(cloud, origin, "test");
  acc = map.getGrid()->getAccessor();
  EXPECT_EQ(acc.getValue(openvdb::Coord(0, 0, 1)), log_hit);
}

TEST(Mapping, GridSerialization)
{
  double resolution = 0.1;
  OccupancyVDBMapping map(resolution);
  Config conf;
  conf.max_range      = 10;
  conf.fast_mode      = false;
  conf.prob_hit       = 0.9;
  conf.prob_miss      = 0.1;
  conf.prob_thres_max = 0.51;
  conf.prob_thres_min = 0.49;
  map.setConfig(conf);
  map.addInputSource("test", conf.max_range, 0);

  OccupancyVDBMapping::PointCloudT::Ptr cloud(new OccupancyVDBMapping::PointCloudT);
  cloud->points.emplace_back(0, 0, 5 * resolution);
  cloud->points.emplace_back(resolution, 0, 5 * resolution);
  Eigen::Matrix<double, 3, 1> origin(0, 0, 0);
  map.insertPointCloud(cloud, origin, "test");

  auto active_before = map.getGrid()->activeVoxelCount();
  EXPECT_GT(active_before, 0u);

  // Serialize to compressed byte array
  auto bytes = map.gridToByteArray<OccupancyVDBMapping::GridT>(map.getGrid());
  EXPECT_GT(bytes.size(), 0u);

  // Deserialize back
  auto restored = map.byteArrayToGrid<OccupancyVDBMapping::GridT>(bytes);
  ASSERT_NE(restored, nullptr);
  EXPECT_EQ(restored->activeVoxelCount(), active_before);
}

TEST(Mapping, DestructWithoutConfig)
{
  // Regression test: the worker threads used to wait for a config forever,
  // ignoring the stop signal, so destroying an unconfigured map deadlocked.
  {
    OccupancyVDBMapping map(1);
  }
  {
    OccupancyVDBMapping map(1);
    map.addInputSource("test", 10, 0);
  }
  SUCCEED();
}

namespace {
// Builds a fast-mode map and inserts the given obstacle points so that the
// volume ray intersectors are initialized for raytrace queries.
void setupFastModeMap(OccupancyVDBMapping& map, const std::vector<std::array<float, 3> >& obstacles)
{
  Config conf;
  conf.max_range      = 50;
  conf.fast_mode      = true;
  conf.prob_hit       = 0.9;
  conf.prob_miss      = 0.1;
  conf.prob_thres_max = 0.51;
  conf.prob_thres_min = 0.49;
  map.setConfig(conf);
  map.addInputSource("test", conf.max_range, 0);

  OccupancyVDBMapping::PointCloudT::Ptr cloud(new OccupancyVDBMapping::PointCloudT);
  for (const auto& p : obstacles)
  {
    cloud->points.emplace_back(p[0], p[1], p[2]);
  }
  map.addPointsToGrid(cloud);

  // Trigger updateVolumeRayIntersectors via an empty integration cycle
  OccupancyVDBMapping::PointCloudT::Ptr empty_cloud(new OccupancyVDBMapping::PointCloudT);
  Eigen::Matrix<double, 3, 1> origin(0, 0, 0);
  map.insertPointCloud(empty_cloud, origin, "test");
}
} // namespace

TEST(Mapping, RaytraceLeafAlignedObstacle)
{
  // Regression test: the span walk stepped before checking, skipping the
  // first voxel of each intersected span. Leaf nodes are 8^3 aligned, so an
  // obstacle at a leaf-aligned index (200 here) was never reported.
  double resolution = 0.1;
  OccupancyVDBMapping map(resolution);
  setupFastModeMap(map, {{20.0f, 0.0f, 0.0f}});

  bool success;
  openvdb::Vec3d end_point;
  map.raytrace(openvdb::Vec3d(0, 0, 0), openvdb::Vec3d(1, 0, 0), 30.0, success, end_point);
  EXPECT_TRUE(success);
  EXPECT_NEAR(end_point.x(), 20.0, resolution);
  EXPECT_NEAR(end_point.y(), 0.0, resolution);
  EXPECT_NEAR(end_point.z(), 0.0, resolution);
}

TEST(Mapping, RaytraceMultiSpan)
{
  // Regression test: raytrace only marched the first intersected leaf span.
  // An active voxel near (but not on) the ray made the query give up before
  // reaching the actual obstacle further along the ray.
  double resolution = 0.1;
  OccupancyVDBMapping map(resolution);
  setupFastModeMap(map, {{20.0f, 0.0f, 0.0f}, {10.0f, 0.7f, 0.0f}});

  bool success;
  openvdb::Vec3d end_point;
  map.raytrace(openvdb::Vec3d(0, 0, 0), openvdb::Vec3d(1, 0, 0), 30.0, success, end_point);
  EXPECT_TRUE(success);
  EXPECT_NEAR(end_point.x(), 20.0, resolution);
}

TEST(Mapping, RaytraceFromNearObstacle)
{
  // Regression test: a ray starting inside an occupied leaf node tripped an
  // OpenVDB Ray::setTimes assertion (t0 must be > 0) and aborted the process.
  double resolution = 0.1;
  OccupancyVDBMapping map(resolution);
  setupFastModeMap(map, {{20.0f, 0.0f, 0.0f}});

  bool success;
  openvdb::Vec3d end_point;
  map.raytrace(openvdb::Vec3d(19.85, 0, 0), openvdb::Vec3d(1, 0, 0), 5.0, success, end_point);
  EXPECT_TRUE(success);
  EXPECT_NEAR(end_point.x(), 20.0, resolution);

  // Ray pointing away from the obstacle must not report a hit
  map.raytrace(openvdb::Vec3d(25.0, 0, 0), openvdb::Vec3d(1, 0, 0), 5.0, success, end_point);
  EXPECT_FALSE(success);
}

TEST(Mapping, RaytraceWithoutFastMode)
{
  // raytrace used to require the volume ray intersector, which only exists
  // in fast mode after an integration on a non-empty grid: with fast_mode
  // off every query failed. Without an intersector the query now falls back
  // to an exact DDA walk over the full ray.
  double resolution = 0.1;
  OccupancyVDBMapping map(resolution);
  Config conf;
  conf.max_range      = 50;
  conf.fast_mode      = false;
  conf.prob_hit       = 0.9;
  conf.prob_miss      = 0.1;
  conf.prob_thres_max = 0.51;
  conf.prob_thres_min = 0.49;
  map.setConfig(conf);

  OccupancyVDBMapping::PointCloudT::Ptr obstacles(new OccupancyVDBMapping::PointCloudT);
  obstacles->points.emplace_back(20.0f, 0.0f, 0.0f);
  map.addPointsToGrid(obstacles);

  bool success;
  openvdb::Vec3d end_point;
  map.raytrace(openvdb::Vec3d(0, 0, 0), openvdb::Vec3d(1, 0, 0), 30.0, success, end_point);
  EXPECT_TRUE(success);
  EXPECT_NEAR(end_point.x(), 20.0, resolution);
  EXPECT_NEAR(end_point.y(), 0.0, resolution);
  EXPECT_NEAR(end_point.z(), 0.0, resolution);

  // A miss must report the max-range point with success == false
  map.raytrace(openvdb::Vec3d(0, 0, 0), openvdb::Vec3d(-1, 0, 0), 5.0, success, end_point);
  EXPECT_FALSE(success);
  EXPECT_NEAR(end_point.x(), -5.0, resolution);

  // Fast mode before the first integration has no intersector either and
  // must take the same fallback instead of failing
  OccupancyVDBMapping map_fast(resolution);
  conf.fast_mode = true;
  map_fast.setConfig(conf);
  map_fast.addPointsToGrid(obstacles);
  map_fast.raytrace(openvdb::Vec3d(0, 0, 0), openvdb::Vec3d(1, 0, 0), 30.0, success, end_point);
  EXPECT_TRUE(success);
  EXPECT_NEAR(end_point.x(), 20.0, resolution);
}

TEST(Mapping, MapSectionPreservesValues)
{
  // Regression test: sparse map section extraction flattened all float voxel
  // values to 1.0 instead of preserving the stored log odds.
  OccupancyVDBMapping map(1);
  Config conf;
  conf.max_range      = 10;
  conf.fast_mode      = false;
  conf.prob_hit       = 0.9;
  conf.prob_miss      = 0.1;
  conf.prob_thres_max = 0.51;
  conf.prob_thres_min = 0.49;
  map.setConfig(conf);
  map.addInputSource("test", conf.max_range, 0);

  auto log_hit = static_cast<float>(log(conf.prob_hit) - log(1 - conf.prob_hit));

  OccupancyVDBMapping::PointCloudT::Ptr cloud(new OccupancyVDBMapping::PointCloudT);
  cloud->points.emplace_back(0, 0, 1);
  Eigen::Matrix<double, 3, 1> origin(0, 0, 0);
  map.insertPointCloud(cloud, origin, "test");

  Eigen::Matrix<double, 3, 1> min_boundary(-5, -5, -5);
  Eigen::Matrix<double, 3, 1> max_boundary(5, 5, 5);
  Eigen::Matrix<double, 4, 4> identity = Eigen::Matrix<double, 4, 4>::Identity();

  auto section = map.getMapSectionGrid(min_boundary, max_boundary, identity, false);
  OccupancyVDBMapping::GridT::Accessor section_acc = section->getAccessor();
  EXPECT_TRUE(section_acc.isValueOn(openvdb::Coord(0, 0, 1)));
  EXPECT_FLOAT_EQ(section_acc.getValue(openvdb::Coord(0, 0, 1)), log_hit);
}

TEST(Mapping, ByteArrayToGridRejectsGarbage)
{
  OccupancyVDBMapping map(1);
  std::vector<uint8_t> garbage = {0xde, 0xad, 0xbe, 0xef, 0x42};
  auto grid                    = map.byteArrayToGrid<OccupancyVDBMapping::GridT>(garbage);
  EXPECT_EQ(grid, nullptr);
}

TEST(Mapping, LogCallbackCapturesMessages)
{
  std::vector<std::pair<OccupancyVDBMapping::LogLevel, std::string> > messages;
  OccupancyVDBMapping map(1);
  map.setLogCallback([&](OccupancyVDBMapping::LogLevel level, const std::string& msg) {
    messages.emplace_back(level, msg);
  });

  // Invalid config must be reported through the callback, not stderr
  Config bad_conf;
  bad_conf.prob_miss = 0.8;
  map.setConfig(bad_conf);

  ASSERT_EQ(messages.size(), 1u);
  EXPECT_EQ(messages[0].first, OccupancyVDBMapping::LogLevel::Error);
  EXPECT_NE(messages[0].second.find("Probability for a miss"), std::string::npos);

  // Unknown source warning also goes through the callback
  OccupancyVDBMapping::PointCloudT::Ptr cloud(new OccupancyVDBMapping::PointCloudT);
  Eigen::Matrix<double, 3, 1> origin(0, 0, 0);
  map.accumulateUpdate(cloud, origin, "unknown_source");
  ASSERT_EQ(messages.size(), 2u);
  EXPECT_EQ(messages[1].first, OccupancyVDBMapping::LogLevel::Warning);
}

TEST(Mapping, CreateMapFromPCDSetsBackground)
{
  double resolution = 0.1;
  Config conf;
  conf.max_range      = 10;
  conf.fast_mode      = false;
  conf.prob_hit       = 0.9;
  conf.prob_miss      = 0.1;
  conf.prob_thres_max = 0.51;
  conf.prob_thres_min = 0.49;

  auto max_logodds = static_cast<float>(log(0.99) - log(0.01));
  auto min_logodds = static_cast<float>(log(0.01) - log(0.99));

  // Two occupied corners spanning a box of unknown space in between
  OccupancyVDBMapping::PointCloudT::Ptr cloud(new OccupancyVDBMapping::PointCloudT);
  cloud->points.emplace_back(0.0f, 0.0f, 0.0f);
  cloud->points.emplace_back(2.0f, 2.0f, 2.0f);
  cloud->width         = cloud->points.size();
  cloud->height        = 1;
  std::string pcd_path = testing::TempDir() + "vdb_mapping_test_cloud.pcd";
  ASSERT_EQ(pcl::io::savePCDFile(pcd_path, *cloud), 0);

  // Loading into an unconfigured map must fail instead of writing garbage
  {
    OccupancyVDBMapping map(resolution);
    EXPECT_FALSE(map.loadMapFromPCD(pcd_path, true, true));
  }

  OccupancyVDBMapping map(resolution);
  map.setConfig(conf);
  EXPECT_TRUE(map.loadMapFromPCD(pcd_path, true, true));

  OccupancyVDBMapping::GridT::Accessor acc = map.getGrid()->getAccessor();
  // Occupied voxels keep the max log odds and stay active
  EXPECT_FLOAT_EQ(acc.getValue(openvdb::Coord(0, 0, 0)), max_logodds);
  EXPECT_TRUE(acc.isValueOn(openvdb::Coord(0, 0, 0)));
  EXPECT_FLOAT_EQ(acc.getValue(openvdb::Coord(20, 20, 20)), max_logodds);
  // Background voxels inside the bounding box read as free space, inactive
  EXPECT_FLOAT_EQ(acc.getValue(openvdb::Coord(10, 10, 10)), min_logodds);
  EXPECT_FALSE(acc.isValueOn(openvdb::Coord(10, 10, 10)));
  // Voxels outside the bounding box stay unknown
  EXPECT_FLOAT_EQ(acc.getValue(openvdb::Coord(50, 50, 50)), 0.0f);
}

TEST(Mapping, ConfigurableClampingBounds)
{
  // The log-odds clamping bounds (OctoMap Eq. 4) are configurable; repeated
  // hits must saturate at the configured clamp, not at the former hardcoded
  // logit(0.99).
  OccupancyVDBMapping map(1);
  Config conf;
  conf.max_range      = 10;
  conf.fast_mode      = false;
  conf.prob_hit       = 0.9;
  conf.prob_miss      = 0.1;
  conf.prob_thres_max = 0.51;
  conf.prob_thres_min = 0.49;
  conf.prob_clamp_min = 0.2;
  conf.prob_clamp_max = 0.8;
  map.setConfig(conf);
  map.addInputSource("test", conf.max_range, 0);

  auto clamp_max = static_cast<float>(log(conf.prob_clamp_max) - log(1 - conf.prob_clamp_max));
  auto clamp_min = static_cast<float>(log(conf.prob_clamp_min) - log(1 - conf.prob_clamp_min));

  OccupancyVDBMapping::PointCloudT::Ptr cloud(new OccupancyVDBMapping::PointCloudT);
  cloud->points.emplace_back(0, 0, 5);
  Eigen::Matrix<double, 3, 1> origin(0, 0, 0);
  for (int i = 0; i < 10; ++i)
  {
    map.insertPointCloud(cloud, origin, "test");
  }
  OccupancyVDBMapping::GridT::Accessor acc = map.getGrid()->getAccessor();
  EXPECT_FLOAT_EQ(acc.getValue(openvdb::Coord(0, 0, 5)), clamp_max);
  EXPECT_FLOAT_EQ(acc.getValue(openvdb::Coord(0, 0, 1)), clamp_min);

  // Clamps that do not enclose the activation thresholds must be rejected
  Config bad_conf         = conf;
  bad_conf.prob_clamp_max = 0.5; // below prob_thres_max
  std::vector<std::string> errors;
  map.setLogCallback([&](OccupancyVDBMapping::LogLevel level, const std::string& msg) {
    if (level == OccupancyVDBMapping::LogLevel::Error)
    {
      errors.push_back(msg);
    }
  });
  map.setConfig(bad_conf);
  EXPECT_EQ(errors.size(), 1u);
}

TEST(Mapping, MapResetInvalidatesIntersectors)
{
  // VolumeRayIntersector references the grid it was built from without
  // keeping it alive; resetting the map must drop the intersectors instead
  // of leaving them pointing at the replaced grid.
  double resolution = 0.1;
  OccupancyVDBMapping map(resolution);
  setupFastModeMap(map, {{20.0f, 0.0f, 0.0f}});

  bool success;
  openvdb::Vec3d end_point;
  map.raytrace(openvdb::Vec3d(0, 0, 0), openvdb::Vec3d(1, 0, 0), 30.0, success, end_point);
  EXPECT_TRUE(success);

  map.resetMap();
  map.raytrace(openvdb::Vec3d(0, 0, 0), openvdb::Vec3d(1, 0, 0), 30.0, success, end_point);
  EXPECT_FALSE(success);

  // The map must come back to life after new data is inserted
  OccupancyVDBMapping::PointCloudT::Ptr obstacles(new OccupancyVDBMapping::PointCloudT);
  obstacles->points.emplace_back(20.0f, 0.0f, 0.0f);
  map.addPointsToGrid(obstacles);
  OccupancyVDBMapping::PointCloudT::Ptr empty_cloud(new OccupancyVDBMapping::PointCloudT);
  Eigen::Matrix<double, 3, 1> origin(0, 0, 0);
  map.insertPointCloud(empty_cloud, origin, "test");
  map.raytrace(openvdb::Vec3d(0, 0, 0), openvdb::Vec3d(1, 0, 0), 30.0, success, end_point);
  EXPECT_TRUE(success);
  EXPECT_NEAR(end_point.x(), 20.0, resolution);
}

TEST(Mapping, IntegrationPrunesUniformLeaves)
{
  // Stable uniform regions must collapse into tiles after integration
  // (clamping + pruning compression, OctoMap Sect. 3.4)
  OccupancyVDBMapping map(1);
  Config conf;
  conf.max_range      = 100;
  conf.fast_mode      = false;
  conf.prob_hit       = 0.9;
  conf.prob_miss      = 0.1;
  conf.prob_thres_max = 0.51;
  conf.prob_thres_min = 0.49;
  map.setConfig(conf);
  map.addInputSource("test", conf.max_range, 0);

  // Fill one complete 8^3 leaf with saturated occupied voxels
  OccupancyVDBMapping::PointCloudT::Ptr block(new OccupancyVDBMapping::PointCloudT);
  for (int x = 0; x < 8; ++x)
  {
    for (int y = 0; y < 8; ++y)
    {
      for (int z = 0; z < 8; ++z)
      {
        block->points.emplace_back(
          static_cast<float>(x + 64), static_cast<float>(y), static_cast<float>(z));
      }
    }
  }
  map.addPointsToGrid(block);
  EXPECT_GE(map.getGrid()->tree().leafCount(), 1u);

  // Trigger an integration cycle, which prunes the grid
  OccupancyVDBMapping::PointCloudT::Ptr empty_cloud(new OccupancyVDBMapping::PointCloudT);
  Eigen::Matrix<double, 3, 1> origin(0, 0, 0);
  map.insertPointCloud(empty_cloud, origin, "test");

  // The uniform leaf is now a tile; the voxels are still active
  EXPECT_EQ(map.getGrid()->tree().leafCount(), 0u);
  EXPECT_EQ(map.getGrid()->activeVoxelCount(), 512u);
}

TEST(Mapping, MorphologicalDilateErode)
{
  OccupancyVDBMapping map(1);

  OccupancyVDBMapping::UpdateGridT::Ptr grid     = OccupancyVDBMapping::UpdateGridT::create(false);
  OccupancyVDBMapping::UpdateGridT::Accessor acc = grid->getAccessor();
  acc.setValueOn(openvdb::Coord(0, 0, 0), true);
  EXPECT_EQ(grid->activeVoxelCount(), 1u);

  // Dilate by 1 with NN_FACE_EDGE_VERTEX expands to 3x3x3 = 27 voxels
  map.morphologicalDilateMap<OccupancyVDBMapping::UpdateGridT>(grid, 1);
  EXPECT_EQ(grid->activeVoxelCount(), 27u);

  // Erode by 1 should restore the single center voxel
  map.morphologicalErodeMap<OccupancyVDBMapping::UpdateGridT>(grid, 1);
  EXPECT_EQ(grid->activeVoxelCount(), 1u);
}

TEST(Mapping, RaytraceFastModeOffAxisHit)
{
  // Regression: the fast-mode (VolumeRayIntersector) branch of raytrace omitted
  // the +0.5 cell-centered shift that every other DDA in the library uses, so
  // for off-axis rays it traversed the lattice half a voxel off and missed
  // obstacles lying on the geometric ray. The axis-aligned raytrace tests
  // cannot see this because floor(integer)=integer makes both conventions land
  // on the same lattice line.
  double resolution = 0.1;
  OccupancyVDBMapping map(resolution);
  // Single obstacle that the cell-centered ray (1,0,2) from the origin passes
  // through; the un-shifted (buggy) ray walks the neighbouring column and
  // misses it entirely.
  setupFastModeMap(map, {{1.4f, 0.0f, 2.7f}});

  bool success;
  openvdb::Vec3d end_point;
  map.raytrace(openvdb::Vec3d(0, 0, 0), openvdb::Vec3d(1, 0, 2), 5.0, success, end_point);
  EXPECT_TRUE(success);
  EXPECT_NEAR(end_point.x(), 1.4, resolution);
  EXPECT_NEAR(end_point.z(), 2.7, resolution);
}

TEST(Mapping, AddPointsInvalidatesIntersector)
{
  // Regression: addPointsToGrid mutated the grid in place but left the
  // VolumeRayIntersector pointing at the pre-mutation topology, so a fast-mode
  // raytrace before the next integration walked the stale snapshot and missed
  // the newly added obstacle.
  double resolution = 0.1;
  OccupancyVDBMapping map(resolution);
  // Obstacle A far along +x; the integration cycle builds the intersector
  // around its leaf.
  setupFastModeMap(map, {{20.0f, 0.0f, 0.0f}});

  bool success;
  openvdb::Vec3d end_point;
  map.raytrace(openvdb::Vec3d(0, 0, 0), openvdb::Vec3d(1, 0, 0), 30.0, success, end_point);
  EXPECT_TRUE(success);
  EXPECT_NEAR(end_point.x(), 20.0, resolution);

  // Add a nearer obstacle B in a different leaf WITHOUT re-integrating.
  OccupancyVDBMapping::PointCloudT::Ptr b(new OccupancyVDBMapping::PointCloudT);
  b->points.emplace_back(10.0f, 0.0f, 0.0f);
  map.addPointsToGrid(b);

  // The ray must now stop at B (10 m), not the stale A (20 m).
  map.raytrace(openvdb::Vec3d(0, 0, 0), openvdb::Vec3d(1, 0, 0), 30.0, success, end_point);
  EXPECT_TRUE(success);
  EXPECT_NEAR(end_point.x(), 10.0, resolution);
}

TEST(Mapping, DirectGridEditsRequireConfig)
{
  // addPointsToGrid / removePointsFromGrid write m_max_logodds / m_min_logodds,
  // which are uninitialized until setConfig runs; both must refuse before the
  // map is configured instead of storing garbage log odds.
  OccupancyVDBMapping map(0.1);
  OccupancyVDBMapping::PointCloudT::Ptr cloud(new OccupancyVDBMapping::PointCloudT);
  cloud->points.emplace_back(1.0f, 0.0f, 0.0f);

  EXPECT_FALSE(map.addPointsToGrid(cloud));
  EXPECT_FALSE(map.removePointsFromGrid(cloud));
  EXPECT_EQ(map.getGrid()->activeVoxelCount(), 0u);
}

TEST(Mapping, NonFinitePointsAreSkipped)
{
  // Regression: the insertion path only filtered NaN, not inf. Lidar drivers
  // commonly encode no-return points as +/-inf; the max-range clipping turned
  // such a point into a NaN endpoint, Coord::floor(NaN) into INT_MIN, and the
  // DDA then walked ~2^31 voxels toward it (multi-GB allocation / hang).
  double resolution = 0.1;
  OccupancyVDBMapping map(resolution);
  Config conf;
  conf.max_range      = 10;
  conf.fast_mode      = false;
  conf.prob_hit       = 0.9;
  conf.prob_miss      = 0.1;
  conf.prob_thres_max = 0.51;
  conf.prob_thres_min = 0.49;
  map.setConfig(conf);
  map.addInputSource("test", conf.max_range, 0);

  const float inf = std::numeric_limits<float>::infinity();
  OccupancyVDBMapping::PointCloudT::Ptr cloud(new OccupancyVDBMapping::PointCloudT);
  cloud->points.emplace_back(inf, 0.0f, 0.0f);
  cloud->points.emplace_back(-inf, inf, 0.0f);
  cloud->points.emplace_back(std::nanf(""), 0.0f, 0.0f);
  cloud->points.emplace_back(0.0f, 0.0f, 5 * static_cast<float>(resolution));
  Eigen::Matrix<double, 3, 1> origin(0, 0, 0);
  map.insertPointCloud(cloud, origin, "test");

  // Only the finite point and the free voxels along its short ray may exist;
  // the map bounding box must stay tight around it.
  OccupancyVDBMapping::GridT::Accessor acc = map.getGrid()->getAccessor();
  EXPECT_TRUE(acc.isValueOn(openvdb::Coord(0, 0, 5)));
  openvdb::CoordBBox bbox = map.getGrid()->evalActiveVoxelBoundingBox();
  EXPECT_GE(bbox.min().z(), -1);
  EXPECT_LE(bbox.max().z(), 6);
  EXPECT_LE(bbox.max().x(), 1);

  // Direct grid edits must skip non-finite points as well
  map.resetMap();
  map.addPointsToGrid(cloud);
  bbox = map.getGrid()->evalActiveVoxelBoundingBox();
  EXPECT_EQ(map.getGrid()->activeVoxelCount(), 1u);
  EXPECT_LE(bbox.max().x(), 1);
}

TEST(Mapping, RaytraceDegenerateInputs)
{
  // Degenerate queries (zero/non-finite direction, non-finite origin,
  // non-positive or non-finite max length, mismatched batch arrays) must
  // report clean misses instead of feeding NaN into the DDA/intersector or
  // reading out of bounds.
  double resolution = 0.1;
  OccupancyVDBMapping map(resolution);
  setupFastModeMap(map, {{20.0f, 0.0f, 0.0f}});

  const double inf = std::numeric_limits<double>::infinity();
  bool success;
  openvdb::Vec3d end_point;

  map.raytrace(openvdb::Vec3d(0, 0, 0), openvdb::Vec3d(0, 0, 0), 30.0, success, end_point);
  EXPECT_FALSE(success);
  EXPECT_TRUE(end_point == openvdb::Vec3d(0, 0, 0));

  map.raytrace(openvdb::Vec3d(0, 0, 0), openvdb::Vec3d(inf, 0, 0), 30.0, success, end_point);
  EXPECT_FALSE(success);

  map.raytrace(openvdb::Vec3d(inf, 0, 0), openvdb::Vec3d(1, 0, 0), 30.0, success, end_point);
  EXPECT_FALSE(success);

  map.raytrace(openvdb::Vec3d(0, 0, 0), openvdb::Vec3d(1, 0, 0), 0.0, success, end_point);
  EXPECT_FALSE(success);

  map.raytrace(openvdb::Vec3d(0, 0, 0), openvdb::Vec3d(1, 0, 0), -5.0, success, end_point);
  EXPECT_FALSE(success);

  // Mismatched batch arrays must fail cleanly with per-ray misses
  std::vector<openvdb::Vec3d> origins = {openvdb::Vec3d(0, 0, 0), openvdb::Vec3d(1, 0, 0)};
  std::vector<openvdb::Vec3d> dirs    = {openvdb::Vec3d(1, 0, 0)};
  std::vector<double> lengths         = {30.0, 30.0};
  std::vector<bool> successes;
  std::vector<openvdb::Vec3d> end_points;
  map.raytrace(origins, dirs, lengths, successes, end_points);
  ASSERT_EQ(successes.size(), 2u);
  EXPECT_FALSE(successes[0]);
  EXPECT_FALSE(successes[1]);

  // A well-formed query on the same map must still work
  map.raytrace(openvdb::Vec3d(0, 0, 0), openvdb::Vec3d(1, 0, 0), 30.0, success, end_point);
  EXPECT_TRUE(success);
  EXPECT_NEAR(end_point.x(), 20.0, resolution);
}

TEST(Mapping, SaveMapToPCDEmptyMapFails)
{
  // savePCDFile throws on an empty cloud; an empty map must produce a logged
  // error and a false return, not an uncaught exception.
  OccupancyVDBMapping map(1);
  Config conf;
  conf.max_range          = 10;
  conf.fast_mode          = false;
  conf.prob_hit           = 0.9;
  conf.prob_miss          = 0.1;
  conf.prob_thres_max     = 0.51;
  conf.prob_thres_min     = 0.49;
  conf.map_directory_path = testing::TempDir();
  map.setConfig(conf);

  EXPECT_FALSE(map.saveMapToPCD());
}

TEST(Mapping, ByteArrayToGridSelectsCompatibleGrid)
{
  // byteArrayToGrid used to cast only the first grid in the payload; a
  // multi-grid stream whose compatible grid is not first returned nullptr.
  OccupancyVDBMapping map(1);

  OccupancyVDBMapping::UpdateGridT::Ptr bool_grid = OccupancyVDBMapping::UpdateGridT::create(false);
  bool_grid->getAccessor().setValueOn(openvdb::Coord(1, 2, 3), true);
  OccupancyVDBMapping::GridT::Ptr float_grid = OccupancyVDBMapping::GridT::create(0.0f);
  float_grid->getAccessor().setValueOn(openvdb::Coord(4, 5, 6), 1.5f);

  openvdb::GridPtrVec grids;
  grids.push_back(bool_grid);
  grids.push_back(float_grid);
  std::ostringstream oss(std::ios_base::binary);
  openvdb::io::Stream(oss).write(grids);
  std::vector<uint8_t> bytes = map.compressString(oss.str());

  auto restored = map.byteArrayToGrid<OccupancyVDBMapping::GridT>(bytes);
  ASSERT_NE(restored, nullptr);
  EXPECT_FLOAT_EQ(restored->getAccessor().getValue(openvdb::Coord(4, 5, 6)), 1.5f);
}

TEST(Mapping, ClampMustStrictlyEncloseThresholds)
{
  // A clamp equal to an activation threshold lets a voxel saturate exactly AT
  // the threshold and never cross the strict comparison, so a saturated
  // obstacle would stay inactive. Such configs must be rejected.
  OccupancyVDBMapping map(1);
  std::vector<std::string> errors;
  map.setLogCallback([&](OccupancyVDBMapping::LogLevel level, const std::string& msg) {
    if (level == OccupancyVDBMapping::LogLevel::Error)
    {
      errors.push_back(msg);
    }
  });

  Config conf;
  conf.max_range      = 10;
  conf.fast_mode      = false;
  conf.prob_hit       = 0.9;
  conf.prob_miss      = 0.1;
  conf.prob_thres_max = 0.51;
  conf.prob_thres_min = 0.49;

  // clamp_max exactly equal to thres_max must be rejected
  conf.prob_clamp_min = 0.01;
  conf.prob_clamp_max = 0.51;
  map.setConfig(conf);
  EXPECT_EQ(errors.size(), 1u);

  // clamp_min exactly equal to thres_min must be rejected
  errors.clear();
  conf.prob_clamp_max = 0.99;
  conf.prob_clamp_min = 0.49;
  map.setConfig(conf);
  EXPECT_EQ(errors.size(), 1u);
}

} // namespace vdb_mapping

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
