#include "gtest/gtest.h"
#include <array>
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

} // namespace vdb_mapping

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
