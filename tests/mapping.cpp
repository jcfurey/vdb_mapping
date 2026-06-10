#include "gtest/gtest.h"
#include <array>
#include <vector>
#include <vdb_mapping/OccupancyVDBMapping.hpp>

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
  Config bad_conf  = conf;
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
void setupFastModeMap(OccupancyVDBMapping& map,
                      const std::vector<std::array<float, 3> >& obstacles)
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
  auto grid = map.byteArrayToGrid<OccupancyVDBMapping::GridT>(garbage);
  EXPECT_EQ(grid, nullptr);
}

TEST(Mapping, MorphologicalDilateErode)
{
  OccupancyVDBMapping map(1);

  OccupancyVDBMapping::UpdateGridT::Ptr grid = OccupancyVDBMapping::UpdateGridT::create(false);
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
