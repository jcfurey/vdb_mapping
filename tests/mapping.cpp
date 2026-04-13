#include "gtest/gtest.h"
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
