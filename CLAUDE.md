# CLAUDE.md

## Project Overview

**vdb_mapping** is a C++ header-only library for high-resolution, real-time 3D volumetric mapping built on top of [OpenVDB](https://www.openvdb.org/). It provides occupancy grid mapping using probabilistic (log-odds) updates with raycasting from sensor point clouds. Developed by FZI Forschungszentrum Informatik, it is designed for use with mobile robots and is typically deployed via its [ROS wrapper](https://github.com/fzi-forschungszentrum-informatik/vdb_mapping_ros) or [ROS2 wrapper](https://github.com/fzi-forschungszentrum-informatik/vdb_mapping_ros2).

**License:** Apache-2.0

## Repository Structure

```
vdb_mapping/
├── include/vdb_mapping/
│   ├── VDBMapping.hpp          # Core template class - raycasting, map management, threading, serialization
│   └── OccupancyVDBMapping.hpp # Occupancy-specific subclass with log-odds probability updates
├── tests/
│   ├── CMakeLists.txt          # Test build config (uses GTest)
│   ├── mapping.cpp             # Unit tests (Mapping + MappingSources suites)
│   └── tsan.supp               # ThreadSanitizer suppressions for race-checking the suite
├── CMakeModules/
│   ├── FindOpenVDB.cmake       # Custom CMake find modules
│   ├── FindBlosc.cmake
│   ├── FindZSTD.cmake
│   ├── FindGTestPackage.cmake
│   └── OpenVDBUtils.cmake
├── CMakeLists.txt              # Root build file (ament_cmake_auto)
├── vdb_mappingConfig.cmake     # CMake package config for downstream consumers
├── package.xml                 # ROS/ROS2 package manifest (build type: ament_cmake)
├── .clang-format               # Code formatting rules
├── .gitlab-ci.yml              # CI pipeline config
└── .gitignore
```

## Architecture

### Header-Only Library

The library is entirely header-only with an `INTERFACE` CMake target. All code lives in two `.hpp` files under `include/vdb_mapping/`.

### Class Hierarchy

- **`VDBMapping<TData, TConfig>`** (`VDBMapping.hpp`) - Template base class parameterized on voxel data type and config type. Provides:
  - Grid creation and lifecycle (`createVDBMap`, `resetMap`, `saveMap`, `loadMap`)
  - Point cloud integration via raycasting (`insertPointCloud`, `accumulateUpdate`, `integrateUpdate`)
  - Two raycasting modes: standard DDA (`castRayIntoGrid`) and fast ray-marching (`castRayIntoGridFast`)
  - Map section extraction and application with bounding boxes
  - Morphological operations (dilate, erode, open, close)
  - Artificial area/wall injection for virtual obstacles
  - Grid serialization with ZSTD compression (`gridToByteArray`, `byteArrayToGrid`)
  - Batch and single raytracing queries (`raytrace`)
  - Multi-threaded accumulation pipeline with per-source worker threads and a dedicated integration thread
  - Per-source integration semantics: registered sources may override `prob_hit`/`prob_miss` and select roles via `ray_clearing`/`endpoint_hits`; a per-source `max_range <= 0` falls back to the map-level `max_range` at use time (never latched at registration, so a source configured before the map cannot go permanently dead)
  - Injectable time source (`setTimeCallback`) and explicit `stop()` for deterministic tests and replay
  - Thread-safe map access using `std::shared_mutex`

- **`OccupancyVDBMapping`** (`OccupancyVDBMapping.hpp`) - Concrete subclass using `float` voxel data. Implements:
  - Log-odds probability updates for occupied/free nodes
  - Activation thresholds `prob_thres_min`/`prob_thres_max` (defaults **0.49/0.51** — a single clean hit activates). These are NOT the OctoMap 0.12/0.97 pair: those are log-odds CLAMPING bounds, kept separately as `prob_clamp_*`; used as activation thresholds they cost ~5 accumulation windows before an obstacle appears at all
  - Config validation (hit probability > 0.5, miss probability < 0.5; clamping bounds must strictly enclose the activation thresholds)
  - Point cloud to map creation

### Key Types

```cpp
template <typename TData, typename TConfig = BaseConfig, typename PointType = pcl::PointXYZ>
class VDBMapping;         // templatized on the point type since b8821d7

using PointT      = PointType;                // pcl::PointXYZ by default
using PointCloudT = pcl::PointCloud<PointT>;
using GridT       = openvdb::Grid<openvdb::tree::Tree4<TData, 5, 4, 3>::Type>;
using UpdateGridT = openvdb::Grid<openvdb::tree::Tree4<bool, 1, 4, 3>::Type>;
```

### Threading Model

- One **integration thread** runs continuously, periodically merging accumulated updates into the main map
- One **accumulation worker thread** per registered input source, waiting on condition variables for new data
- Map access is protected by a `std::shared_mutex` (shared for reads, exclusive for writes)
- A priority mechanism (`m_map_mutex_requested`) lets the integration thread signal accumulation threads to yield

## Build Instructions

### Dependencies

```bash
apt-get install -y libeigen3-dev libtbb-dev libpcl-dev libilmbase-dev libzstd-dev
```

OpenVDB >= 8.3 is required (v9.0.0+ from source recommended since apt packages are outdated).

### Build (standalone)

```bash
mkdir build && cd build
cmake ..
make -j8
```

### Build with tests

Tests are enabled by default (`BUILDING_TESTS=ON`). To disable:

```bash
cmake -DBUILDING_TESTS=OFF ..
```

### Run tests

```bash
cd build
ctest
# or directly:
./tests/mapping_tests
```

### ROS/ROS2 workspace

This is an **ament_cmake** package built with `ament_cmake_auto` (converted from plain cmake in f92fe50). In a ROS 2 workspace: `colcon build --packages-select vdb_mapping`. The standalone cmake build above still works, but needs the ROS 2 environment sourced so `ament_cmake_auto` resolves.

## CI/CD

CI is configured via `.gitlab-ci.yml` and runs on an internal GitLab instance at FZI. It tests against a matrix of ROS distributions:
- **noetic** (Ubuntu 20.04) - with clang-tidy version 12
- **humble** (Ubuntu 22.04)
- **jazzy** (Ubuntu 24.04)
- **rolling** (Ubuntu 24.04)

The pipeline is defined externally in `continuous_integration/ci_scripts` using `fla_pipeline.yml`.

## Code Style and Conventions

### Formatting

Code formatting is enforced via `.clang-format`. Key settings:
- **Column limit:** 100
- **Indent width:** 2 spaces (no tabs)
- **Brace style:** Custom (Allman-like with braces on new lines for classes, structs, functions, control statements)
- **Pointer alignment:** Left (`int* ptr`)
- **Namespace indentation:** None
- **Constructor initializers:** Break before comma
- **Template declarations:** Always break
- **Short functions:** Inline only
- **Sort includes:** Yes

Run formatting with: `clang-format -i <file>`

### Naming Conventions

- **Classes:** PascalCase (`VDBMapping`, `OccupancyVDBMapping`)
- **Member variables:** `m_` prefix with snake_case (`m_vdb_grid`, `m_max_range`, `m_logodds_hit`)
- **Methods:** camelCase (`insertPointCloud`, `resetMap`, `accumulateUpdate`)
- **Type aliases:** PascalCase with `T` suffix (`PointT`, `GridT`, `UpdateGridT`, `RayT`)
- **Template parameters:** `T` prefix PascalCase (`TData`, `TConfig`, `TGrid`, `TResultGrid`)
- **Config structs:** PascalCase (`BaseConfig`, `Config`)
- **Constants/enums:** Not commonly used; probabilities stored as member floats
- **Namespaces:** snake_case (`vdb_mapping`)

### File Structure

- License block (Apache 2.0) at the top of every source file
- Emacs mode line at very first line: `// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-`
- Author/date block with `\author` and `\date` doxygen tags
- Include guards using `#ifndef`/`#define`/`#endif` (not `#pragma once`)
- Guard naming: `VDB_MAPPING_<FILENAME>_H_INCLUDED`

### Documentation

- Doxygen-style `/*! \brief */` comments for all public/protected methods
- `\param`, `\returns`, `@tparam` tags for parameters and return values
- Inline comments for non-obvious logic

### C++ Standard

- **C++17** required (uses `std::optional`, `std::shared_mutex`, structured bindings, `if constexpr`)
- Tests require at minimum C++14

## Testing

Tests use **Google Test (GTest)** and are located in `tests/mapping.cpp` — 44 tests in two suites, all built on `OccupancyVDBMapping` with log-odds verification:

- **`Mapping`** — config validation and activation-threshold defaults (`DefaultConfigActivatesOnFirstHit`, `ConfigurableClampingBounds`, `ClampMustStrictlyEncloseThresholds`), insertion/raycasting on and off fast mode (including degenerate and non-finite inputs), serialization round-trips and garbage rejection, map sections with pruned tiles, morphological ops, cell-centered coordinate rounding, and threading/liveness (`ExplicitStopStopsThreads`, `TimeCallbackOverridesSystemTime`, `SleepEpochShiftDoesNotFreeze`).
- **`MappingSources`** — the per-source semantics the sonar integration relies on: clearing-only sources carve without ever adding hits, hit-only sources with beyond-range behavior, cross-source superimposition, per-source overrides applying through integration, hit dominance when same-source clouds merge, window dedup of repeated hits, and `SourceRegisteredBeforeConfigComesAlive` (the use-time `max_range` fallback).

`tests/tsan.supp` carries ThreadSanitizer suppressions so the whole suite race-checks in one command: build with `-fsanitize=thread` and run with `TSAN_OPTIONS=suppressions=<repo>/tests/tsan.supp`.

## Development Notes

- The library is **under active development** - interfaces may change
- The `devel` branch is the primary development branch
- The main grid uses a `Tree4` structure with node sizes `5, 4, 3` for optimal memory/performance
- The update grid uses a sparser `Tree4<bool, 1, 4, 3>` structure
- ZSTD compression is used for grid serialization (compression level 1 by default)
- `fast_mode` uses `VolumeRayIntersector` for ray-marching through existing occupied regions rather than full DDA traversal
- Map sections can be extracted, serialized, transmitted, and applied - designed for distributed/remote mapping scenarios
