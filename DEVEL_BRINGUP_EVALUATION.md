# Devel Bring-up Evaluation

Evaluation of which features from the fork's `devel` branch are worthwhile bringing up
into `main`. This is an **assessment only** — no source has been changed and nothing has
been cherry-picked. Use the commands at the bottom to bring up whichever commits you choose.

## Branch topology

`main` is identical to upstream `fzi-forschungszentrum-informatik/vdb_mapping` `main`.
The fork's `devel` is a **linear stack of 18 commits** on top of `main`:

```
cddc6be  Boost 1.90 / CMake 4 compat          ┐
3af4855  cleanup: createVDBMap param           │
74f8c3c  perf: clip applyMapSectionUpdateGrid  │
750f0a3  fix: shallow-copy ray intersector     │
6c00fea  fix: serialise artificial-area writes │
1b06507  fix: align worldToIndex rounding      │  15 fork-authored commits
4902867  fix: validate accumulation_period     │  (the features to evaluate)
5efd016  fix: C++20 warnings + ZSTD API        │
183741c  fix: self-includes + raytrace success │
22c8110  fix: thread safety + null guard +tests│
08e4788  docs: CLAUDE.md libzstd-dev           │
a706ef6  package.xml ROS2 format 3             │
e07560a  perf: raycasting hot loop             │
a01da14  fix: mapping accuracy                 │
8dd882b  docs: add CLAUDE.md                   ┘
68bf9d9  removed iron from ci          ┐  3 upstream-devel commits
d7a5444  added transformable map sect. │  (base the fork work sits on;
7257e76  Added getResolution           ┘   68bf9d9 = "last pre-fork" commit)
5f524a3  set clang tidy version ........ main / upstream main
```

`68bf9d9` is the last upstream commit before the fork's own work begins. Because `devel`
is strictly linear on top of `main`, a full bring-up is effectively a fast-forward.
Selective cherry-pick is possible, but the commits that touch the same code twice must
keep their relative order (see notes).

## Reference SHAs

| Short | Full |
|-------|------|
| cddc6be | `cddc6be0789fad67e68f1d35500e50e326cc97f5` (fork `devel` tip) |
| 68bf9d9 | `68bf9d97f047fc911c50e69b71c8439dc9a85aee` (upstream `devel` tip) |
| 5f524a3 | `5f524a3d624131dbf8570614ba9613e45c8f4ebd` (`main`) |

## Evaluation of the 15 fork commits

### Correctness fixes — high value, recommend bring-up
- **`a01da14` mapping accuracy** — simplify `worldToIndex` to direct index-space rounding
  (the old `fmod` boundary checks were effectively a no-op); move occupancy config
  validation before the base `setConfig`; fix single-ray raytrace passing a scalar where
  the batch overload expects the `max_ray_length` vector (latent compile error).
- **`1b06507` align `worldToIndex` rounding across mutators** — `addPointsToGrid`,
  `removePointsFromGrid`, and `createMapFromPointCloud` used `Coord::floor` while the
  raycast path uses round-to-nearest, so a point near a voxel boundary could be added to
  one cell and removed from its neighbour, leaving a stale occupied voxel. *Refines
  `a01da14`; take both, in order.*
- **`183741c` correct raytrace success reporting** — when `VolumeRayIntersector::march()`
  returned a tile intersection but the DDA found no active voxel, success was reported
  unconditionally (phantom hits for Nav2 obstacle / LoC-recovery queries). Now reports
  success only when the loop exits on an active voxel. Also removes dead self-includes.

### Thread-safety fixes — high value for multithreaded / remote-sync use
- **`22c8110` thread safety + null guard + tests** — `m_config_set` → `std::atomic<bool>`
  (real data race between `setConfig` and the accumulation/integration threads that
  spin-wait on it); null guard in batch raytrace; adds `InvalidConfigRejected`,
  `GridSerialization`, `MorphologicalDilateErode` tests.
- **`750f0a3` shallow-copy volume ray intersector per call** — `m_volume_ray_intersector`
  is shared but `setIndexRay`/`march` mutate its state; concurrent callers under the map
  lock raced. Takes a per-call shallow copy (the pattern already used for per-source
  intersectors).
- **`6c00fea` serialise artificial-area writes through the map mutex** —
  `addArtificialAreas`/`restoreMapIntegrity` were unlocked while the integration thread
  reads `m_artificial_area_grid` under the unique lock. Locks at the public API and splits
  bodies into `_Locked` helpers.

### Build / toolchain & safety — high value, likely build-blocking on modern systems
- **`cddc6be` Boost 1.90 / CMake 4 compat** — `FindOpenVDB.cmake` drops `Boost::system`
  (Boost 1.90 made Boost.System header-only; the component no longer exists and CMake 4
  removed the legacy `FindBoost`); bump `tests/CMakeLists.txt` `cmake_minimum_required`
  3.0.2 → 3.5 (CMake 4 rejects < 3.5).
- **`5efd016` C++20 warnings + ZSTD API** — replace deprecated `ZSTD_getDecompressedSize`
  with `ZSTD_getFrameContentSize`. The old API returned 0 for unknown sizes; the new one
  returns sentinels, so without the added check a corrupt/headerless frame could allocate
  ~18 EB and crash — a genuine safety fix. Plus `[[maybe_unused]]` to silence C++20
  pedantic warnings on the virtual no-op defaults.

### Performance
- **`74f8c3c` clip `applyMapSectionUpdateGrid` to overlapping leaves** — the deactivation
  pass iterated every active voxel in the whole map; switch to the leaf-overlap pattern
  `getMapSection` already uses. O(global active voxels) → O(active voxels in overlapping
  leaves); large win on big maps with periodic remote sync.
- **`e07560a` raycast hot-loop micro-opts** — drop dead `RayT`/`DDAT` allocations per
  call, hoist the constant NaN-origin check out of the per-point loop, iterate point
  clouds by `const auto&` instead of copying each `pcl::PointXYZ`.

### Robustness / cleanup
- **`4902867` validate `accumulation_period` + default-init configs** — reject
  non-positive periods (the int-ms cast wrapped for negatives / busy-spun at zero);
  default-initialize `Config`/`BaseConfig` members (callers relied on UB).
- **`3af4855` cleanup** — drop the unused `resolution` parameter from `createVDBMap` (the
  transform is built from `m_resolution`) at all three call sites; remove the redundant
  `m_config_set = true` in `OccupancyVDBMapping::setConfig`.

### Docs / packaging — useful, low-risk, optional
- **`a706ef6` `package.xml` → ROS2 format 3** — add `libzstd-dev` (used by serialization
  but missing → rosdep failures), `libgtest-dev` test dep, repo URL, second author.
  Genuinely useful for ROS2 consumers.
- **`8dd882b` + `08e4788` CLAUDE.md** — AI-assistant context documentation only.

## Recommendation

The fork `devel` is a coherent line of bug-fixes plus compatibility work over a stable
upstream base — **all 18 commits are worth bringing into `main`**. If bringing them in
selectively, prioritise the correctness, thread-safety, ZSTD-safety, and build-compat
groups; the `CLAUDE.md` docs are optional.

Ordering constraints when cherry-picking:
- The `worldToIndex` pair `a01da14` → `1b06507` must stay in order.
- The config-validation commits `a01da14` → `4902867` → `22c8110` must stay in order.
- Any fork commit implicitly needs the 3 upstream commits (`7257e76`, `d7a5444`,
  `68bf9d9`) as its base; taking the whole stack pulls them in automatically.

## Ready-to-use bring-up commands

The fork's `devel` is on the same `origin`, so no extra remote is needed:

```bash
# Fetch the fork devel branch (cddc6be)
git fetch origin devel

# Create the bring-up branch off main
git checkout -b devel-bringup main

# Option A — bring up the entire stack (linear, fast-forward)
git merge --ff-only origin/devel

# Option B — selective cherry-pick, preserving order (example: everything)
git cherry-pick 7257e76^..cddc6be
```
