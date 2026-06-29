cmake_minimum_required(VERSION 3.16.3)
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules" ${CMAKE_MODULE_PATH})

option(BUILDING_TESTS "Build unit tests." ON)

project(vdb_mapping VERSION 0.1.0 LANGUAGES CXX C)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  message("${PROJECT_NAME}: You did not request a specific build type: selecting 'Release'.")
  set(CMAKE_BUILD_TYPE Release)
endif()

## System dependencies are found with CMake's conventions
find_package(Eigen3 REQUIRED)
find_package(OpenVDB REQUIRED)
find_package(PCL REQUIRED)
find_package(TBB REQUIRED)
find_package(ZSTD REQUIRED)

###########
## Build ##
###########


add_library(${PROJECT_NAME} INTERFACE)


### Declare a C++ library
target_include_directories(${PROJECT_NAME}
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${PCL_INCLUDE_DIRS}
    ${ZSTD_INCLUDE_DIR}
    )

# Imath/Half is linked transitively through OpenVDB::openvdb (see
# FindOpenVDB.cmake's OpenVDB_USES_IMATH_HALF handling); hardcoding -lImath
# here breaks systems where the library name differs.
target_link_libraries(${PROJECT_NAME}
  INTERFACE
    OpenVDB::openvdb
    Eigen3::Eigen
    ${PCL_LIBRARIES}
    ${ZSTD_LIBRARY}
    )

target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_17)

add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

##
## Build testing if enabled by option
##
if (BUILDING_TESTS)
  enable_testing()
  add_subdirectory(tests)
else()
  message(STATUS "Building tests disabled.")
endif()

#############
## Install ##
#############

install(TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}_targets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

install(
  EXPORT ${PROJECT_NAME}_targets
  DESTINATION lib/cmake/${PROJECT_NAME}
  FILE ${PROJECT_NAME}Targets.cmake
  NAMESPACE ${PROJECT_NAME}::
  )

## Mark cpp header files for installation
 install(
   DIRECTORY include/${PROJECT_NAME}/
   DESTINATION include/${PROJECT_NAME}
   FILES_MATCHING
   PATTERN "*.h"
   PATTERN "*.hpp"
 )

include(CMakePackageConfigHelpers)
write_basic_package_version_file(${PROJECT_NAME}ConfigVersion.cmake VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion)
install(
  FILES
    ${PROJECT_NAME}Config.cmake ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
    CMakeModules/FindOpenVDB.cmake
    CMakeModules/OpenVDBUtils.cmake
    CMakeModules/FindBlosc.cmake
    CMakeModules/FindZSTD.cmake
  DESTINATION
    lib/cmake/${PROJECT_NAME}
  )
