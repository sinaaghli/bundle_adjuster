cmake_minimum_required(VERSION 2.8.10)
project(bundle_adjuster)

# Include local cmake modules directory.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules/")

# Default flags.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -Wall -Wextra")

option(WITH_GUI "Enable in order to build GUI" OFF)

find_package(OpenCV2 REQUIRED)
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})
include_directories(${OpenCV2_INCLUDE_DIR})

set( HDRS  BALProblem.h
           GLPathAbs.h
           GLPathRel.h
           PoseLandmarkContainer.h)
set( SRCS  main.cpp )

if(WITH_GUI)
    find_package(Eigen3 REQUIRED)
    find_package(Sophus REQUIRED)
    find_package(HAL 0.1 REQUIRED)
    find_package(Pangolin 0.1 REQUIRED)
    find_package(SceneGraph 0.1 REQUIRED)
    find_package(Calibu 0.1 REQUIRED)
    find_package(PbMsgs 0.1 REQUIRED)
    find_package(Protobuf REQUIRED)


    include_directories(${EIGEN3_INCLUDE_DIR})
    include_directories(${HAL_INCLUDE_DIRS})
    include_directories(${Sophus_INCLUDE_DIR})
    include_directories(${Pangolin_INCLUDE_DIRS})
    include_directories(${SceneGraph_INCLUDE_DIRS})
    include_directories(${Calibu_INCLUDE_DIRS})
    include_directories(${PbMsgs_INCLUDE_DIRS})
    include_directories(${PROTOBUF_INCLUDE_DIRS})


    link_directories(${Pangolin_LIBRARY_DIRS})

    list(APPEND HDRS )
    list(APPEND SRCS )

    add_definitions(-DWITH_GUI)
endif()

add_executable(bundle_adjuster ${HDRS} ${SRCS})
add_executable(SIFT_FlannMatcher SIFT_FlannMatcher.cpp)

target_link_libraries(SIFT_FlannMatcher ${CERES_LIBRARIES} ${OpenCV2_LIBRARIES})
target_link_libraries(bundle_adjuster ${CERES_LIBRARIES})
if(WITH_GUI)
    target_link_libraries(bundle_adjuster ${HAL_LIBRARIES})
    target_link_libraries(bundle_adjuster ${OpenCV2_LIBRARIES})
    target_link_libraries(bundle_adjuster ${Pangolin_LIBRARIES})
    target_link_libraries(bundle_adjuster ${SceneGraph_LIBRARIES})
    target_link_libraries(bundle_adjuster ${Calibu_LIBRARIES})
    target_link_libraries(bundle_adjuster ${PbMsgs_LIBRARIES})
    target_link_libraries(bundle_adjuster ${PROTOBUF_LIBRARIES})

endif()
