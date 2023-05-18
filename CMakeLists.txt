cmake_minimum_required(VERSION 3.0.0)
project(oSLAM VERSION 0.1.0)

include(CTest)
enable_testing()

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})

find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

find_package(g2o REQUIRED)
include_directories(${G2O_INCLUDE_DIR})

file(GLOB_RECURSE cpp_srcs src/*.cpp)

add_executable(oSLAM ${cpp_srcs})
target_link_libraries(oSLAM ${OpenCV_LIBS})
target_link_libraries(oSLAM ${Pangolin_LIBRARIES})
target_link_libraries(oSLAM g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
