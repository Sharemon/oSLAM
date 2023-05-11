cmake_minimum_required(VERSION 3.0.0)
project(oSLAM VERSION 0.1.0)

include(CTest)
enable_testing()

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

file(GLOB_RECURSE cpp_srcs src/*.cpp)

add_executable(oSLAM ${cpp_srcs})
target_link_libraries(oSLAM ${OpenCV_LIBS})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)