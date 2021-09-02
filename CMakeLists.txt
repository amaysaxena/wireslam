cmake_minimum_required(VERSION 3.1)

project(wireslam)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "-O3 -funroll-loops -march=native -I/usr/local/include -L/usr/local/lib")

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

find_package( Boost REQUIRED )
include_directories(${Boost_INCLUDE_DIRS})

find_package(GTSAM REQUIRED)
include_directories(${GTSAM_INCLUDE_DIR})

find_package(TBB)

include_directories(src)

add_executable(wireslam-demo src/main_line)

target_link_libraries(wireslam-demo TBB::tbb Eigen3::Eigen gtsam)