cmake_minimum_required(VERSION 3.18)
project(ndt_cuda)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

include(CheckLanguage)
check_language(CUDA)
if(NOT CMAKE_CUDA_COMPILER)
  message(WARNING "CUDA not found")
  return()
endif()

enable_language(CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL COMPONENTS common filters io registration search REQUIRED)

add_library(${PROJECT_NAME} SHARED
  src/cuda/covariance_regularization.cu
  src/cuda/find_voxel_correspondences.cu
  src/cuda/gaussian_voxelmap.cu
  src/cuda/ndt_compute_derivatives.cu
  src/cuda/ndt_cuda.cu
  src/ndt/lsq_registration.cpp
  src/ndt/ndt_cuda.cpp
)

target_include_directories(${PROJECT_NAME}
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:include/${PROJECT_NAME}>"
)

target_link_libraries(${PROJECT_NAME}
  Eigen3::Eigen
  pcl_common
  pcl_filters
  pcl_registration
  pcl_search
)

add_executable(${PROJECT_NAME}_align
  src/align.cpp
)

target_link_libraries(${PROJECT_NAME}_align
  ${PROJECT_NAME}
  pcl_io
)

install(
  EXPORT export_${PROJECT_NAME}
  DESTINATION share/${PROJECT_NAME}/cmake
  NAMESPACE ${PROJECT_NAME}::
  FILE export_${PROJECT_NAME}Export.cmake
)

install(
  TARGETS ${PROJECT_NAME} EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(
  TARGETS ${PROJECT_NAME}_align
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY include/
  DESTINATION include/${PROJECT_NAME}
)
