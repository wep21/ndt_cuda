#ifndef NDT_CUDA_FIND_VOXEL_CORRESPONDENCES_CUH
#define NDT_CUDA_FIND_VOXEL_CORRESPONDENCES_CUH

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <thrust/pair.h>
#include <thrust/device_vector.h>
#include <ndt_cuda/cuda/gaussian_voxelmap.cuh>

namespace ndt_cuda {
namespace cuda {

void find_voxel_correspondences(
  const thrust::device_vector<Eigen::Vector3f>& src_points,
  const GaussianVoxelMap& voxelmap,
  const thrust::device_ptr<const Eigen::Isometry3f>& x_ptr,
  const thrust::device_vector<Eigen::Vector3i>& offsets,
  thrust::device_vector<thrust::pair<int, int>>& correspondences) ;

}  // namespace cuda
}  // namespace ndt_cuda

#endif