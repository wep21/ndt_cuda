#ifndef NDT_CUDA_COVARIANCE_REGULARIZATION_CUH
#define NDT_CUDA_COVARIANCE_REGULARIZATION_CUH

#include <Eigen/Core>
#include <thrust/device_vector.h>
#include <ndt_cuda/ndt/ndt_settings.hpp>

namespace ndt_cuda {
namespace cuda {

void covariance_regularization(thrust::device_vector<Eigen::Vector3f>& means, thrust::device_vector<Eigen::Matrix3f>& covs, RegularizationMethod method);

}  // namespace cuda
}  // namespace ndt_cuda

#endif