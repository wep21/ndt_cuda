#ifndef NDT_CUDA_NDT_SETTINGS_HPP
#define NDT_CUDA_NDT_SETTINGS_HPP

namespace ndt_cuda {

enum class NDTDistanceMode { P2D, D2D };

enum class NeighborSearchMethod {
  DIRECT27,
  DIRECT7,
  DIRECT1,
  /* supported on only VGICP_CUDA */ DIRECT_RADIUS
};

enum class RegularizationMethod { NONE, MIN_EIG, NORMALIZED_MIN_EIG, PLANE, FROBENIUS };

}  // namespace ndt_cuda

#endif