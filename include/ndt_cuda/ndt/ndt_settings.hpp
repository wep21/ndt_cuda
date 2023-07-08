#ifndef NDT_CUDA__NDT__NDT_SETTINGS_HPP_
#define NDT_CUDA__NDT__NDT_SETTINGS_HPP_

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

#endif  // NDT_CUDA__NDT__NDT_SETTINGS_HPP_
