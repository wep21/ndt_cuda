#ifndef NDT_CUDA__NDT__NDT_CUDA_HPP_
#define NDT_CUDA__NDT__NDT_CUDA_HPP_

#include <memory>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <ndt_cuda/ndt/lsq_registration.hpp>
#include <ndt_cuda/ndt/ndt_settings.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/registration.h>
#include <pcl/search/kdtree.h>

namespace ndt_cuda {

namespace cuda {
class NDTCudaCore;
}

template <typename PointSource, typename PointTarget>
class NDTCuda : public LsqRegistration<PointSource, PointTarget> {
public:
  using Scalar = float;
  using Matrix4 = typename pcl::Registration<PointSource, PointTarget, Scalar>::Matrix4;

  using PointCloudSource =
    typename pcl::Registration<PointSource, PointTarget, Scalar>::PointCloudSource;
  using PointCloudSourcePtr = typename PointCloudSource::Ptr;
  using PointCloudSourceConstPtr = typename PointCloudSource::ConstPtr;

  using PointCloudTarget =
    typename pcl::Registration<PointSource, PointTarget, Scalar>::PointCloudTarget;
  using PointCloudTargetPtr = typename PointCloudTarget::Ptr;
  using PointCloudTargetConstPtr = typename PointCloudTarget::ConstPtr;

  using Ptr = pcl::shared_ptr<NDTCuda<PointSource, PointTarget>>;
  using ConstPtr = pcl::shared_ptr<const NDTCuda<PointSource, PointTarget>>;

protected:
  using pcl::Registration<PointSource, PointTarget, Scalar>::reg_name_;
  using pcl::Registration<PointSource, PointTarget, Scalar>::input_;
  using pcl::Registration<PointSource, PointTarget, Scalar>::target_;
  using pcl::Registration<PointSource, PointTarget, Scalar>::corr_dist_threshold_;

public:
  NDTCuda();
  ~NDTCuda() override;

  void setDistanceMode(NDTDistanceMode mode);
  void setResolution(double resolution);
  void setNeighborSearchMethod(NeighborSearchMethod method, double radius = -1.0);

  void swapSourceAndTarget() override;
  void clearSource() override;
  void clearTarget() override;

  void setInputSource(const PointCloudSourceConstPtr& cloud) override;
  void setInputTarget(const PointCloudTargetConstPtr& cloud) override;

protected:
  void computeTransformation(PointCloudSource& output, const Matrix4& guess) override;
  double linearize(const Eigen::Isometry3d& trans, Eigen::Matrix<double, 6, 6>* H,
                           Eigen::Matrix<double, 6, 1>* b) override;
  double compute_error(const Eigen::Isometry3d& trans) override;

protected:
  std::unique_ptr<cuda::NDTCudaCore> ndt_cuda_;
};
}  // namespace ndt_cuda

#endif  // NDT_CUDA__NDT__NDT_CUDA_HPP_
