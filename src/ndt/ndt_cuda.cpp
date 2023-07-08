#include <ndt_cuda/ndt/ndt_cuda.hpp>
#include <ndt_cuda/ndt/impl/ndt_cuda_impl.hpp>

template class ndt_cuda::NDTCuda<pcl::PointXYZ, pcl::PointXYZ>;
template class ndt_cuda::NDTCuda<pcl::PointXYZI, pcl::PointXYZI>;
template class ndt_cuda::NDTCuda<pcl::PointNormal, pcl::PointNormal>;