#include <ndt_cuda/ndt/impl/lsq_registration_impl.hpp>
#include <ndt_cuda/ndt/lsq_registration.hpp>

template class ndt_cuda::LsqRegistration<pcl::PointXYZ, pcl::PointXYZ>;
template class ndt_cuda::LsqRegistration<pcl::PointXYZI, pcl::PointXYZI>;
template class ndt_cuda::LsqRegistration<pcl::PointNormal, pcl::PointNormal>;