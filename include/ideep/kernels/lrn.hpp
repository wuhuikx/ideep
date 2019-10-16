#ifndef IDEEP_KERNELS_LRN_HPP
#define IDEEP_KERNELS_LRN_HPP

namespace ideep {

struct lrn_forward : public dnnl::lrn_forward {
  static void compute(const tensor& src, tensor& dst, int local_size, float alpha, float beta,
      float k = 1.0, algorithm aalgorithm = algorithm::lrn_across_channels,
      prop_kind aprop_kind = prop_kind::forward_training) {
  }
};

struct lrn_backward : public dnnl::lrn_backward {
  static void compute(const tensor& x, const tensor& grady, const tensor& y, tensor& gradx,
      int local_size, float alpha, float beta, float k = 1.0,
      algorithm aalgorithm = algorithm::lrn_across_channels) {
  }
};

}  // namespace ideep

#endif