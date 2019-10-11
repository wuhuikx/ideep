#ifndef IDEEP_KERNELS_BATCHNORM_HPP
#define IDEEP_KERNELS_BATCHNORM_HPP

#include "common.hpp"

namespace ideep {

struct batch_normalization_forward_inference
    : public dnnl::batch_normalization_forward {
  static void compute(const tensor& src, const tensor& scale,
                      const tensor& shift, tensor& dst, float epsilon) {
    
  }

  static void compute(const tensor& src, const tensor& mean,
                      const tensor& variance, const tensor& scale,
                      const tensor& shift, tensor& dst, float epsilon) {

  }
};

struct batch_normalization_forward_training
    : public dnnl::batch_normalization_forward {
  float get_epsilon() const { return 0.f; }

  // batch_normalization_forward_training(const tdesc_t& src_desc, const
  // tdesc_t& scale,
  //     const tdesc_t& shift, float momentum, float epsilon,
  //     unsigned flags = dnnl_normalization_flags_t::dnnl_use_scaleshift) {
  // }

  void running_statistic(const tensor& mean, const tensor& variance,
                         const tensor& running_mean,
                         const tensor& running_var) {}

  tdesc_t expected_statistic_descriptor() const { return tensor::desc(); }

  static void compute(const tensor& src, const tensor& scale,
                      const tensor& shift, tensor& dst, tensor& mean,
                      tensor& variance, float momentum, float epsilon) {}

  static void compute(const tensor& src, const tensor& scale,
                      const tensor& shift, tensor& dst, tensor& mean,
                      tensor& variance, tensor& running_mean,
                      tensor& running_var, float momentum, float epsilon) {}
};

struct batch_normalization_backward
    : public dnnl::batch_normalization_backward {
  float get_epsilon() const { return 0.f; }

  prop_kind get_prop_kind() const { return prop_kind::forward; }

  static void compute(const tensor& src, const tensor& mean,
                      const tensor& variance, const tensor& grady,
                      const tensor& scale, tensor& gradx, tensor& gradw,
                      float epsilon) {}

  static void compute(const tensor& src, const tensor& mean,
                      const tensor& variance, const tensor& grady,
                      const tensor& scale, tensor& gradx, tensor& grad_scale,
                      tensor& grad_shift, float epsilon) {}
};

}  // namespace ideep

#endif