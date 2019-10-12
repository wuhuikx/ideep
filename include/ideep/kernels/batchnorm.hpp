#ifndef IDEEP_KERNELS_BATCHNORM_HPP
#define IDEEP_KERNELS_BATCHNORM_HPP

#include "common.hpp"

namespace ideep {

struct batch_normalization_forward_inference
    : public dnnl::batch_normalization_forward {
  static void compute(const tensor& src,
                      const tensor& scale,
                      const tensor& shift,
                      tensor& dst,
                      float epsilon,
                      const engine& aengine = engine::cpu_engine()) {
    // XPZ: this overload has not been tested
    auto src_desc = src.get_desc();

    auto flags = batch_normalization_flag::use_scale_shift;
    // XPZ: TODO: attr?
    auto pd = dnnl::batch_normalization_forward::primitive_desc(
        {prop_kind::forward_inference, src_desc, epsilon, flags},
        aengine);

    tensor scale_shift {pd.weights_desc()};
    std::memcpy(scale_shift.get_data_handle(),
                scale.get_data_handle(), scale.get_size());
    std::memcpy(scale_shift.get_data_handle() + scale.get_size(),
                shift.get_data_handle(), shift.get_size());

    dnnl::batch_normalization_forward(pd).execute(stream::default_stream(),
                                          {{DNNL_ARG_SRC, src},
                                           {DNNL_ARG_SCALE_SHIFT, scale_shift},
                                           {DNNL_ARG_DST, dst}});
  }

  static void compute(const tensor& src,
                      const tensor& mean,
                      const tensor& variance,
                      const tensor& scale,
                      const tensor& shift,
                      tensor& dst,
                      float epsilon,
                      const engine& aengine = engine::cpu_engine()) {
    auto src_desc = src.get_desc();
    dst.reinit_if_necessary(src_desc);

    auto flags = batch_normalization_flag::use_scale_shift |
                 batch_normalization_flag::use_global_stats;
    // XPZ: TODO: attr?
    auto pd = dnnl::batch_normalization_forward::primitive_desc(
        {prop_kind::forward_inference, src_desc, epsilon, flags},
        aengine);

    tensor scale_shift {pd.weights_desc()};
    std::memcpy(scale_shift.get_data_handle(),
                scale.get_data_handle(), scale.get_size());
    std::memcpy(scale_shift.get_data_handle() + scale.get_size(),
                shift.get_data_handle(), shift.get_size());

    dnnl::batch_normalization_forward(pd).execute(stream::default_stream(),
                                          {{DNNL_ARG_SRC, src},
                                           {DNNL_ARG_SCALE_SHIFT, scale_shift},
                                           {DNNL_ARG_VARIANCE, variance},
                                           {DNNL_ARG_MEAN, mean},
                                           {DNNL_ARG_DST, dst}});
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