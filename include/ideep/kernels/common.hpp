#ifndef IDEEP_KERNELS_COMMON_HPP
#define IDEEP_KERNELS_COMMON_HPP

#include "../abstract_types.hpp"
#include "../tensor.hpp"
#include "../utils.hpp"

namespace ideep {

using tdims_t = tensor::dims;
using tdim_t = tensor::dim;
using post_ops = dnnl::post_ops;

/// Attribute class for extra information into computations
struct attr_t : public dnnl::primitive_attr {
public:
  attr_t() {}

  attr_t(int mask, scale_t& scales) {
    set_output_scales(mask, scales);
  }

  std::pair<scale_t, int> get_output_scales() const {
    dnnl_dim_t count;
    int c_mask;
    const float* c_scales;
    error::wrap_c_api(dnnl_primitive_attr_get_output_scales(get(), &count,
                                                            &c_mask, &c_scales),
                      "could not get int output scales");
    return std::make_pair(scale_t(c_scales, c_scales + count), c_mask);
  }

  // Helper factory
  static inline attr_t fuse_sum(float scale = 1.0) {
    attr_t attr;
    post_ops po;
    po.append_sum(scale);
    attr.set_post_ops(po);
    return attr;
  }

  static inline attr_t fuse_relu(float scale = 1.0, float alpha = 0.f,
                                 float beta = 0.f) {
    attr_t attr;
    post_ops po;
    po.append_eltwise(scale, algorithm::eltwise_relu, alpha, beta);
    attr.set_post_ops(po);
    return attr;
  }


  static inline attr_t residual(float sum_scale = 1.0, float relu_scale = 1.0,
                                float alpha = 0.f, float beta = 0.f) {
    attr_t attr;
    post_ops po;
    po.append_sum(sum_scale);
    po.append_eltwise(relu_scale, algorithm::eltwise_relu, alpha, beta);
    attr.set_post_ops(po);
    return attr;
  }

  static inline attr_t attr_post_ops(post_ops po) {
    attr_t attr;
    attr.set_post_ops(po);
    return attr;
  }
};

}  // namespace ideep

#endif