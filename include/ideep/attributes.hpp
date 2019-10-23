#ifndef IDEEP_ATTRIBUTES_HPP
#define IDEEP_ATTRIBUTES_HPP

#include "abstract_types.hpp"

namespace ideep {

struct post_ops : public dnnl::post_ops {

  bool has_op_kind(dnnl::primitive::kind op_kind) const {
    for (int i = 0; i < len(); i++)
      if (op_kind == kind(i))
        return true;
    return false;
  }

  bool non_negitive_output() const {
    // auto last = len() - 1;
    // if (last < 0) {
    //   return false;
    // }

    // auto params = get_params(last);
    // if (std::get<0>(params) != kind::eltwise || std::get<1>(params) <= 0.f ||
    //     std::get<2>(params) != 0.f || std::get<3>(params) != 0.f ||
    //     std::get<4>(params) != algorithm::eltwise_relu)
    //   return false;

    // return true;
  }
};

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