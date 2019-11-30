#ifndef IDEEP_ATTRIBUTES_HPP
#define IDEEP_ATTRIBUTES_HPP

#include "abstract_types.hpp"

namespace ideep {

struct post_ops : public dnnl::post_ops {
 public:
  // bool has_op_kind(dnnl::primitive::kind op_kind) const {
  //   for (int i = 0; i < len(); i++)
  //     if (op_kind == kind(i))
  //       return true;
  //   return false;
  // }

  // bool non_negitive_output() const {
  //   // auto last = len() - 1;
  //   // if (last < 0) {
  //   //   return false;
  //   // }

  // auto params = get_params(last);
  // if (std::get<0>(params) != kind::eltwise || std::get<1>(params) <= 0.f ||
  //     std::get<2>(params) != 0.f || std::get<3>(params) != 0.f ||
  //     std::get<4>(params) != algorithm::eltwise_relu)
  //   return false;

  //   // return true;
  // }
  // Helper factory
  static post_ops sum(float scale = 1.0) {
    post_ops ret;
    ret.append(dnnl::primitive::kind::sum, scale, 1.0, 0.0, algorithm::eltwise_relu);
    return ret;
  }

  static post_ops relu(float scale = 1.f, float alpha = 0.f, float beta = 0.f) {
    post_ops ret;
    ret.append(dnnl::primitive::kind::eltwise, scale, alpha, beta, algorithm::eltwise_relu);
    return ret;
  }

  static post_ops residual(
      float sum_scale = 1.0,
      float relu_scale = 1.0,
      float alpha = 0.f,
      float beta = 0.f) {
    post_ops ret;
    ret.append(dnnl::primitive::kind::sum, sum_scale, 1.0, 0.0, algorithm::eltwise_relu);
    ret.append(dnnl::primitive::kind::eltwise, relu_scale, alpha, beta, algorithm::eltwise_relu);
    return ret;
  }

  void append(dnnl::primitive::kind op_kind, float scale, float alpha, float beta, algorithm alg) {
    switch(op_kind) {
      case dnnl::primitive::kind::sum:
        error::wrap_c_api(dnnl_post_ops_append_sum(get(), scale), "could not append sum");
        break;
      case dnnl::primitive::kind::eltwise:
        error::wrap_c_api(dnnl_post_ops_append_eltwise(
              get(), scale, convert_to_c(alg), alpha, beta), "could not append eltwise");
        break;
      default:
        error::wrap_c_api(dnnl_invalid_arguments, "Unsupport op kind");
    }
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

  inline bool has_op_kind(dnnl::primitive::kind op_kind) const {
    auto po = get_post_ops();
    for (int i = 0; i < po.len(); i++)
      if (op_kind == po.kind(i))
        return true;
    return false;
  }

  inline std::tuple<dnnl::primitive::kind, float, float, float, algorithm> get_params (
      int index) const {
    auto po = get_post_ops();
    IDEEP_ENFORCE(index < po.len(), "post_ops index is out of range");

    dnnl_alg_kind_t c_alg = dnnl_eltwise_relu;
    float scale = 1.0, alpha = 1.0, beta = 0.0;

    // auto akind = op_kind(index);
    auto akind = static_cast<dnnl::primitive::kind>(
        dnnl_post_ops_get_kind(po.get(), index));
    switch (akind) {
      case kind::sum:
        error::wrap_c_api(
            dnnl_post_ops_get_params_sum(po.get(), index, &scale),
            "could not get sum params");
        break;
      case kind::eltwise:
        error::wrap_c_api(
            dnnl_post_ops_get_params_eltwise(
                po.get(), index, &scale, &c_alg, &alpha, &beta),
            "could not get eltwise params");
        break;
      default:
        error::wrap_c_api(dnnl_invalid_arguments, "could not get params");
        break;
    }

    return std::make_tuple(
        akind, scale, alpha, beta, static_cast<algorithm>(c_alg));
  }

  inline bool non_negitive_output() const {
    auto po = get_post_ops();
    auto last = po.len() - 1;
    if (last < 0) {
      return false;
    }

    auto params = get_params(last);
    if (std::get<0>(params) != kind::eltwise || std::get<1>(params) <= 0.f ||
        std::get<2>(params) != 0.f || std::get<3>(params) != 0.f ||
        std::get<4>(params) != algorithm::eltwise_relu)
      return false;

    return true;
  }
};

}  // namespace ideep

#endif
