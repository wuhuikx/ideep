#ifndef IDEEP_KERNELS_INNER_PRODUCT_HPP
#define IDEEP_KERNELS_INNER_PRODUCT_HPP

#include "common.hpp"

namespace ideep {

struct inner_product_forward: public dnnl::inner_product_forward {
  template<bool with_bias=true>
  static inline void compute(const tensor& src, const tensor& weights, const tensor& bias, tensor& dst,
      const scale_t& src_scales = scale_t(), const scale_t& weights_scales = scale_t(), const scale_t& dst_scales = scale_t(),
      const attr_t& attr = attr_t(), prop_kind aprop_kind = prop_kind::forward, const lowp_kind alowp_kind = LOWP_U8S8) {
  }

  static void compute(const tensor& src, const tensor& weights, tensor& dst,
      const scale_t& src_scales = scale_t(), const scale_t& weights_scales = scale_t(), const scale_t& dst_scales = scale_t(),
      const attr_t& attr = attr_t(), prop_kind aprop_kind = prop_kind::forward, const lowp_kind alowp_kind = LOWP_U8S8) {
  }
};

// TODO: parameter sequence adjust?
struct inner_product_backward_data : public dnnl::inner_product_backward_data {
  static void compute(const tensor& grady, const tensor& weights, const tdims_t& gradx_dims, tensor& gradx) {
  }
};

struct inner_product_backward_weights : public dnnl::inner_product_backward_weights {
  template<bool with_gradb = true>
  static void compute(const tensor& x, const tensor& grady, tensor& gradw, tensor& gradb) {
  }

  static void compute(const tensor& x, const tensor& grady, tensor& gradw) {
  }
};

}  // namespace ideep

#endif