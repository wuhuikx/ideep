#ifndef IDEEP_KERNELS_INNER_PRODUCT_HPP
#define IDEEP_KERNELS_INNER_PRODUCT_HPP

#include "common.hpp"

namespace ideep {

struct inner_product_forward : public dnnl::inner_product_forward {

  typedef dnnl::inner_product_forward super;

  static void compute(const tensor& src,
                      const tensor& weights,
                      const tensor& bias,
                      tensor& dst,
                      prop_kind aprop_kind = prop_kind::forward,
                      const engine& aengine = engine::cpu_engine()) {
    auto src_desc = src.get_desc().to_format_any();
    auto weights_desc = weights.get_desc().to_format_any();
    auto bias_desc = bias.get_desc().to_format_any();
    auto dst_desc = dst.get_desc().to_format_any();

    auto pd = primitive_desc(
        {aprop_kind, src_desc, weights_desc, bias_desc, dst_desc}, aengine);

    auto expected_src = src.reorder_if_necessary(pd.src_desc());
    auto expected_weights = weights.reorder_if_necessary(pd.weights_desc());
    dst.reinit_if_necessary(pd.dst_desc());

    super(pd).execute(stream::default_stream(),
                      {{DNNL_ARG_SRC, expected_src},
                       {DNNL_ARG_WEIGHTS, expected_weights},
                       {DNNL_ARG_BIAS, bias},
                       {DNNL_ARG_DST, dst}});
  }

  static void compute(const tensor& src,
                      const tensor& weights,
                      tensor& dst,
                      prop_kind aprop_kind = prop_kind::forward,
                      const engine& aengine = engine::cpu_engine()) {
    auto src_desc = src.get_desc().to_format_any();
    auto weights_desc = weights.get_desc().to_format_any();
    auto dst_desc = dst.get_desc().to_format_any();

    auto pd = primitive_desc(
        {aprop_kind, src_desc, weights_desc, dst_desc}, aengine);

    auto expected_src = src.reorder_if_necessary(pd.src_desc());
    auto expected_weights = weights.reorder_if_necessary(pd.weights_desc());
    dst.reinit_if_necessary(pd.dst_desc());

    super(pd).execute(stream::default_stream(),
                      {{DNNL_ARG_SRC, expected_src},
                       {DNNL_ARG_WEIGHTS, expected_weights},
                       {DNNL_ARG_DST, dst}});
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