#ifndef IDEEP_OPERATORS_INNER_PRODUCT_HPP
#define IDEEP_OPERATORS_INNER_PRODUCT_HPP

namespace ideep {

struct inner_product_forward : public dnnl::inner_product_forward {

  using super = dnnl::inner_product_forward;
  template <bool with_bias = true>
  static void compute(const tensor& src,
                      const tensor& weights,
                      const tensor& bias,
                      tensor& dst,
                      prop_kind aprop_kind = prop_kind::forward,
                      const engine& aengine = engine::cpu_engine()) {
  compute_impl<with_bias>(src, weights, bias, dst, aprop_kind, aengine);
  }

  static void compute(const tensor& src,
                      const tensor& weights,
                      tensor& dst,
                      prop_kind aprop_kind = prop_kind::forward,
                      const engine& aengine = engine::cpu_engine()) {
    static tensor dummy_bias;
    compute<false>(src, weights, dummy_bias, dst, aprop_kind, aengine);
  }


  static memory::desc expected_weights_desc(const dims& weights_dims,
                                            data_type dtype = data_type::f32,
                                            data_type x_dtype = data_type::f32) {
    // auto x_dims = weights_dims;
    // x_dims[0] = 1;
    // auto y_dims = {x_dims[0], weights_dims[0]};
    // auto ndims = weights_dims.size();
    // auto y_dtype = (dtype != tdtype_t::s8) ? dtype : tdtype_t::s32;

    // IDEEP_ENFORCE(x_dims.size() == weights_dims.size(), "Invalid dims for data and weights");
    // tdesc_t x_desc(x_dims, x_dtype, ndims == 2 ? format::nc : format::nchw);
    // tdesc_t y_desc(y_dims, y_dtype, format::nc);
    // tdesc_t weights_desc(weights_dims, dtype, ndims == 2 ? format::oi : format::oihw);

    // inner_product_forward comp(x_desc, weights_desc, tdesc_t(), y_desc);
    // return comp.dup_descriptor_of(query::weights_pd);
    return tensor::desc();
  }

private:
  template <bool with_bias = true>
  static void compute_impl(const tensor& src,
                           const tensor& weights,
                           const tensor& bias,
                           tensor& dst,
                           prop_kind aprop_kind = prop_kind::forward,
                           const engine& aengine = engine::cpu_engine()) {
    auto src_desc = src.get_desc().to_format_any();
    auto weights_desc = weights.get_desc().to_format_any();
    auto bias_desc = with_bias ? bias.get_desc().to_format_any() : tensor::desc();
    auto dst_desc = dst.get_desc().to_format_any();
    auto pd = with_bias ? primitive_desc(
        {aprop_kind, src_desc, weights_desc, bias_desc, dst_desc}, aengine)
        : primitive_desc({aprop_kind, src_desc, weights_desc, dst_desc}, aengine);
    auto expected_src = src.reorder_if_differ_in(pd.src_desc());
    auto expected_weights = weights.reorder_if_differ_in(pd.weights_desc());
    dst.reinit_if_necessary(pd.dst_desc());
    if (with_bias){
      auto expected_bias = bias.reorder_if_differ_in(pd.bias_desc());
      super(pd).execute(stream::default_stream(),
                        {{DNNL_ARG_SRC, expected_src},
                         {DNNL_ARG_WEIGHTS, expected_weights},
                         {DNNL_ARG_BIAS, expected_bias},
                         {DNNL_ARG_DST, dst}});
    } else {
      super(pd).execute(stream::default_stream(),
                        {{DNNL_ARG_SRC, expected_src},
                         {DNNL_ARG_WEIGHTS, expected_weights},
                         {DNNL_ARG_DST, dst}});
    }
  }
};

// TODO: parameter sequence adjust?
struct inner_product_backward_data : public dnnl::inner_product_backward_data {
  using super = dnnl::inner_product_backward_data;
  static void compute(const tensor& diff_dst,
                      const tensor& weights,
                      const dims& diff_src_dims,
                      tensor& diff_src,
                      const engine& aengine = engine::cpu_engine()) {
    tensor::desc diff_src_desc = tensor::desc(diff_src_dims, diff_dst.get_data_type());
    auto weights_desc = weights.get_desc();
    auto diff_dst_desc = diff_dst.get_desc();
    auto forward_hints = inner_product_forward::primitive_desc(
        {prop_kind::forward, diff_src_desc, weights_desc, diff_dst_desc}, aengine);
    auto pd =primitive_desc(
        {diff_src_desc, weights_desc, diff_dst_desc}, aengine, forward_hints);
    auto expected_diff_dst = diff_dst.reorder_if_differ_in(pd.diff_dst_desc());
    auto expected_weights = weights.reorder_if_differ_in(pd.weights_desc());
    diff_src.reinit_if_necessary(pd.diff_src_desc());
    super(pd).execute(stream::default_stream(),
                      {{DNNL_ARG_DIFF_DST, expected_diff_dst},
                       {DNNL_ARG_WEIGHTS, expected_weights},
                       {DNNL_ARG_DIFF_SRC, diff_src}});
  }
};

struct inner_product_backward_weights : public dnnl::inner_product_backward_weights {
  using super = dnnl::inner_product_backward_weights;
  template<bool with_diff_bias = true>
  static void compute(const tensor& src,
                      const tensor& diff_dst,
                      tensor& diff_weights,
                      tensor& diff_bias,
                      const engine& aengine = engine::cpu_engine()) {
    compute_impl<with_diff_bias>(src, diff_dst, diff_weights, diff_bias);
  }

  static void compute(const tensor& src,
                      const tensor& diff_dst,
                      tensor& diff_weights,
                      const engine& aengine = engine::cpu_engine()) {
   static tensor dummy_diff_bias;
   compute<false>(src, diff_dst, diff_weights, dummy_diff_bias);
  }

private:
  template<bool with_diff_bias = true>
    static void compute_impl(const tensor& src,
                             const tensor& diff_dst,
                             tensor& diff_weights,
                             tensor& diff_bias,
                             const engine& aengine = engine::cpu_engine()) {
      auto src_desc = src.get_desc();
      auto diff_dst_desc = diff_dst.get_desc();
      auto diff_weights_dims = src.get_dims();
      diff_weights_dims[0] = diff_dst.get_dim(1);
      tensor::desc diff_weights_desc = tensor::desc(diff_weights_dims, diff_dst.get_data_type());
      tensor::desc diff_bias_desc = with_diff_bias
          ? tensor::desc({diff_dst.get_dim(1)}, diff_dst.get_data_type())
          : tensor::desc();
      auto forward_hints = with_diff_bias
          ? inner_product_forward::primitive_desc({prop_kind::forward, src_desc,
              diff_weights_desc, diff_bias_desc, diff_dst_desc}, aengine)
          : inner_product_forward::primitive_desc({prop_kind::forward, src_desc,
              diff_weights_desc, diff_dst_desc}, aengine);
      auto pd = with_diff_bias
          ? primitive_desc({src_desc, diff_weights_desc, diff_bias_desc, diff_dst_desc}, aengine, forward_hints)
          : primitive_desc({src_desc, diff_weights_desc, diff_dst_desc}, aengine, forward_hints);
      auto expected_diff_dst = diff_dst.reorder_if_differ_in(pd.diff_dst_desc());
      auto expected_src = src.reorder_if_differ_in(pd.src_desc());
      diff_weights.reinit_if_necessary(pd.diff_weights_desc());
      if (with_diff_bias) {
        diff_bias.reinit_if_necessary(pd.diff_bias_desc());
        super(pd).execute(stream::default_stream(),
                          {{DNNL_ARG_DIFF_DST, expected_diff_dst},
                           {DNNL_ARG_SRC, expected_src},
                           {DNNL_ARG_DIFF_WEIGHTS ,diff_weights},
                           {DNNL_ARG_DIFF_BIAS, diff_bias}});
      } else {
        super(pd).execute(stream::default_stream(),
                          {{DNNL_ARG_DIFF_DST, expected_diff_dst},
                           {DNNL_ARG_SRC, expected_src},
                           {DNNL_ARG_DIFF_WEIGHTS ,diff_weights},
                           {DNNL_ARG_DIFF_BIAS, diff_bias}});
      }
    }
};

}  // namespace ideep

#endif
