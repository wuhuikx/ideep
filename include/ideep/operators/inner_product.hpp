#ifndef IDEEP_OPERATORS_INNER_PRODUCT_HPP
#define IDEEP_OPERATORS_INNER_PRODUCT_HPP

namespace ideep {

struct inner_product_forward : public dnnl::inner_product_forward {

  using super = dnnl::inner_product_forward;

  static void compute(const tensor& src,
                      const tensor& weights,
                      const tensor& bias,
                      tensor& dst,
                      prop_kind aprop_kind = prop_kind::forward,
                      const engine& aengine = engine::cpu_engine()) {
    compute_impl</*with_bias=*/true>(
        src, weights, bias, dst, aprop_kind, aengine);
  }

  static void compute(const tensor& src,
                      const tensor& weights,
                      tensor& dst,
                      prop_kind aprop_kind = prop_kind::forward,
                      const engine& aengine = engine::cpu_engine()) {
    static tensor dummy_bias;
    compute_impl</*with_bias=*/false>(
        src, weights, dummy_bias, dst, aprop_kind, aengine);
  }


  static memory::desc expected_weights_desc(
      const dims& weights_dims,
      data_type dtype = data_type::f32,
      data_type x_dtype = data_type::f32,
      prop_kind aprop_kind = prop_kind::forward,
      const engine& aengine = engine::cpu_engine()) {
    auto x_dims = weights_dims;
    x_dims[0] = 1;
    auto y_dims = {x_dims[0], weights_dims[0]};
    auto ndims = weights_dims.size();
    auto y_dtype = (dtype != data_type::s8) ? dtype : data_type::s32;

    IDEEP_ENFORCE(x_dims.size() == weights_dims.size(),
                  "Invalid dims for data and weights");
    tensor::desc src_desc(x_dims, x_dtype);
    tensor::desc dst_desc(y_dims, y_dtype);
    tensor::desc weights_desc(weights_dims, dtype);
    auto pd =
        primitive_desc({aprop_kind, src_desc, weights_desc, dst_desc}, aengine);
    return pd.weights_desc();
  }

private:
  template <bool with_bias = true>
  static void compute_impl(const tensor& src,
                           const tensor& weights,
                           const tensor& bias,
                           tensor& dst,
                           prop_kind aprop_kind,
                           const engine& aengine) {
    // workaround: src and weights from caffe2 may have different dims.
    // It would be better for caffe2 to do this reshape anyway.
    auto src_ = src;
    if (src.ndims() != weights.ndims()) {
      auto new_dims = weights.get_dims();
      new_dims[0] = src.get_dim(0);
      src_.reshape(new_dims);
    }
    compute_impl_<with_bias>(src_, weights, bias, dst, aprop_kind, aengine);
  }

  template <bool with_bias = true>
  static void compute_impl_(const tensor& src,
                            const tensor& weights,
                            const tensor& bias,
                            tensor& dst,
                            prop_kind aprop_kind,
                            const engine& aengine) {
    auto src_desc = src.get_desc().to_format_any();
    auto weights_desc = weights.get_desc().to_format_any();
    auto bias_desc = bias.get_desc().to_format_any();
    auto dst_dims = {src.get_dim(0), weights.get_dim(0)};
    auto dst_desc = tensor::desc(dst_dims, src.get_data_type(), tag::any);

    auto pd = with_bias
        ? primitive_desc({aprop_kind, src_desc, weights_desc,
                          bias_desc, dst_desc}, aengine)
        : primitive_desc({aprop_kind, src_desc, weights_desc,
                          dst_desc}, aengine);

    auto expected_src = src.reorder_if_differ_in(pd.src_desc());
    auto expected_weights = weights.reorder_if_differ_in(pd.weights_desc());
    dst.reinit_if_necessary(pd.dst_desc());

    exec_args args {{DNNL_ARG_SRC, expected_src},
                    {DNNL_ARG_WEIGHTS, expected_weights},
                    {DNNL_ARG_DST, dst}};

    if (with_bias){
      auto expected_bias = bias.reorder_if_differ_in(pd.bias_desc());
      args.insert({DNNL_ARG_BIAS, expected_bias});
    }

    super(pd).execute(stream::default_stream(), args);
  }
};


struct inner_product_backward_data : public dnnl::inner_product_backward_data {

  using super = dnnl::inner_product_backward_data;

  static void compute(const tensor& diff_dst,
                      const tensor& weights,
                      const dims& diff_src_dims,
                      tensor& diff_src,
                      const engine& aengine = engine::cpu_engine()) {
    // workaround: diff_src and weights from caffe2 may have different dims.
    // It would be better for caffe2 to do this reshape anyway.
    auto weights_ = weights;
    if (diff_src_dims.size() != weights.ndims()) {
      auto new_dims = diff_src_dims;
      new_dims[0] = weights.get_dim(0);
      weights_.reshape(new_dims);
    }

    auto diff_dst_desc = diff_dst.get_desc().to_format_any();
    auto weights_desc = weights_.get_desc().to_format_any();
    auto diff_src_desc =
        tensor::desc(diff_src_dims, diff_dst.get_data_type(), tag::any);

    auto forward_hints = inner_product_forward::primitive_desc(
        {prop_kind::forward, diff_src_desc, weights_desc, diff_dst_desc},
        aengine);
    auto pd = primitive_desc({diff_src_desc, weights_desc, diff_dst_desc},
                             aengine, forward_hints);

    auto expected_diff_dst = diff_dst.reorder_if_differ_in(pd.diff_dst_desc());
    auto expected_weights = weights_.reorder_if_differ_in(pd.weights_desc());
    diff_src.reinit_if_necessary(pd.diff_src_desc());

    super(pd).execute(stream::default_stream(),
                      {{DNNL_ARG_DIFF_DST, expected_diff_dst},
                       {DNNL_ARG_WEIGHTS, expected_weights},
                       {DNNL_ARG_DIFF_SRC, diff_src}});
  }
};

struct inner_product_backward_weights
    : public dnnl::inner_product_backward_weights {

  using super = dnnl::inner_product_backward_weights;

  static void compute(const tensor& src,
                      const tensor& diff_dst,
                      tensor& diff_weights,
                      tensor& diff_bias,
                      const engine& aengine = engine::cpu_engine()) {
    compute_impl</*with_diff_bias=*/true>(
        src, diff_dst, diff_weights, diff_bias);
  }

  static void compute(const tensor& src,
                      const tensor& diff_dst,
                      tensor& diff_weights,
                      const engine& aengine = engine::cpu_engine()) {
    static tensor dummy_diff_bias;
    compute_impl</*with_diff_bias=*/false>(
        src, diff_dst, diff_weights, dummy_diff_bias);
  }

private:
  template<bool with_diff_bias = true>
  static void compute_impl(const tensor& src,
                           const tensor& diff_dst,
                           tensor& diff_weights,
                           tensor& diff_bias,
                           const engine& aengine = engine::cpu_engine()) {
    auto src_desc = src.get_desc().to_format_any();
    auto diff_dst_desc = diff_dst.get_desc().to_format_any();
    auto diff_weights_dims = src.get_dims();
    diff_weights_dims[0] = diff_dst.get_dim(1);
    auto diff_weights_desc =
        tensor::desc(diff_weights_dims, diff_dst.get_data_type(), tag::any);
    auto diff_bias_dims = {diff_dst.get_dim(1)};
    auto diff_bias_desc =
        tensor::desc(diff_bias_dims, diff_dst.get_data_type(), tag::any);

    auto forward_hints = with_diff_bias
        ? inner_product_forward::primitive_desc({prop_kind::forward, src_desc,
            diff_weights_desc, diff_bias_desc, diff_dst_desc}, aengine)
        : inner_product_forward::primitive_desc({prop_kind::forward, src_desc,
            diff_weights_desc, diff_dst_desc}, aengine);
    auto pd = with_diff_bias
        ? primitive_desc({src_desc, diff_weights_desc, diff_bias_desc,
                          diff_dst_desc}, aengine, forward_hints)
        : primitive_desc({src_desc, diff_weights_desc, diff_dst_desc},
                          aengine, forward_hints);

    auto expected_diff_dst = diff_dst.reorder_if_differ_in(pd.diff_dst_desc());
    auto expected_src = src.reorder_if_differ_in(pd.src_desc());
    diff_weights.reinit_if_necessary(pd.diff_weights_desc());

    exec_args args {{DNNL_ARG_DIFF_DST, expected_diff_dst},
                    {DNNL_ARG_SRC, expected_src},
                    {DNNL_ARG_DIFF_WEIGHTS ,diff_weights}};

    if (with_diff_bias) {
      diff_bias.reinit_if_necessary(pd.diff_bias_desc());
      args.insert({DNNL_ARG_DIFF_BIAS, diff_bias});
    }

    super(pd).execute(stream::default_stream(), args);

    // XPZ: TODO: ???
    diff_weights = std::move(diff_weights.to_public());
  }
};

}  // namespace ideep

#endif
