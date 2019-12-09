#ifndef IDEEP_OPERATORS_INNER_PRODUCT_HPP
#define IDEEP_OPERATORS_INNER_PRODUCT_HPP

namespace ideep {

struct inner_product_forward : public dnnl::inner_product_forward {

  using super = dnnl::inner_product_forward;

  static void compute(const tensor& src,
                      const tensor& weights,
                      const tensor& bias,
                      tensor& dst,
                      const scale_t& src_scales = scale_t(),
                      const scale_t& weights_scales = scale_t(),
                      const scale_t& dst_scales = scale_t(),
                      const attr_t& attr = attr_t(),
                      const prop_kind aprop_kind = prop_kind::forward,
                      const lowp_kind alowp_kind = LOWP_U8S8,
                      const engine& aengine = engine::cpu_engine()) {
    compute_impl</*with_bias=*/true>(
        src,
        weights,
        bias,
        dst,
        src_scales,
        weights_scales,
        dst_scales,
        attr,
        aprop_kind,
        alowp_kind,
        aengine);
  }

  static void compute(const tensor& src,
                      const tensor& weights,
                      tensor& dst,
                      const scale_t& src_scales = scale_t(),
                      const scale_t& weights_scales = scale_t(),
                      const scale_t& dst_scales = scale_t(),
                      const attr_t& attr = attr_t(),
                      const prop_kind aprop_kind = prop_kind::forward,
                      const lowp_kind alowp_kind = LOWP_U8S8,
                      const engine& aengine = engine::cpu_engine()) {
    static tensor dummy_bias;
    compute_impl</*with_bias=*/false>(
        src,
        weights,
        dummy_bias,
        dst,
        src_scales,
        weights_scales,
        dst_scales,
        attr,
        aprop_kind,
        alowp_kind,
        aengine);
  }


  static tensor::desc expected_weights_desc(
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
  template <bool with_bias>
  static void compute_impl(const tensor& src,
                           const tensor& weights,
                           const tensor& bias,
                           tensor& dst,
                           const scale_t& src_scales,
                           const scale_t& weights_scales,
                           const scale_t& dst_scales,
                           const attr_t& attr,
                           const prop_kind aprop_kind,
                           const lowp_kind alowp_kind,
                           const engine& aengine) {
    // workaround: src and weights from caffe2 may have different dims.
    // It would be better for caffe2 to do this reshape anyway.
    auto src_ = src;
    if (src.ndims() != weights.ndims()) {
      auto new_dims = weights.get_dims();
      new_dims[0] = src.get_dim(0);
      src_.reshape(new_dims);
    }
    compute_impl_<with_bias>(
        src_,
        weights,
        bias,
        dst,
        src_scales,
        weights_scales,
        dst_scales,
        attr,
        aprop_kind,
        alowp_kind,
        aengine);
  }

  template <bool with_bias>
  static void compute_impl_(const tensor& src,
                            const tensor& weights,
                            const tensor& bias,
                            tensor& dst,
                            const scale_t& src_scales,
                            const scale_t& weights_scales,
                            const scale_t& dst_scales,
                            const attr_t& attr,
                            const prop_kind aprop_kind,
                            const lowp_kind alowp_kind,
                            const engine& aengine) {
    tensor::desc src_desc, weights_desc, bias_desc;
    attr_t op_attr, src_attr, weights_attr, bias_attr;
    scale_t dst_scales_in;
    auto dst_data_type = data_type::f32;
    tensor::dims dst_dims;

    auto weights_scales_in =
        weights.has_scale() ? weights.get_scale() : weights_scales;

    if (!weights_scales_in.empty()) {
      IDEEP_ENFORCE(
          alowp_kind == LOWP_U8S8 || alowp_kind == LOWP_S8S8,
          "Unsupported lowp kind");

      auto src_scales_in = src.has_scale()
          ? src.get_scale()
          : (src_scales.empty() ? IDEEP_DEF_SCALE : src_scales);

      src_desc = {src.get_dims(),
                  alowp_kind == LOWP_U8S8 ? data_type::u8 : data_type::s8,
                  format_tag::any};
      if (src.get_data_type() == data_type::f32) {
        src_attr = {0, src_scales_in};
      }

      dst_dims = {src_desc.get_dim(0), weights.get_dim(0)};
      int scale_size = (weights_scales_in.size() > 1) ? dst_dims[1] : 1;

      weights_desc = {weights.get_dims(), data_type::s8, format_tag::any};
      if (weights.get_data_type() == data_type::f32) {
        weights_attr = {utils::tensor_scale_mask(scale_size, false),
                        weights_scales_in};
      }

      // determine dst data type
      if (dst_scales.empty() || dst_scales == IDEEP_DEF_SCALE) {
        dst_data_type = data_type::f32;
      } else if (attr.non_negitive_output()) {
        dst_data_type = data_type::u8;
      } else {
        dst_data_type = data_type::s8;
      }

      // fill primitive attr
      scale_t op_scales(scale_size), bias_scales(scale_size);
      dst_scales_in = (dst_scales.empty() || dst_data_type == data_type::f32)
          ? IDEEP_DEF_SCALE
          : dst_scales;
      for (int i = 0; i < scale_size; i++) {
        bias_scales[i] = src_scales_in[0] * weights_scales_in[i];
        op_scales[i] = dst_scales_in[0] / bias_scales[i];
      }
      op_attr.set_output_scales(utils::op_scale_mask(scale_size), op_scales);

      if (with_bias) {
        bias_desc = {bias.get_dims(), data_type::s32, format_tag::any};
        if (bias.get_data_type() == data_type::f32) {
          bias_attr = {utils::tensor_scale_mask(scale_size, false), bias_scales};
        }
      }
    } else {
      op_attr = attr;
      src_desc = {src.get_dims(), data_type::f32, format_tag::any};
      if (src.has_scale()) {
        auto src_scale = src.get_scale();
        src_scale[0] = 1.0f / src_scale[0];
        src_attr = {0, src_scale};
      }
      dst_dims = {src_desc.get_dim(0), weights.get_dim(0)};
      weights_desc = weights.get_desc().to_format_any();
      IDEEP_ENFORCE(
          weights.get_data_type() == data_type::f32,
          "Incorrect data type in weights");
      if (with_bias) {
        IDEEP_ENFORCE(
            bias.get_data_type() == data_type::f32,
            "Incorrect data type in bias");
        bias_desc = bias.get_desc().to_format_any();
      }
    }
    tensor::desc dst_desc(dst_dims, dst_data_type, format_tag::any);
    auto pd = with_bias
       ? primitive_desc(
             {aprop_kind, src_desc, weights_desc, bias_desc, dst_desc}, op_attr, aengine)
       : primitive_desc(
             {aprop_kind, src_desc, weights_desc, dst_desc}, op_attr, aengine);

    auto expected_src = src.reorder_if_differ_in(pd.src_desc(), src_attr);
    auto expected_weights = weights.reorder_if_differ_in(pd.weights_desc(), weights_attr);
    dst.reinit_if_possible(pd.dst_desc());
    if (!dst_scales.empty() && dst_data_type != data_type::f32) {
      dst.set_scale(dst_scales_in);
    }

    if (with_bias){
      auto expected_bias = bias.reorder_if_differ_in(pd.bias_desc(), bias_attr);
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

    if (attr.non_negitive_output() && dst.get_data_type() == data_type::s8) {
      tensor::desc dst_u8_desc = dst.get_desc().to_type(data_type::u8);
      dst.set_desc(dst_u8_desc);
    }
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

    auto forward_hints =
        inner_product_forward::primitive_desc(
            {prop_kind::forward, diff_src_desc, weights_desc, diff_dst_desc},
            aengine);

    auto pd = primitive_desc(
        {diff_src_desc, weights_desc, diff_dst_desc}, aengine, forward_hints);

    auto expected_diff_dst = diff_dst.reorder_if_differ_in(pd.diff_dst_desc());
    auto expected_weights = weights_.reorder_if_differ_in(pd.weights_desc());
    diff_src.reinit_if_possible(pd.diff_src_desc());

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
    diff_weights.reinit_if_possible(pd.diff_weights_desc());

    exec_args args {{DNNL_ARG_DIFF_DST, expected_diff_dst},
                    {DNNL_ARG_SRC, expected_src},
                    {DNNL_ARG_DIFF_WEIGHTS ,diff_weights}};

    if (with_diff_bias) {
      diff_bias.reinit_if_possible(pd.diff_bias_desc());
      args.insert({DNNL_ARG_DIFF_BIAS, diff_bias});
    }

    super(pd).execute(stream::default_stream(), args);
  }
};

}  // namespace ideep

#endif
