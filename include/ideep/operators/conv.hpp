#ifndef IDEEP_OPERATORS_CONV_HPP
#define IDEEP_OPERATORS_CONV_HPP

namespace ideep {

struct convolution_forward : public dnnl::convolution_forward {

  using super = dnnl::convolution_forward;

  // fp32 w/ bias
  static void compute(
      const tensor& src,
      const tensor& weights,
      const tensor& bias,
      const dims& dst_dims,
      tensor& dst,
      const dims& strides,
      const dims& dilates,
      const dims& padding_l,
      const dims& padding_r,
      int groups,
      const scale_t& src_scales = scale_t(),
      const scale_t& weights_scales = scale_t(),
      const scale_t& dst_scales = scale_t(),
      const attr_t& attr = attr_t(),
      algorithm aalgorithm = algorithm::convolution_direct,
      prop_kind aprop_kind = prop_kind::forward,
      const lowp_kind alowp_kind = LOWP_U8S8,
      const engine& aengine = engine::cpu_engine()) {
    compute_impl</*with_bias=*/true>(
        src,
        weights,
        bias,
        dst_dims,
        dst,
        strides,
        dilates,
        padding_l,
        padding_r,
        groups,
        src_scales,
        weights_scales,
        dst_scales,
        attr,
        aalgorithm,
        aprop_kind,
        alowp_kind,
        aengine);
  }

  // fp32 w/o bias
  static void compute(
      const tensor& src,
      const tensor& weights,
      const dims& dst_dims,
      tensor& dst,
      const dims& strides,
      const dims& dilates,
      const dims& padding_l,
      const dims& padding_r,
      int groups,
      const scale_t& src_scales = scale_t(),
      const scale_t& weights_scales = scale_t(),
      const scale_t& dst_scales = scale_t(),
      const attr_t& attr = attr_t(),
      algorithm aalgorithm = algorithm::convolution_direct,
      prop_kind aprop_kind = prop_kind::forward,
      const lowp_kind alowp_kind = LOWP_U8S8,
      const engine& aengine = engine::cpu_engine()) {
    static tensor dummy_bias;
    compute_impl</*with_bias=*/false>(
        src,
        weights,
        dummy_bias,
        dst_dims,
        dst,
        strides,
        dilates,
        padding_l,
        padding_r,
        groups,
        src_scales,
        weights_scales,
        dst_scales,
        attr,
        aalgorithm,
        aprop_kind,
        alowp_kind,
        aengine);
  }

  // TODO: XPZ: refactor it
  static memory::desc expected_weights_desc(
      const dims& weights_dims,
      data_type dtype = data_type::f32,
      const dims& strides = {1, 1},
      const dims& padding_l = {0, 0},
      const dims& padding_r = {0, 0},
      const dims& dilates = {0, 0},
      int groups = 1,
      algorithm aalgorithm = algorithm::convolution_direct,
      prop_kind aprop_kind = prop_kind::forward,
      data_type x_dtype = data_type::f32,
      const dims& src_dims = dims(),
      const attr_t& attr = attr_t(),
      const engine& aengine = engine::cpu_engine()) {

    auto grouped = groups > 1;
    auto weights_desc_usr = tensor::desc(weights_dims, dtype);
    auto weights_desc =
        grouped ? weights_desc_usr.to_grouped(groups) : weights_desc_usr;

    auto dims_in = weights_desc.get_dims();
    auto ndims = dims_in.size();
    auto g = grouped ? dims_in[0] : 1;
    auto dilates_ = utils::get_compatible_dilates(dilates);

    IDEEP_ENFORCE(
        !(aalgorithm == algorithm::convolution_winograd && src_dims.empty()),
        "Incorrect src_dims");
    auto ic = g * dims_in[1 + grouped];
    auto oc = g * dims_in[0 + grouped];
    auto kh = dims_in[ndims - 2];
    auto kw = dims_in[ndims - 1];
    int mb, h, w;
    if (src_dims.empty()) {
      // Construct a dummy case
      mb = 1;
      h = 2 * kh;
      w = 4 * kw;
    } else {
      // Use the real data
      mb = src_dims[0];
      h = src_dims[2];
      w = src_dims[3];
    }
    auto oh = (h - ((kh - 1) * (dilates_[0] + 1) + 1) + (padding_l[0] + padding_r[0])) / strides[0] + 1;
    auto ow = (w - ((kw - 1) * (dilates_[1] + 1) + 1) + (padding_l[1] + padding_r[1])) / strides[1] + 1;

    dims x_dims = { mb, ic, h, w };
    dims y_dims = { mb, oc, oh, ow };
    auto y_dtype =
        dtype != data_type::s8 ? dtype : data_type::s32;
    tensor::desc src_desc(x_dims, x_dtype);
    tensor::desc dst_desc(y_dims, y_dtype);

    // FIXME: workaroud winograd format issue in inference
    // If prop_kind == forward_inference, the dnnl_wino_fmt for weights is required by winograd primitive.
    // Then, in the cases of variable input shape, the detials of dnnl_wino_fmt will be changed.
    // And, extra weihgts reorder is inevitable each time, leading to bad performance.
    // Here, we set the prop_kind to forward, in order to reorder and cache weights as blocked format,
    // instead of dnnl_wino_fmt.
    auto apkind = aprop_kind;
    if (aalgorithm == algorithm::convolution_winograd &&
        aprop_kind == prop_kind::forward_inference) {
      apkind = prop_kind::forward;
    }

    auto pd = get_primitive_desc</*with_bias=*/false>(
        src_desc, weights_desc, tensor::desc(), dst_desc, strides, dilates_,
        padding_l, padding_r, attr_t(), aalgorithm, apkind);

    // if (!grouped) {
      return pd.weights_desc();
    // } else {
    //   // hide group info from the outside world
    //   return tensor::desc(pd.weights_desc()).to_ungrouped();
    // }
  }

  template <bool with_bias>
  static primitive_desc get_primitive_desc(
      const tensor::desc& src_desc,
      const tensor::desc& weights_desc,
      const tensor::desc& bias_desc,
      const tensor::desc& dst_desc,
      const dims& strides,
      const dims& dilates,
      const dims& padding_l,
      const dims& padding_r,
      const attr_t& attr = attr_t(),
      algorithm aalgorithm = algorithm::convolution_direct,
      prop_kind aprop_kind = prop_kind::forward,
      const engine& aengine = engine::cpu_engine()) {
    auto src_desc_any = src_desc.to_format_any();
    auto weights_desc_any = weights_desc.to_format_any();
    auto bias_desc_any = with_bias ? bias_desc.to_format_any() : tensor::desc();
    auto dst_desc_any = dst_desc.to_format_any();

    if (with_bias) {
      return primitive_desc({aprop_kind, aalgorithm, src_desc_any,
                             weights_desc_any, bias_desc_any, dst_desc_any,
                             strides, dilates, padding_l, padding_r},
                            attr, aengine);
    } else {
      return primitive_desc({aprop_kind, aalgorithm, src_desc_any,
                             weights_desc_any, dst_desc_any,
                             strides, dilates, padding_l, padding_r},
                            attr, aengine);
    }
  }

private:
  template <bool with_bias>
  static void compute_impl(const tensor& src,
                           const tensor& weights,
                           const tensor& bias,
                           const dims& dst_dims,
                           tensor& dst,
                           const dims& strides,
                           const dims& dilates,
                           const dims& padding_l,
                           const dims& padding_r,
                           int groups,
                           const scale_t& src_scales,
                           const scale_t& weights_scales,
                           const scale_t& dst_scales,
                           const attr_t& attr,
                           algorithm aalgorithm,
                           prop_kind aprop_kind,
                           const lowp_kind alowp_kind,
                           const engine& aengine) {
    scale_t dst_scales_in;
    auto dst_data_type = data_type::f32;
    tensor::desc src_desc, weights_desc, bias_desc;
    attr_t op_attr, src_attr, weights_attr, bias_attr;

    // make weights and dilates compatible with DNNL
    auto weights_ = weights.make_grouped_weights(groups);
    auto dilates_ = utils::get_compatible_dilates(dilates);

    auto weights_scales_in = weights_.has_scale() ? weights_.get_scale() : weights_scales;
    if (!weights_scales_in.empty()) {
      IDEEP_ENFORCE(alowp_kind == LOWP_U8S8 || alowp_kind == LOWP_S8S8, "Unsupported lowp kind");
      int scale_size = (weights_scales_in.size() > 1) ? dst_dims[1] : 1;
      auto src_scales_in = src.has_scale() ? src.get_scale()
        : (src_scales.empty() ? IDEEP_DEF_SCALE : src_scales);

      // determine dst data type
      if (attr.has_op_kind(kind::sum)) {
        dst_data_type = dst.get_data_type();
      } else if (dst_scales.empty() || dst_scales == IDEEP_DEF_SCALE) {
        dst_data_type = data_type::f32;
      } else if (attr.non_negitive_output()){
        dst_data_type = data_type::u8;
      } else {
        dst_data_type = data_type::s8;
      }

      // fill primitive attr
      scale_t op_scales(scale_size), bias_scales(scale_size);
      dst_scales_in = (dst_scales.empty() || dst_data_type == data_type::f32)
        ? IDEEP_DEF_SCALE : dst_scales;
      for (int i = 0; i < scale_size; i++) {
        bias_scales[i] = src_scales_in[0] * weights_scales_in[i];
        op_scales[i] = dst_scales_in[0] / bias_scales[i];
        // op_scales[i] = 1.0f;
      }

      // if (attr.has_op_kind(kind::sum)) {
      //   float sum_scale = dst_scales_in[0] / (dst.has_scale() ? dst.get_scale()[0] : 1.0f);
      //   if (attr.has_op_kind(kind::eltwise)) {
      //     op_attr = attr_t::residual(sum_scale);
      //   } else {
      //     op_attr = attr_t::fuse_sum(sum_scale);
      //   }
      // } else if (attr.has_op_kind(kind::eltwise)) {
      //   op_attr = attr_t::fuse_relu();
      // }
      op_attr.set_output_scales(IDEEP_OP_SCALE_MASK(scale_size), op_scales);
      // op_attr.set_int_output_round_mode(round_mode::round_nearest);
      if (attr.has_op_kind(kind::sum)) {
        float sum_scale = dst_scales_in[0] / (dst.has_scale() ? dst.get_scale()[0] : 1.0f);
        if (attr.has_op_kind(kind::eltwise)) {
          op_attr.set_post_ops(post_ops::residual(sum_scale));
        } else {
          op_attr.set_post_ops(post_ops::sum(sum_scale));
        }
      } else if (attr.has_op_kind(kind::eltwise)) {
        op_attr.set_post_ops(post_ops::relu());
      }

      src_desc = {src.get_dims(), alowp_kind == LOWP_U8S8 ? data_type::u8 : data_type::s8};
      if (src.get_data_type() == data_type::f32) {
        src_attr = {0 , src_scales_in};
      }

      weights_desc = {weights_.get_dims(), data_type::s8};
      if (weights_.get_data_type() == data_type::f32) {
        weights_attr = {IDEEP_TENSOR_SCALE_MASK(scale_size, (groups > 1)), weights_scales_in};
      }

      if (with_bias) {
        bias_desc = {bias.get_dims(), data_type::s32};
        if (bias.get_data_type() == data_type::f32) {
          bias_attr = {IDEEP_TENSOR_SCALE_MASK(scale_size, false), bias_scales};
        }
      }
    } else {
      op_attr = attr;

      src_desc = {src.get_dims(), data_type::f32};
      if (src.has_scale()) {
        auto src_scale = src.get_scale();
        src_scale[0] = 1.0f / src_scale[0];
        src_attr = {0, src_scale};
      }

      weights_desc = weights_.get_desc();
      IDEEP_ENFORCE(weights_.get_data_type() == data_type::f32, "Incorrect data type in weights");

      if (with_bias) {
        IDEEP_ENFORCE(bias.get_data_type() == data_type::f32, "Incorrect data type in bias");
        bias_desc = bias.get_desc();
      }
    }

    // TODO: XPZ: is it ok to use src type as dst type?
    tensor::desc dst_desc;
    if (attr.has_op_kind(kind::sum))
      dst_desc = dst.get_desc();
    else
      dst_desc = {dst_dims, dst_data_type};

    auto pd = get_primitive_desc<with_bias>(
        src_desc, weights_desc, bias_desc, dst_desc,
        strides, dilates_, padding_l, padding_r, op_attr, aalgorithm,
        aprop_kind, aengine);

    auto expected_src = src.reorder_if_differ_in(pd.src_desc(), src_attr);
    auto expected_weights = weights_.reorder_if_differ_in(pd.weights_desc(), weights_attr);
    dst.reinit_if_necessary(pd.dst_desc());
    if (!dst_scales.empty() && dst_data_type != data_type::f32) {
      dst.set_scale(dst_scales_in);
    }

    if (with_bias) {
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


struct convolution_backward_data : public dnnl::convolution_backward_data {

  using super = dnnl::convolution_backward_data;

  static void compute(const tensor& diff_dst,
                      const tensor& weights,
                      const dims& diff_src_dims,
                      tensor& diff_src,
                      const dims& strides,
                      const dims& dilates,
                      const dims& padding_l,
                      const dims& padding_r,
                      const int groups,
                      algorithm aalgorithm = algorithm::convolution_direct,
                      const engine& aengine = engine::cpu_engine()) {
    // make weights and dilates compatible with DNNL
    auto weights_ = weights.make_grouped_weights(groups);
    auto dilates_ = utils::get_compatible_dilates(dilates);

    auto diff_dst_desc = diff_dst.get_desc().to_format_any();
    auto weights_desc = weights_.get_desc().to_format_any();

    tensor::desc diff_src_desc(diff_src_dims, diff_dst_desc.get_data_type());

    auto forward_hints =
        convolution_forward::get_primitive_desc</*with_bias=*/false>(
            diff_src_desc, weights_desc, tensor::desc(), diff_dst_desc, strides,
            dilates_, padding_l, padding_r);

    auto pd = primitive_desc(
        {aalgorithm, diff_src_desc, weights_desc, diff_dst_desc, strides,
         dilates_, padding_l, padding_r}, aengine, forward_hints);

    auto expected_diff_dst = diff_dst.reorder_if_differ_in(pd.diff_dst_desc());
    auto expected_weights = weights_.reorder_if_differ_in(pd.weights_desc());
    diff_src.reinit_if_necessary(pd.diff_src_desc());

    super(pd).execute(stream::default_stream(), 
                      {{DNNL_ARG_DIFF_DST, expected_diff_dst},
                       {DNNL_ARG_WEIGHTS, expected_weights},
                       {DNNL_ARG_DIFF_SRC, diff_src}});
  }
};


struct convolution_backward_weights
    : public dnnl::convolution_backward_weights {

  using super = dnnl::convolution_backward_weights;

  static void compute(const tensor& src,
                      const tensor& diff_dst,
                      const dims& diff_weights_dims,
                      tensor& diff_weights,
                      tensor& diff_bias,
                      const dims& strides,
                      const dims& dilates,
                      const dims& padding_l,
                      const dims& padding_r,
                      const int groups,
                      algorithm aalgorithm = algorithm::convolution_direct,
                      const engine& aengine = engine::cpu_engine()) {
    compute_impl</*with_diff_bias=*/true>(
        src, diff_dst, diff_weights_dims, diff_weights, diff_bias,
        strides, dilates, padding_l, padding_r, groups, aalgorithm, aengine);
  }

  static void compute(const tensor& src,
                      const tensor& diff_dst,
                      const dims& diff_weights_dims,
                      tensor& diff_weights,
                      const dims& strides,
                      const dims& dilates,
                      const dims& padding_l,
                      const dims& padding_r,
                      const int groups,
                      algorithm aalgorithm = algorithm::convolution_direct,
                      const engine& aengine = engine::cpu_engine()) {
    static tensor dummy_diff_bias;
    compute_impl</*with_diff_bias=*/false>(
        src, diff_dst, diff_weights_dims, diff_weights, dummy_diff_bias,
        strides, dilates, padding_l, padding_r, groups, aalgorithm, aengine);
  }

 private:
  template <bool with_diff_bias>
  static void compute_impl(const tensor& src,
                           const tensor& diff_dst,
                           const dims& diff_weights_dims,
                           tensor& diff_weights,
                           tensor& diff_bias,
                           const dims& strides,
                           const dims& dilates,
                           const dims& padding_l,
                           const dims& padding_r,
                           const int groups,
                           algorithm aalgorithm,
                           const engine& aengine) {

    // make diff_weights and dilates compatible with DNNL
    auto dilates_ = utils::get_compatible_dilates(dilates);
    auto diff_weights_desc =
        tensor::desc(diff_weights_dims, diff_dst.get_data_type())
            .to_format_any()
            .to_grouped(groups);

    auto diff_dst_desc = diff_dst.get_desc().to_format_any();
    auto src_desc = src.get_desc().to_format_any();

    auto diff_bias_desc = with_diff_bias
        ? tensor::desc({diff_dst.get_dim(1)}, diff_dst.get_data_type())
              .to_format_any()
        : tensor::desc();

    auto forward_hints =
        convolution_forward::get_primitive_desc<with_diff_bias>(
            src_desc, diff_weights_desc, diff_bias_desc, diff_dst_desc, strides,
            dilates_, padding_l, padding_r, attr_t(), aalgorithm,
            prop_kind::forward, aengine);

    auto pd = with_diff_bias
        ? primitive_desc({aalgorithm, src_desc, diff_weights_desc,
                          diff_bias_desc, diff_dst_desc, strides, dilates_,
                          padding_l, padding_r}, aengine, forward_hints)
        : primitive_desc({aalgorithm, src_desc, diff_weights_desc,
                          diff_dst_desc, strides, dilates_,
                          padding_l, padding_r}, aengine, forward_hints);

    auto expected_diff_dst = diff_dst.reorder_if_differ_in(pd.diff_dst_desc());
    auto expected_src = src.reorder_if_differ_in(pd.src_desc());
    diff_weights.reinit_if_necessary(pd.diff_weights_desc());

    if (with_diff_bias) {
      diff_bias.reinit_if_necessary(pd.diff_bias_desc());
      super(pd).execute(stream::default_stream(),
                        {{DNNL_ARG_DIFF_DST, expected_diff_dst},
                         {DNNL_ARG_SRC, expected_src},
                         {DNNL_ARG_DIFF_WEIGHTS, diff_weights},
                         {DNNL_ARG_DIFF_BIAS, diff_bias}});
    } else {
      super(pd).execute(stream::default_stream(),
                        {{DNNL_ARG_DIFF_DST, expected_diff_dst},
                         {DNNL_ARG_SRC, expected_src},
                         {DNNL_ARG_DIFF_WEIGHTS, diff_weights}});
    }

    if (groups > 1) {
      diff_weights.reshape(diff_weights_dims);
    }
  }
};
}  // namespace ideep

#endif
