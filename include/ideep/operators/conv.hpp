#ifndef IDEEP_OPERATORS_CONV_HPP
#define IDEEP_OPERATORS_CONV_HPP

namespace ideep {

struct convolution_forward : public dnnl::convolution_forward {

  using super = dnnl::convolution_forward;

  // fp32 w/ bias
  static void compute(const tensor& src,
                      const tensor& weights,
                      const tensor& bias,
                      const tdims_t& output_sizes,
                      tensor& dst,
                      const tdims_t& strides,
                      const tdims_t& dilates,
                      const tdims_t& padding_l,
                      const tdims_t& padding_r,
                      int groups,
                      const attr_t& attr = attr_t(),
                      algorithm aalgorithm = algorithm::convolution_direct,
                      prop_kind aprop_kind = prop_kind::forward,
                      const engine& aengine = engine::cpu_engine()) {
    compute_impl</*with_bias=*/true>(
        src, weights, bias, dst, strides, dilates, padding_l, padding_r,
        groups, attr, aalgorithm, aprop_kind, aengine);
  }

  // fp32 w/o bias
  static void compute(const tensor& src,
                      const tensor& weights,
                      const tdims_t& output_sizes,
                      tensor& dst,
                      const tdims_t& strides,
                      const tdims_t& dilates,
                      const tdims_t& padding_l,
                      const tdims_t& padding_r,
                      int groups,
                      const attr_t& attr = attr_t(),
                      algorithm aalgorithm = algorithm::convolution_direct,
                      prop_kind aprop_kind = prop_kind::forward,
                      const engine& aengine = engine::cpu_engine()) {
    static tensor dummy_bias;
    compute_impl</*with_bias=*/false>(
        src, weights, dummy_bias, dst, strides, dilates, padding_l, padding_r,
        groups, attr, aalgorithm, aprop_kind, aengine);
  }

  // TODO: XPZ: refactor it
  static dnnl::memory::desc expected_weights_desc(
      const tdims_t& weights_dims,
      tensor::data_type dtype = tensor::data_type::f32,
      const tdims_t& strides = {1, 1},
      const tdims_t& padding_l = {0, 0},
      const tdims_t& padding_r = {0, 0},
      const tdims_t& dilates = {0, 0},
      int groups = 1,
      algorithm aalgorithm = algorithm::convolution_direct,
      prop_kind aprop_kind = prop_kind::forward,
      tensor::data_type x_dtype = tensor::data_type::f32,
      const tdims_t& src_dims = tdims_t()) {

    auto grouped = groups > 1;
    auto weights_desc_usr = tensor::desc(weights_dims, dtype, format_tag::oihw);
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

    tdims_t x_dims = { mb, ic, h, w };
    tdims_t y_dims = { mb, oc, oh, ow };
    auto y_dtype =
        dtype != tensor::data_type::s8 ? dtype : tensor::data_type::s32;
    tensor::desc src_desc(x_dims, x_dtype, format_tag::nchw);
    tensor::desc dst_desc(y_dims, y_dtype, format_tag::nchw);

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
        padding_l, padding_r, groups, attr_t(), aalgorithm, apkind);

    if (!grouped) {
      return pd.weights_desc();
    } else {
      // hide group info from the outside world
      return tensor::desc(pd.weights_desc()).to_ungrouped();
    }
  }

  template <bool with_bias>
  static primitive_desc get_primitive_desc(
      const tensor::desc& src_desc,
      const tensor::desc& weights_desc,
      const tensor::desc& bias_desc,
      const tensor::desc& dst_desc,
      const tdims_t& strides,
      const tdims_t& dilates,
      const tdims_t& padding_l,
      const tdims_t& padding_r,
      int groups,
      const attr_t& attr = attr_t(),
      algorithm aalgorithm = algorithm::convolution_direct,
      prop_kind aprop_kind = prop_kind::forward,
      const engine& aengine = engine::cpu_engine()) {
    auto src_desc_any = src_desc.to_format_any();
    auto weights_desc_any = weights_desc.to_format_any();
    auto bias_desc_any = with_bias ? bias_desc.to_format_any() : tensor::desc();

    auto output_size = infer_output_size(
        src_desc_any, weights_desc_any, padding_l, padding_r, strides, dilates);
    auto dst_desc_any = tensor::desc(
        output_size, src_desc_any.get_data_type(), format_tag::any);

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
                           tensor& dst,
                           const tdims_t& strides,
                           const tdims_t& dilates,
                           const tdims_t& padding_l,
                           const tdims_t& padding_r,
                           int groups,
                           const attr_t& attr,
                           algorithm aalgorithm,
                           prop_kind aprop_kind,
                           const engine& aengine) {

    // make weights and dilates compatible with DNNL
    auto weights_ = weights.make_grouped_weights(groups);
    auto dilates_ = utils::get_compatible_dilates(dilates);

    auto pd = get_primitive_desc<with_bias>(
        src.get_desc(), weights_.get_desc(), bias.get_desc(), dst.get_desc(),
        strides, dilates_, padding_l, padding_r, groups, attr, aalgorithm,
        aprop_kind, aengine);

    auto expected_src = src.reorder_if_necessary(pd.src_desc());
    auto expected_weights = weights_.reorder_if_necessary(pd.weights_desc());
    dst.reinit_if_necessary(pd.dst_desc());

    if (with_bias) {
      super(pd).execute(stream::default_stream(), 
                        {{DNNL_ARG_SRC, expected_src},
                         {DNNL_ARG_WEIGHTS, expected_weights},
                         {DNNL_ARG_BIAS, bias},
                         {DNNL_ARG_DST, dst}});
    } else {
      super(pd).execute(stream::default_stream(), 
                        {{DNNL_ARG_SRC, expected_src},
                         {DNNL_ARG_WEIGHTS, expected_weights},
                         {DNNL_ARG_DST, dst}});
    }
  }

  inline static tdims_t infer_output_size(const tensor::desc& input_desc,
                                          const tensor::desc& weights_desc,
                                          const tdims_t& padding_l,
                                          const tdims_t& padding_r,
                                          const tdims_t& strides,
                                          const tdims_t& dilates) {
    // XPZ: TODO: Assert format. Assume NCHW
    auto input_size = input_desc.get_dims();
    auto kernel_size = weights_desc.get_dims();
    auto with_groups = kernel_size.size() == (input_size.size() + 1);

    auto dim = input_size.size();
    tdims_t output_size(dim);
    output_size[0] = input_size[0];
    output_size[1] = kernel_size[0] * (with_groups ? kernel_size[1] : 1);
    for (size_t d = 2; d < dim; ++d) {
      auto src = input_size[d];
      auto ker = kernel_size[with_groups + d];
      auto str = strides[d - 2];
      auto dil = dilates[d - 2];
      auto pad_l = padding_l[d - 2];
      auto pad_r = padding_r[d - 2];
      auto ker_range = 1 + (ker - 1) * (dil + 1);
      output_size[d] = (src + pad_l + pad_r - ker_range) / str + 1;
    }
    return output_size;
  }
};


struct convolution_backward_data : public dnnl::convolution_backward_data {

  using super = dnnl::convolution_backward_data;

  static void compute(const tensor& diff_dst,
                      const tensor& weights,
                      const tdims_t& diff_src_dims,
                      tensor& diff_src,
                      const tdims_t& strides,
                      const tdims_t& dilates,
                      const tdims_t& padding_l,
                      const tdims_t& padding_r,
                      const int groups,
                      algorithm aalgorithm = algorithm::convolution_direct,
                      const engine& aengine = engine::cpu_engine()) {
    // make weights and dilates compatible with DNNL
    auto weights_ = weights.make_grouped_weights(groups);
    auto dilates_ = utils::get_compatible_dilates(dilates);

    auto diff_dst_desc = diff_dst.get_desc().to_format_any();
    auto weights_desc = weights_.get_desc().to_format_any();

    tensor::desc diff_src_desc = tensor::desc(
        diff_src_dims, diff_dst_desc.get_data_type(), format_tag::any);

    auto forward_hints =
        convolution_forward::get_primitive_desc</*with_bias=*/false>(
            diff_src_desc, weights_desc, tensor::desc(), diff_dst_desc, strides,
            dilates, padding_l, padding_r, groups);

    auto pd = primitive_desc(
        {aalgorithm, diff_src_desc, weights_desc, diff_dst_desc, strides,
         dilates_, padding_l, padding_r}, aengine, forward_hints);

    auto expected_diff_dst = diff_dst.reorder_if_necessary(pd.diff_dst_desc());
    auto expected_weights = weights_.reorder_if_necessary(pd.weights_desc());
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
                      const tdims_t& diff_weights_dims,
                      tensor& diff_weights,
                      tensor& diff_bias,
                      const tdims_t& strides,
                      const tdims_t& dilates,
                      const tdims_t& padding_l,
                      const tdims_t& padding_r,
                      const int groups,
                      algorithm aalgorithm = algorithm::convolution_direct,
                      const engine& aengine = engine::cpu_engine()) {
    compute_impl</*with_diff_bias=*/true>(
        src, diff_dst, diff_weights_dims, diff_weights, diff_bias,
        strides, dilates, padding_l, padding_r, groups, aalgorithm, aengine);
  }

  static void compute(const tensor& src,
                      const tensor& diff_dst,
                      const tdims_t& diff_weights_dims,
                      tensor& diff_weights,
                      const tdims_t& strides,
                      const tdims_t& dilates,
                      const tdims_t& padding_l,
                      const tdims_t& padding_r,
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
                           const tdims_t& diff_weights_dims,
                           tensor& diff_weights,
                           tensor& diff_bias,
                           const tdims_t& strides,
                           const tdims_t& dilates,
                           const tdims_t& padding_l,
                           const tdims_t& padding_r,
                           const int groups,
                           algorithm aalgorithm,
                           const engine& aengine) {

    // make diff_weights and dilates compatible with DNNL
    auto diff_weights_desc =
        tensor::desc(diff_weights_dims, diff_dst.get_data_type(),
                     format_tag::any).to_grouped(groups);
    auto dilates_ = utils::get_compatible_dilates(dilates);

    auto diff_dst_desc = diff_dst.get_desc().to_format_any();
    auto src_desc = src.get_desc().to_format_any();

    auto diff_bias_desc = with_diff_bias
        ? tensor::desc({diff_dst.get_dim(1)}, diff_dst.get_data_type(),
                        format_tag::any)
        : tensor::desc();

    auto forward_hints =
        convolution_forward::get_primitive_desc<with_diff_bias>(
            src_desc, diff_weights_desc, diff_bias_desc, diff_dst_desc, strides,
            dilates_, padding_l, padding_r, groups, attr_t(), aalgorithm,
            prop_kind::forward, aengine);

    auto pd = with_diff_bias
        ? primitive_desc({aalgorithm, src_desc, diff_weights_desc,
                          diff_bias_desc, diff_dst_desc, strides, dilates_,
                          padding_l, padding_r}, aengine, forward_hints)
        : primitive_desc({aalgorithm, src_desc, diff_weights_desc,
                          diff_dst_desc, strides, dilates_,
                          padding_l, padding_r}, aengine, forward_hints);

    auto expected_diff_dst = diff_dst.reorder_if_necessary(pd.diff_dst_desc());
    auto expected_src = src.reorder_if_necessary(pd.src_desc());
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
