#ifndef IDEEP_OPERATORS_DECONV_HPP
#define IDEEP_OPERATORS_DECONV_HPP

namespace ideep {

// XPZ: TODO: DECONV FORWARD IS ENTIRELY IDENTICAL TO CONV'S IMPL. MERGE THEM

struct convolution_transpose_forward : public dnnl::deconvolution_forward {

  using super = dnnl::deconvolution_forward;

  static void compute(const tensor& src,
                      const tensor& weights,
                      const tensor& bias,
                      const dims& dst_dims,
                      tensor& dst,
                      const dims& strides,
                      const dims& padding_l,
                      const dims& padding_r,
                      const dims& dilates = {1, 1},
                      int groups = 1,
                      const attr_t& attr = attr_t(),
                      algorithm aalgorithm = algorithm::deconvolution_direct,
                      prop_kind aprop_kind = prop_kind::forward,
                      const engine& aengine = engine::cpu_engine()) {
    compute_impl</*with_bias=*/true>(
        src, weights, bias, dst_dims, dst, strides, dilates,
        padding_l, padding_r, groups, attr, aalgorithm, aprop_kind, aengine);
  }

  static void compute(const tensor& src,
                      const tensor& weights,
                      const dims& dst_dims,
                      tensor& dst,
                      const dims& strides,
                      const dims& padding_l,
                      const dims& padding_r,
                      const dims& dilates = {1, 1},
                      int groups = 1,
                      const attr_t& attr = attr_t(),
                      algorithm aalgorithm = algorithm::deconvolution_direct,
                      prop_kind aprop_kind = prop_kind::forward,
                      const engine& aengine = engine::cpu_engine()) {
    static tensor dummy_bias;
    compute_impl</*with_bias=*/false>(
        src, weights, dummy_bias, dst_dims, dst, strides, dilates,
        padding_l, padding_r, groups, attr, aalgorithm, aprop_kind, aengine);
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
      algorithm aalgorithm = algorithm::deconvolution_direct,
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


  // static tdesc_t expected_weights_descriptor(const tdims_t& weights_dims,
  //     tdtype_t dtype = tdtype_t::f32, const tdims_t& strides = {1, 1}, const tdims_t& padding_l = {0, 0},
  //     const tdims_t& padding_r = {0, 0}, int group = 1, const tdims_t& dilates = {1, 1}) {
  //   auto dims_in = weights_dims;
  //   if (group > 1 && !IDEEP_IS_GROUPED_4DIMS(dims_in)) {
  //     tensor::group_dims(dims_in, group);
  //   }

  //   auto ndims = dims_in.size();
  //   auto grouped = IDEEP_IS_GROUPED_4DIMS(dims_in);
  //   auto g = grouped ? dims_in[0] : 1;
  //   auto ic = g * dims_in[1 + grouped];
  //   auto oc = g * dims_in[0 + grouped];
  //   auto kh = dims_in[ndims - 2];
  //   auto kw = dims_in[ndims - 1];
  //   auto h = 8 * kh;
  //   auto w = 8 * kw;
  //   auto oh = (h - 1) * strides[0] + (1 + (kh - 1) * (dilates[0])) - padding_l[0] - padding_r[0];
  //   auto ow = (w - 1) * strides[1] + (1 + (kw - 1) * (dilates[1])) - padding_l[1] - padding_r[1];
  //   tdims_t x_dims = {1, ic, h, w};
  //   tdims_t y_dims = {1, oc, oh, ow};
  //   auto x_dtype = (dtype != tdtype_t::s8) ? dtype : tdtype_t::u8;
  //   auto y_dtype = (dtype != tdtype_t::s8) ? dtype : tdtype_t::s32;

  //   tdesc_t x_desc(x_dims, x_dtype, format::nchw);
  //   tdesc_t y_desc(y_dims, y_dtype, format::nchw);
  //   tdesc_t weights_desc(dims_in, dtype, grouped ? format::goihw : format::oihw);

  //   convolution_transpose_forward comp(x_desc, weights_desc, tdesc_t(), y_desc,
  //       strides, dilates, padding_l, padding_r);
  //   return comp.dup_descriptor_of(query::weights_pd);
  // }


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
      const dims& src_dims = dims()) {

    // auto grouped = groups > 1;
    // auto weights_desc_usr = tensor::desc(weights_dims, dtype);
    // auto weights_desc =
    //     grouped ? weights_desc_usr.to_grouped(groups) : weights_desc_usr;

    // auto dims_in = weights_desc.get_dims();
    // auto ndims = dims_in.size();
    // auto g = grouped ? dims_in[0] : 1;
    // auto dilates_ = utils::get_compatible_dilates(dilates);

    // IDEEP_ENFORCE(
    //     !(aalgorithm == algorithm::convolution_winograd && src_dims.empty()),
    //     "Incorrect src_dims");
    // auto ic = g * dims_in[1 + grouped];
    // auto oc = g * dims_in[0 + grouped];
    // auto kh = dims_in[ndims - 2];
    // auto kw = dims_in[ndims - 1];
    // int mb, h, w;
    // if (src_dims.empty()) {
    //   // Construct a dummy case
    //   mb = 1;
    //   h = 2 * kh;
    //   w = 4 * kw;
    // } else {
    //   // Use the real data
    //   mb = src_dims[0];
    //   h = src_dims[2];
    //   w = src_dims[3];
    // }
    // auto oh = (h - ((kh - 1) * (dilates_[0] + 1) + 1) + (padding_l[0] + padding_r[0])) / strides[0] + 1;
    // auto ow = (w - ((kw - 1) * (dilates_[1] + 1) + 1) + (padding_l[1] + padding_r[1])) / strides[1] + 1;

    // dims x_dims = { mb, ic, h, w };
    // dims y_dims = { mb, oc, oh, ow };
    // auto y_dtype =
    //     dtype != data_type::s8 ? dtype : data_type::s32;
    // tensor::desc src_desc(x_dims, x_dtype);
    // tensor::desc dst_desc(y_dims, y_dtype);

    // // FIXME: workaroud winograd format issue in inference
    // // If prop_kind == forward_inference, the dnnl_wino_fmt for weights is required by winograd primitive.
    // // Then, in the cases of variable input shape, the detials of dnnl_wino_fmt will be changed.
    // // And, extra weihgts reorder is inevitable each time, leading to bad performance.
    // // Here, we set the prop_kind to forward, in order to reorder and cache weights as blocked format,
    // // instead of dnnl_wino_fmt.
    // auto apkind = aprop_kind;
    // if (aalgorithm == algorithm::convolution_winograd &&
    //     aprop_kind == prop_kind::forward_inference) {
    //   apkind = prop_kind::forward;
    // }

    // auto pd = get_primitive_desc</*with_bias=*/false>(
    //     src_desc, weights_desc, tensor::desc(), dst_desc, strides, dilates_,
    //     padding_l, padding_r, attr_t(), aalgorithm, apkind);

    // if (!grouped) {
    //   return pd.weights_desc();
    // } else {
    //   // hide group info from the outside world
    //   return tensor::desc(pd.weights_desc()).to_ungrouped();
    // }
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
                           const attr_t& attr,
                           algorithm aalgorithm,
                           prop_kind aprop_kind,
                           const engine& aengine) {

    // make weights and dilates compatible with DNNL
    auto weights_ = weights.make_grouped_weights(groups);
    auto dilates_ = utils::get_compatible_dilates(dilates);

    // TODO: XPZ: is it ok to use src type as dst type?
    tensor::desc dst_desc(dst_dims, src.get_data_type());

    auto pd = get_primitive_desc<with_bias>(
        src.get_desc(), weights_.get_desc(), bias.get_desc(), dst_desc,
        strides, dilates_, padding_l, padding_r, attr, aalgorithm,
        aprop_kind, aengine);

    auto expected_src = src.reorder_if_differ_in(pd.src_desc());
    auto expected_weights = weights_.reorder_if_differ_in(pd.weights_desc());
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
};

struct convolution_transpose_backward_data
    : public dnnl::deconvolution_backward_data {

  using super = dnnl::deconvolution_backward_data;

  static void compute(const tensor& diff_dst,
                      const tensor& weights,
                      const dims& diff_src_dims,
                      tensor& diff_src,
                      const dims& strides,
                      const dims& padding_l,
                      const dims& padding_r,
                      const dims& dilates = {1, 1},
                      const int groups = 1,
                      algorithm aalgorithm = algorithm::deconvolution_direct,
                      const engine& aengine = engine::cpu_engine()) {

  }
};

struct convolution_transpose_backward_weights
    : public dnnl::deconvolution_backward_weights {

  using super = dnnl::deconvolution_backward_weights;

  static void compute(const tensor& src,
                      const tensor& diff_dst,
                      const dims& diff_weights_dims,
                      tensor& diff_weights,
                      tensor& diff_bias,
                      const dims& strides,
                      const dims& padding_l,
                      const dims& padding_r,
                      const dims& dilates = {1, 1},
                      const int groups = 1,
                      algorithm aalgorithm = algorithm::deconvolution_direct,
                      const engine& aengine = engine::cpu_engine()) {

  }

  static void compute(const tensor& src,
                      const tensor& diff_dst,
                      const dims& diff_weights_dims,
                      tensor& diff_weights,
                      const dims& strides,
                      const dims& padding_l,
                      const dims& padding_r,
                      const dims& dilates = {1, 1},
                      const int groups = 1,
                      algorithm aalgorithm = algorithm::deconvolution_direct,
                      const engine& aengine = engine::cpu_engine()) {

  }
};

}  // namespace ideep

#endif