#ifndef IDEEP_KERNELS_CONV_HPP
#define IDEEP_KERNELS_CONV_HPP

namespace ideep {

struct convolution_forward : public dnnl::convolution_forward {

  typedef dnnl::convolution_forward super;

  // fp32 w/ bias
  static void compute(const tensor& src,
                      const tensor& weights,
                      const tensor& bias,
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
    compute_impl</*with_bias=*/true>(src, weights, bias, dst, strides,
                                     dilates, padding_l, padding_r, groups,
                                     attr, aalgorithm, aprop_kind, aengine);
  }

  // fp32 w/o bias
  static void compute(const tensor& src,
                      const tensor& weights,
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
    compute_impl</*with_bias=*/false>(src, weights, dummy_bias, dst, strides,
                                      dilates, padding_l, padding_r, groups,
                                      attr, aalgorithm, aprop_kind, aengine);
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

    auto weights_desc =
        tensor::desc(weights_dims, dtype, format_tag::oihw).to_grouped(groups);
    auto dims_in = weights_desc.get_dims();

    auto ndims = dims_in.size();
    auto grouped = groups > 1;
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
  
    auto pd =
        primitive_desc({aprop_kind, aalgorithm, src_desc, weights_desc,
                        dst_desc, strides, dilates_, padding_l, padding_r},
                       engine::cpu_engine());

    return pd.weights_desc();
  }

private:
  template<bool with_bias>
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
    auto weights_ = weights.make_tmp_grouped_weights_if_necessary(groups);
    auto dilates_ = utils::get_compatible_dilates(dilates);

    auto src_desc = src.get_desc().to_format_any();
    auto weights_desc = weights_.get_desc().to_format_any();
    auto bias_desc = bias.get_desc().to_format_any();

    tensor::desc dst_desc;
    if (!dst.is_empty()) {
      dst_desc = dst.get_desc().to_format_any();
    } else {
      auto output_size = infer_output_size(
          src, weights_, padding_l, padding_r, strides, dilates_);
      dst_desc = tensor::desc(
          output_size, src_desc.get_data_type(), tensor::format_tag::any);
    }

    auto pd = with_bias
        ? primitive_desc({aprop_kind, aalgorithm, src_desc, weights_desc,
                          bias_desc, dst_desc, strides, dilates_, padding_l,
                          padding_r}, attr, aengine)
        : primitive_desc({aprop_kind, aalgorithm, src_desc, weights_desc,
                          dst_desc, strides, dilates_, padding_l,
                          padding_r}, attr, aengine);

    auto expected_src = src.reorder_if_necessary(pd.src_desc());
    auto expected_weights = weights_.reorder_if_necessary(pd.weights_desc());
    dst.reinit_if_necessary(pd.dst_desc());

    std::unordered_map<int, dnnl::memory> args
        {{DNNL_ARG_SRC, expected_src},
         {DNNL_ARG_WEIGHTS, expected_weights},
         {DNNL_ARG_DST, dst}};

    if (with_bias) {
      args.insert({DNNL_ARG_BIAS, bias});
    }

    super(pd).execute(stream::default_stream(), args);
  }

  inline static tdims_t infer_output_size(const tensor& input,
                                          const tensor& weights,
                                          const tdims_t& padding_l,
                                          const tdims_t& padding_r,
                                          const tdims_t& strides,
                                          const tdims_t& dilates) {
    // XPZ: TODO: Assert format. Assume NCHW
    auto input_size = input.get_dims();
    auto kernel_size = weights.get_dims();
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
  template <class alloc = utils::allocator>
  static void compute(const tensor& grady, const tensor& weights,
                      const tdims_t& gradx_dims, tensor& gradx,
                      const tdims_t& strides, const tdims_t& dilates,
                      const tdims_t& padding_l, const tdims_t& padding_r,
                      const int group,
                      algorithm aalgorithm = algorithm::convolution_direct) {}
};


struct convolution_backward_weights : public dnnl::convolution_backward_weights {
  template<bool with_gradb = true>
  static void compute(const tensor& src, const tensor& grady, const tdims_t& gradw_dims,
      tensor& gradw, tensor& gradb, const tdims_t& strides, const tdims_t& dilates, const tdims_t& padding_l,
      const tdims_t& padding_r, const int group, algorithm aalgorithm = algorithm::convolution_direct) {
  }

  static void compute(const tensor& src, const tensor& grady, const tdims_t& gradw_dims,
      tensor& gradw, const tdims_t& strides, const tdims_t& dilates,
      const tdims_t& padding_l, const tdims_t& padding_r, const int group,
      algorithm aalgorithm = algorithm::convolution_direct) {
  }
};

}  // namespace ideep

#endif