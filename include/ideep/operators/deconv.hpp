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

    // auto output_size = infer_output_size(
    //     src_desc_any, weights_desc_any, padding_l, padding_r, strides, dilates);
    // auto dst_desc_any = tensor::desc(
    //     output_size, src_desc_any.get_data_type(), format_tag::any);

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

  // static dims infer_output_size(const tensor::desc& input_desc,
  //                               const tensor::desc& weights_desc,
  //                               const dims& padding_l,
  //                               const dims& padding_r,
  //                               const dims& output_padding,
  //                               const dims& strides,
  //                               const dims& dilates) {
  //   auto input_size = input_desc.get_dims();
  //   auto kernel_size = weights_desc.get_dims();
  //   auto with_groups = kernel_size.size() == (input_size.size() + 1);

  //   auto dim = input_size.size();
  //   dims output_size(dim);
  //   output_size[0] = input_size[0];
  //   output_size[1] = kernel_size[0] * (with_groups ? kernel_size[1] : 1);
  //   for (size_t d = 2; d < dim; ++d) {
  //     auto src = input_size[d];
  //     auto ker = kernel_size[with_groups + d];
  //     auto str = strides[d - 2];
  //     auto dil = dilates[d - 2];
  //     auto pad = padding_l[d - 2] + padding_r[d - 2];
  //     auto out_pad = output_padding[d - 2];
  //     auto ker_range = 1 + (ker - 1) * (dil + 1);
  //     output_size[d] = (src - 1) * str - pad + ker + out_pad;
  //   }
  //   return output_size;
  // }
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