#ifndef IDEEP_KERNELS_CONV_HPP
#define IDEEP_KERNELS_CONV_HPP

#include "common.hpp"

namespace ideep {

struct convolution_forward : public dnnl::convolution_forward {

  // fp32 w/ bias
  static void compute(
      const tensor& src,
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
      engine aengine = engine::cpu_engine()) {

    auto weights_ = weights.make_tmp_grouped_weights_if_necessary(groups);

    auto src_desc = src.get_desc().to_format_any();
    auto weights_desc = weights_.get_desc().to_format_any();
    auto bias_desc = bias.get_desc().to_format_any();
    auto dst_desc = dst.get_desc().to_format_any();

    auto op_desc = dnnl::convolution_forward::desc(
        aprop_kind, aalgorithm, src_desc, weights_desc, bias_desc, dst_desc,
        strides, utils::get_compatible_dilates(dilates), padding_l, padding_r);

    auto pd = dnnl::convolution_forward::primitive_desc(op_desc, attr, aengine);

    auto expected_src = src.reorder_if_necessary(pd.src_desc());
    auto expected_weights = weights_.reorder_if_necessary(pd.weights_desc());
    dst.reinit_if_necessary(pd.dst_desc());

    dnnl::convolution_forward(pd)
        .execute(stream::default_stream(), {
          {DNNL_ARG_SRC, expected_src},
          {DNNL_ARG_WEIGHTS, expected_weights},
          {DNNL_ARG_BIAS, bias},
          {DNNL_ARG_DST, dst}
        });
  }

  // fp32 w/o bias
  static void compute(
      const tensor& src,
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
      engine aengine = engine::cpu_engine()) {

    auto src_desc = src.get_desc().to_format_any();
    auto weights_desc = weights.get_desc().to_format_any();
    auto dst_desc = dst.get_desc().to_format_any();

    auto op_desc = dnnl::convolution_forward::desc(
        aprop_kind, aalgorithm, src_desc, weights_desc, dst_desc,
        strides, utils::get_compatible_dilates(dilates), padding_l, padding_r);

    auto pd = dnnl::convolution_forward::primitive_desc(op_desc, attr, aengine);

    auto expected_src = src.reorder_if_necessary(pd.src_desc());
    auto expected_weights = weights.reorder_if_necessary(pd.weights_desc());
    dst.reinit_if_necessary(pd.dst_desc());

    dnnl::convolution_forward(pd)
        .execute(stream::default_stream(), {
          {DNNL_ARG_SRC, expected_src},
          {DNNL_ARG_WEIGHTS, expected_weights},
          {DNNL_ARG_DST, dst}
        });
  }

  //  // for int8 w/ bias
  // template<class alloc = utils::allocator, bool with_bias = true>
  // static void compute(const tensor& src, const tensor& weights, const tensor& bias,
  //     const tdims_t& result_dims, tensor& dst, const tdims_t& strides, const tdims_t& dilates,
  //     const tdims_t& padding_l, const tdims_t& padding_r, int group,
  //     const scale_t& src_scales = scale_t(), const scale_t& weights_scales = scale_t(),
  //     const scale_t& dst_scales = scale_t(), const attr_t& attr = attr_t(),
  //     algorithm aalgorithm = algorithm::convolution_direct, prop_kind aprop_kind = prop_kind::forward,
  //     const lowp_kind alowp_kind = LOWP_U8S8) {
  //   compute<alloc, with_bias>(src, weights, bias, result_dims, dst, strides, dilates,
  //       padding_l, padding_r, group, src_scales, weights_scales, dst_scales, attr,
  //       aalgorithm, aprop_kind, appading_kind, alowp_kind);
  // }

  // // for int8 w/o bias
  // template<class alloc = utils::allocator>
  // static void compute(const tensor& src, const tensor& weights,
  //     const tdims_t& result_dims, tensor& dst, const tdims_t& strides, const tdims_t& dilates,
  //     const tdims_t& padding_l, const tdims_t& padding_r, int group,
  //     const scale_t& src_scales = scale_t(), const scale_t& weights_scales = scale_t(),
  //     const scale_t& dst_scales = scale_t(), const attr_t& attr = attr_t(),
  //     algorithm aalgorithm = algorithm::convolution_direct, prop_kind aprop_kind = prop_kind::forward,
  //     const lowp_kind alowp_kind = LOWP_U8S8) {
  //   static tensor dummy_bias;
  //   compute<alloc, false>(src, weights, dummy_bias, result_dims, dst, strides, dilates,
  //       padding_l, padding_r, group, src_scales, weights_scales, dst_scales, attr,
  //       aalgorithm, aprop_kind, appading_kind, alowp_kind);
  // }

  // // for fp32 w/ bias
  // template<class alloc = utils::allocator, bool with_bias = true>
  // static void compute(const tensor& src, const tensor& weights, const tensor& bias,
  //     const tdims_t& result_dims, tensor& dst, const tdims_t& strides, const tdims_t& dilates,
  //     const tdims_t& padding_l, const tdims_t& padding_r, int group, const attr_t& attr = attr_t(),
  //     algorithm aalgorithm = algorithm::convolution_direct, prop_kind aprop_kind = prop_kind::forward) {
  //   static scale_t dummy_scales;
  //   compute<alloc, with_bias>(src, weights, bias, result_dims, dst, strides, dilates,
  //       padding_l, padding_r, group, dummy_scales, dummy_scales, dummy_scales, attr,
  //       aalgorithm, aprop_kind, appading_kind);
  // }

  // // for fp32 w/o bias
  // template<class alloc = utils::allocator>
  // static void compute(const tensor& src, const tensor& weights,
  //     const tdims_t& result_dims, tensor& dst, const tdims_t& strides, const tdims_t& dilates,
  //     const tdims_t& padding_l, const tdims_t& padding_r, int group, const attr_t& attr = attr_t(),
  //     algorithm aalgorithm = algorithm::convolution_direct, prop_kind aprop_kind = prop_kind::forward) {
  //   static tensor dummy_bias;
  //   compute<alloc, false>(src, weights, dummy_bias, result_dims, dst, strides, dilates,
  //       padding_l, padding_r, group, attr, aalgorithm, aprop_kind, appading_kind);
  // }

  // static tdesc_t expected_weights_descriptor(const tdims_t& weights_dims,
  //     tdtype_t dtype = tdtype_t::f32, const tdims_t& strides = {1, 1},
  //     const tdims_t& padding_l = {0, 0}, const tdims_t& padding_r = {0, 0},
  //     const tdims_t& dilates = {0, 0}, int group = 1, algorithm aalgorithm = algorithm::convolution_direct,
  //     prop_kind aprop_kind = prop_kind::forward, tdtype_t x_dtype = tdtype_t::f32,
  //     const tdims_t& src_dims = tdims_t()) {
  //   auto dims_in = weights_dims;
  //   if (group > 1 && !IDEEP_IS_GROUPED_4DIMS(dims_in)) {
  //     tensor::group_dims(dims_in, group);
  //   }
  //   auto ndims = dims_in.size();
  //   auto grouped = IDEEP_IS_GROUPED_4DIMS(dims_in);
  //   auto g = grouped ? dims_in[0] : 1;
  //   auto dilates_in = utils::get_compatible_dilates(dilates);

  //   IDEEP_ENFORCE(!(aalgorithm == algorithm::convolution_winograd && src_dims.empty()),
  //       "Incorrect src_dims");
  //   auto ic = g * dims_in[1 + grouped];
  //   auto oc = g * dims_in[0 + grouped];
  //   auto kh = dims_in[ndims - 2];
  //   auto kw = dims_in[ndims - 1];
  //   int mb, h, w;
  //   if (src_dims.empty()) {
  //     // Construct a dummy case
  //     mb = 1;
  //     h = 2 * kh;
  //     w = 4 * kw;
  //   } else {
  //     // Use the real data
  //     mb = src_dims[0];
  //     h = src_dims[2];
  //     w = src_dims[3];
  //   }
  //   auto oh = (h - ((kh - 1) * (dilates_in[0] + 1) + 1) + (padding_l[0] + padding_r[0])) / strides[0] + 1;
  //   auto ow = (w - ((kw - 1) * (dilates_in[1] + 1) + 1) + (padding_l[1] + padding_r[1])) / strides[1] + 1;

  //   tdims_t x_dims = { mb, ic, h, w};
  //   tdims_t y_dims = { mb, oc, oh, ow};
  //   auto y_dtype = (dtype != tdtype_t::s8) ? dtype : tdtype_t::s32;
  //   tdesc_t x_desc(x_dims, x_dtype, format::nchw);
  //   tdesc_t y_desc(y_dims, y_dtype, format::nchw);
  //   tdesc_t weights_desc(dims_in, dtype, grouped ? format::goihw : format::oihw);

  //   // FIXME: workaroud winograd format issue in inference
  //   // If prop_kind == forward_inference, the dnnl_wino_fmt for weights is required by winograd primitive.
  //   // Then, in the cases of variable input shape, the detials of dnnl_wino_fmt will be changed.
  //   // And, extra weihgts reorder is inevitable each time, leading to bad performance.
  //   // Here, we set the prop_kind to forward, in order to reorder and cache weights as blocked format,
  //   // instead of dnnl_wino_fmt.
  //   auto apkind = aprop_kind;
  //   if (aalgorithm == algorithm::convolution_winograd && aprop_kind == prop_kind::forward_inference) {
  //     apkind = prop_kind::forward;
  //   }

  //   convolution_forward comp(x_desc, weights_desc, tdesc_t(), y_desc, strides, dilates, padding_l, padding_r,
  //       attr_t(), aalgorithm, apkind);
  //   return comp.dup_descriptor_of(query::weights_pd);
  // }
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