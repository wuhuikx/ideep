#ifndef IDEEP_OPERATORS_DECONV_HPP
#define IDEEP_OPERATORS_DECONV_HPP

namespace ideep {

struct convolution_transpose_forward : public dnnl::deconvolution_forward {
  template <bool with_bias = true>
  static void compute(const tensor& src,
                      const tensor& weights,
                      const tensor& bias,
                      const tdims_t& result_dims,
                      tensor& dst,
                      const tdims_t& strides,
                      const tdims_t& padding_l,
                      const tdims_t& padding_r,
                      const tdims_t& dilates = {1, 1},
                      const attr_t& attr = attr_t(),
                      algorithm aalgorithm = algorithm::deconvolution_direct,
                      prop_kind aprop_kind = prop_kind::forward) {}

  static void compute(const tensor& src,
                      const tensor& weights,
                      const tdims_t& result_dims,
                      tensor& dst,
                      const tdims_t& strides,
                      const tdims_t& padding_l,
                      const tdims_t& padding_r,
                      const tdims_t& dilates = {1, 1},
                      const attr_t& attr = attr_t(),
                      algorithm aalgorithm = algorithm::deconvolution_direct,
                      prop_kind aprop_kind = prop_kind::forward) {}
};

struct convolution_transpose_backward_data
    : public dnnl::deconvolution_backward_data {
  static void compute(const tensor& grady, const tensor& weights,
                      const tdims_t& gradx_dims, tensor& gradx,
                      const tdims_t& strides, const tdims_t& padding_l,
                      const tdims_t& padding_r, const tdims_t& dilates = {1, 1},
                      algorithm aalgorithm = algorithm::deconvolution_direct) {}
};

struct convolution_transpose_backward_weights
    : public dnnl::deconvolution_backward_weights {
  template <bool with_gradb = true>
  static void compute(const tensor& src, const tensor& grady,
                      const tdims_t& gradw_dims, tensor& gradw, tensor& gradb,
                      const tdims_t& strides, const tdims_t& padding_l,
                      const tdims_t& padding_r, const tdims_t& dilates = {1, 1},
                      algorithm aalgorithm = algorithm::deconvolution_direct) {}

  static void compute(const tensor& src, const tensor& grady,
                      const tdims_t& gradw_dims, tensor& gradw,
                      const tdims_t& strides, const tdims_t& padding_l,
                      const tdims_t& padding_r, const tdims_t& dilates = {1, 1},
                      algorithm aalgorithm = algorithm::deconvolution_direct) {}
};

}  // namespace ideep

#endif