#ifndef IDEEP_OPERATORS_POOL_HPP
#define IDEEP_OPERATORS_POOL_HPP

namespace ideep {

struct pooling_forward : public dnnl::pooling_forward {

  using super = dnnl::pooling_forward;

  static void compute(const tensor& src,
                      tensor& dst,
                      const tdims_t& strides,
                      const tdims_t& kernel,
                      const tdims_t& padding_l,
                      const tdims_t& padding_r,
                      algorithm aalgorithm,
                      prop_kind aprop_kind = prop_kind::forward_inference,
                      const engine& aengine = engine::cpu_engine()) {

    // XPZ: TODO: workspace for training
    // bool with_workspace = aprop_kind == prop_kind::forward_training &&
    //                       aalgorithm == dnnl::algorithm::pooling_max;

    auto src_desc = src.get_desc();
    auto dst_desc = dst.get_desc();

    auto pd = primitive_desc(
        {aprop_kind, aalgorithm, src_desc, dst_desc, strides, kernel, padding_l,
         padding_r}, aengine);

    dst.reinit_if_necessary(pd.dst_desc());

    super(pd).execute(stream::default_stream(),
                      {{DNNL_ARG_SRC, src}, {DNNL_ARG_DST, dst}});
  }

private:
  // tdims_t infer_output_sizes(const tdims_t& input_size,
  //                            const tdims_t& kernel_size,
  //                            const tdims_t& stride,
  //                            const tdims_t& padding_l,
  //                            const tdims_t& padding_r,
  //                            const tdims_t& dilation,
  //                            bool ceil_mode) {

  //   auto dim = input_size.size();
  //   tdims_t output_size(dim);
  //   output_size[0] = input_size[0];
  //   output_size[1] = input_size[1];
  //   for (size_t i = 2; i < dim; ++i) {
  //     output_size[i] = pooling_output_shape_pad_lr<int64_t>(
  //       input_size[i],
  //       kernel_size[i - 2],
  //       padding_l[i - 2],
  //       padding_r[i - 2],
  //       stride[i - 2],
  //       dilation[i - 2],
  //       ceil_mode
  //     );
  //   }

  //   return output_size;
  // }
};

struct pooling_backward : public dnnl::pooling_backward {
  static void compute(const tensor& grady,
                      const tensor& y,
                      const tensor& x,
                      tensor& gradx,
                      const tdims_t& strides,
                      const tdims_t& kernel,
                      const tdims_t& padding_l,
                      const tdims_t& padding_r,
                      algorithm aalgorithm) {}
};

}  // namespace ideep

#endif