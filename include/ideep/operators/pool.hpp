#ifndef IDEEP_OPERATORS_POOL_HPP
#define IDEEP_OPERATORS_POOL_HPP

namespace ideep {

struct pooling_forward : public dnnl::pooling_forward {
  using super = dnnl::pooling_forward;
  static void compute(const tensor& src,
                      const dims& output_sizes,
                      tensor& dst,
                      const dims& strides,
                      const dims& kernel,
                      const dims& padding_l,
                      const dims& padding_r,
                      algorithm aalgorithm,
                      prop_kind aprop_kind = prop_kind::forward,
                      const engine& aengine = engine::cpu_engine()) {
    bool with_workspace = aprop_kind == prop_kind::forward_training &&
                          aalgorithm == dnnl::algorithm::pooling_max;

    auto src_desc = src.get_desc();
    auto dst_desc = tensor::desc(output_sizes, data_type::f32);

    auto pd = primitive_desc(
        {aprop_kind, aalgorithm, src_desc, dst_desc, strides, kernel, padding_l,
         padding_r}, aengine);

    auto expected_src = src.reorder_if_necessary(pd.src_desc());
    dst.reinit_if_necessary(pd.dst_desc());
    exec_args args = {{DNNL_ARG_SRC, expected_src}, {DNNL_ARG_DST, dst}};
    if (with_workspace) {
      dst.init_workspace(pd.workspace_desc());
      args.insert({DNNL_ARG_WORKSPACE, dst.get_workspace()});
    }
    super(pd).execute(stream::default_stream(), args);
 }

private:
  // dims infer_output_sizes(const dims& input_size,
  //                            const dims& kernel_size,
  //                            const dims& stride,
  //                            const dims& padding_l,
  //                            const dims& padding_r,
  //                            const dims& dilation,
  //                            bool ceil_mode) {

  //   auto dim = input_size.size();
  //   dims output_size(dim);
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
  using super = dnnl::pooling_backward;
  static void compute(const tensor& diff_dst,
                      const tensor& dst,
                      const tensor& src,
                      tensor& diff_src,
                      const dims& strides,
                      const dims& kernel,
                      const dims& padding_l,
                      const dims& padding_r,
                      algorithm aalgorithm,
                      const engine& aengine = engine::cpu_engine()) {
    auto src_desc = src.get_desc();
    auto dst_desc = dst.get_desc();
    auto forward_hints = pooling_forward::primitive_desc(
        {prop_kind::forward, aalgorithm, src_desc, dst_desc, strides, kernel,
         padding_l, padding_r}, aengine);
    auto pd = primitive_desc(
        {aalgorithm, src_desc, dst_desc, strides, kernel, padding_l, padding_r},
        aengine, forward_hints);
    auto expected_diff_dst = diff_dst.reorder_if_necessary(pd.diff_dst_desc());
    diff_src.reinit_if_necessary(pd.diff_src_desc());
    exec_args args = {{DNNL_ARG_DIFF_DST, expected_diff_dst},
                      {DNNL_ARG_DIFF_SRC, diff_src}};

    if (dst.has_workspace()) {
      auto expected_workspace =
          dst.get_workspace().reorder_if_necessary(pd.workspace_desc());
      args.insert({DNNL_ARG_WORKSPACE, expected_workspace});
    }
    super(pd).execute(stream::default_stream(), args);
  }
};

}  // namespace ideep

#endif
