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

    // workaround: use src.get_desc() once issue intel/mkl-dnn#588 is resolved
    auto src_desc = src._get_unblocked_desc_if_4c_blocked();
    // auto src_desc = src.get_desc();
    auto dst_desc = tensor::desc(output_sizes, src.get_data_type()).to_format_any();

    auto pd = primitive_desc(
        {aprop_kind, aalgorithm, src_desc, dst_desc, strides, kernel, padding_l,
         padding_r}, aengine);

    // auto expected_src = src.reorder_if_differ_in(pd.src_desc());
    if (dst != src) {
      dst.reinit_if_necessary(pd.dst_desc());
      // exec_args args = {{DNNL_ARG_SRC, src}, {DNNL_ARG_DST, dst}};
      if (src.has_scale()) {
        dst.set_scale(src.get_scale());
      }
      // args.insert({DNNL_ARG_DST, dst});
      if (with_workspace) {
        dst.init_workspace(pd.workspace_desc());
        // args.insert({DNNL_ARG_WORKSPACE, dst.get_workspace()});
      }
    }

    exec_args args = {{DNNL_ARG_SRC, src}, {DNNL_ARG_DST, dst}};
    if (with_workspace) {
      args.insert({DNNL_ARG_WORKSPACE, dst.get_workspace()});
    }

    super(pd).execute(stream::default_stream(), args);
  }
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
    auto expected_diff_dst = diff_dst.reorder_if_differ_in(pd.diff_dst_desc());
    diff_src.reinit_if_necessary(pd.diff_src_desc());
    exec_args args = {{DNNL_ARG_DIFF_DST, expected_diff_dst},
                      {DNNL_ARG_DIFF_SRC, diff_src}};

    if (dst.has_workspace()) {
      auto expected_workspace =
          dst.get_workspace().reorder_if_differ_in(pd.workspace_desc());
      args.insert({DNNL_ARG_WORKSPACE, expected_workspace});
    }
    super(pd).execute(stream::default_stream(), args);
  }
};

}  // namespace ideep

#endif
