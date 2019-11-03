#ifndef IDEEP_OPERATORS_LRN_HPP
#define IDEEP_OPERATORS_LRN_HPP

namespace ideep {

struct lrn_forward : public dnnl::lrn_forward {

  using super = dnnl::lrn_forward;

  static void compute(const tensor& src,
                      tensor& dst,
                      tensor::dim local_size,
                      float alpha,
                      float beta,
                      float k = 1.0,
                      algorithm aalgorithm = algorithm::lrn_across_channels,
                      prop_kind aprop_kind = prop_kind::forward_training,
                      const engine& aengine = engine::cpu_engine()) {
    auto src_desc = src.get_desc();
    auto pd = primitive_desc({prop_kind::forward_inference, aalgorithm,
                              src_desc, local_size, alpha, beta, k}, aengine);

    auto expected_src = src.reorder_if_necessary(pd.src_desc());
    dst.reinit_if_necessary(pd.dst_desc());

    exec_args args = {{DNNL_ARG_SRC, expected_src}, {DNNL_ARG_DST, dst}};

    bool with_workspace = aprop_kind == prop_kind::forward_training;
    if (with_workspace) {
      dst.init_workspace(pd.workspace_desc());
      args.insert({DNNL_ARG_WORKSPACE, dst.get_workspace()});
    }

    super(pd).execute(stream::default_stream(), args);
  }
};

struct lrn_backward : public dnnl::lrn_backward {
  static void compute(const tensor& x,
                      const tensor& grady,
                      const tensor& y,
                      tensor& gradx,
                      tensor::dim local_size,
                      float alpha,
                      float beta,
                      float k = 1.0,
                      algorithm aalgorithm = algorithm::lrn_across_channels) {
  }
};

}  // namespace ideep

#endif