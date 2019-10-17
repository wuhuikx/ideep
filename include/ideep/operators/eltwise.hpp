#ifndef IDEEP_OPERATORS_ELTWISE_HPP
#define IDEEP_OPERATORS_ELTWISE_HPP

namespace ideep {

struct eltwise_forward : public dnnl::eltwise_forward {

  using super = dnnl::eltwise_forward;

  static void compute(const tensor& src,
                      tensor& dst,
                      algorithm aalgorithm = algorithm::eltwise_relu,
                      prop_kind aprop_kind = prop_kind::forward,
                      float alpha = 0.0,
                      float beta = 0.0,
                      const engine& aengine = engine::cpu_engine()) {
    auto src_desc = src.get_desc();
    dst.reinit_if_necessary(src_desc);

    auto pd = primitive_desc(
        {aprop_kind, aalgorithm, src_desc, alpha, beta}, aengine);

    super(pd).execute(stream::default_stream(),
                      {{DNNL_ARG_SRC, src}, {DNNL_ARG_DST, dst}});
  }
};

struct eltwise_backward : public dnnl::eltwise_backward {
  // If grady and x had different format, performance is bad.
  // TODO: Seeking a single shot solution.
  static void compute(const tensor& src,
                      const tensor& grady,
                      tensor& gradx,
                      algorithm aalgorithm = algorithm::eltwise_relu,
                      float alpha = 0.0,
                      float beta = 0.0) {}
};

}  // namespace ideep

#endif