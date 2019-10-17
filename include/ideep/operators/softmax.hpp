#ifndef IDEEP_OPERATORS_SOFTMAX_HPP
#define IDEEP_OPERATORS_SOFTMAX_HPP

namespace ideep {

struct softmax_forward : public dnnl::softmax_forward {

  using super = dnnl::softmax_forward;

  static void compute(const tensor& src,
                      tensor& dst,
                      int softmax_axis,
                      prop_kind aprop_kind = prop_kind::forward,
                      const engine& aengine = engine::cpu_engine()) {
    auto src_desc = src.get_desc();
    dst.reinit_if_necessary(src_desc);

    auto pd = primitive_desc(
        {aprop_kind, src_desc, softmax_axis}, aengine);

    super(pd).execute(stream::default_stream(),
                      {{DNNL_ARG_SRC, src}, {DNNL_ARG_DST, dst}});
  }
};

struct softmax_backward : public dnnl::softmax_backward {
  static void compute(const tensor& y, const tensor& grady, tensor& gradx, int softmax_axis) {
  }
};

}  // namespace ideep

#endif