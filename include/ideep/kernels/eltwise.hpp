#ifndef IDEEP_KERNELS_ELTWISE_HPP
#define IDEEP_KERNELS_ELTWISE_HPP

#include "common.hpp"

namespace ideep {

struct eltwise_forward : public dnnl::eltwise_forward {
  static void compute(
      const tensor& src,
      tensor& dst,
      algorithm aalgorithm = algorithm::eltwise_relu,
      prop_kind aprop_kind = prop_kind::forward,
      float alpha = 0.0,
      float beta = 0.0,
      engine aengine = engine::cpu_engine()) {
    auto src_desc = src.get_desc();

    auto op_desc = dnnl::eltwise_forward::desc(
        aprop_kind, aalgorithm, src_desc, alpha, beta);

    auto pd = dnnl::eltwise_forward::primitive_desc(op_desc, aengine);

    dnnl::eltwise_forward(pd)
        .execute(stream::default_stream(), {
            {DNNL_ARG_SRC, src}, {DNNL_ARG_DST, dst}
        });
  }
};

struct eltwise_backward : public dnnl::eltwise_backward {
  // If grady and x had different format, performance is bad.
  // TODO: Seeking a single shot solution.
  static void compute(const tensor& src, const tensor& grady, tensor& gradx,
      algorithm aalgorithm = algorithm::eltwise_relu, float alpha = 0.0, float beta = 0.0) {
  }
};

}  // namespace ideep

#endif