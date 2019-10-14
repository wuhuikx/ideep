#ifndef IDEEP_KERNELS_BINARY_HPP
#define IDEEP_KERNELS_BINARY_HPP

#include "common.hpp"

namespace ideep {

struct binary : public dnnl::binary {

  typedef dnnl::binary super;

  static void compute(const tensor& src0,
                      const tensor& src1,
                      tensor& dst,
                      algorithm aalgorithm,
                      const engine& aengine = engine::cpu_engine()) {
    auto src0_desc = src0.get_desc();
    auto src1_desc = src1.get_desc();

    dst.reinit_if_necessary(src0_desc);
    auto dst_desc = dst.get_desc();

    auto pd = primitive_desc(
        {aalgorithm, src0_desc, src1_desc, dst_desc}, aengine);

    super(pd).execute(stream::default_stream(),
                      {{DNNL_ARG_SRC_0, src0},
                       {DNNL_ARG_SRC_1, src1},
                       {DNNL_ARG_DST, dst}});
  }
};

}  // namespace ideep

#endif