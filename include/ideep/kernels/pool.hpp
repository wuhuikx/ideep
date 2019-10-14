#ifndef IDEEP_KERNELS_POOL_HPP
#define IDEEP_KERNELS_POOL_HPP

#include "common.hpp"

namespace ideep {

struct pooling_forward : public dnnl::pooling_forward {

  typedef dnnl::pooling_forward super;

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