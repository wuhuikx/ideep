#ifndef IDEEP_KERNELS_DIRECT_COPY_HPP
#define IDEEP_KERNELS_DIRECT_COPY_HPP

namespace ideep {

// XPZ: might remove this op
struct direct_copy {
  static void compute(const tensor& src, tensor& dst) {
    dst.reinit_if_necessary(src.get_desc());
    src.reorder_to(dst);
  }
};

}  // namespace ideep

#endif