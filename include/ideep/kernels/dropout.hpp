#ifndef IDEEP_KERNELS_DROPOUT_HPP
#define IDEEP_KERNELS_DROPOUT_HPP

#include "common.hpp"

namespace ideep {

struct dropout_forward {
  static void compute(const tensor& src, float ratio, tensor& dst, tensor& mask) {
  }
};

struct dropout_backward {
  static void compute(const tensor& mask, const tensor& gy, tensor& gx) {
  }
};

}  // namespace ideep

#endif