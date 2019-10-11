#ifndef IDEEP_KERNELS_SUM_HPP
#define IDEEP_KERNELS_SUM_HPP

#include "common.hpp"

namespace ideep {

struct sum : public dnnl::sum {
  static void compute(const scale_t& scales, const std::vector<tensor>& inputs, tensor& output) {
  }
};

}  // namespace ideep

#endif