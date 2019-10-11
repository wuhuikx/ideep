#ifndef IDEEP_KERNELS_SPLITER_HPP
#define IDEEP_KERNELS_SPLITER_HPP

#include "common.hpp"

namespace ideep {

struct spliter {
  static std::vector<tensor> compute(const tensor& input, std::vector<int32_t>& axis_info, int axis, bool add_axis) {
  }
};


}  // namespace ideep

#endif