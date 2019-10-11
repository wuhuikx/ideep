#ifndef IDEEP_KERNELS_ELTWISE_BINARY_HPP
#define IDEEP_KERNELS_ELTWISE_BINARY_HPP

#include "common.hpp"

namespace ideep {

struct eltwise_binary {
  enum eltwise_binary_op {
    ELTWISE_ADD,
    ELTWISE_MUL,
    ELTWISE_DIV,
  };
  static void compute(eltwise_binary_op op, tensor& inputA, tensor& inputB, tensor& outputC) {
  }
};

}  // namespace ideep

#endif