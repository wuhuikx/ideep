#ifndef IDEEP_KERNELS_SOFTMAX_HPP
#define IDEEP_KERNELS_SOFTMAX_HPP

#include "common.hpp"

namespace ideep {

struct softmax_forward : public dnnl::softmax_forward {
  static void compute(const tensor& src, tensor& dst, int softmax_axis, prop_kind aprop_kind = prop_kind::forward) {
  }

};

struct softmax_backward : public dnnl::softmax_backward {
  static void compute(const tensor& y, const tensor& grady, tensor& gradx, int softmax_axis) {
  }
};

}  // namespace ideep

#endif