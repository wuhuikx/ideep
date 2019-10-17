#ifndef IDEEP_OPERATORS_CHANNEL_SHUFFLE_HPP
#define IDEEP_OPERATORS_CHANNEL_SHUFFLE_HPP

namespace ideep {

struct channel_shuffle_forward: public dnnl::shuffle_forward {
  static void compute(const tensor& src, tensor& dst, const int group, const int axis = 1,
      prop_kind aprop_kind = prop_kind::forward) {
  }
};

struct channel_shuffle_backward : public dnnl::shuffle_backward {
  static void compute(const tensor& grady, tensor& gradx, const int group, const int axis = 1) {
  }
};

}  // namespace ideep

#endif