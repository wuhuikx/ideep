#ifndef IDEEP_OPERATORS_CONCAT_HPP
#define IDEEP_OPERATORS_CONCAT_HPP

namespace ideep {

struct concat : public dnnl::concat {
  static void compute(std::vector<tensor>& inputs, int axis, tensor& output) {
  }

  static std::vector<int32_t> compute(std::vector<tensor>& inputs, int axis, bool add_axis, tensor& dst) {
  }
};

}  // namespace ideep

#endif