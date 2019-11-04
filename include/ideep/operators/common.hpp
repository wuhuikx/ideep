#ifndef IDEEP_OPERATORS_COMMON_HPP
#define IDEEP_OPERATORS_COMMON_HPP

#include "../abstract_types.hpp"
#include "../tensor.hpp"
#include "../utils.hpp"

namespace ideep {

using dims = tensor::dims;
using tdim_t = tensor::dim;
using exec_args = std::unordered_map<int, memory>;

}  // namespace ideep

#endif