#ifndef IDEEP_KERNELS_COMMON_HPP
#define IDEEP_KERNELS_COMMON_HPP

#include "../abstract_types.hpp"
#include "../tensor.hpp"
#include "../utils.hpp"

namespace ideep {

using tdims_t = tensor::dims;
using tdim_t = tensor::dim;
using attr_t = dnnl::primitive_attr;
using post_ops = dnnl::post_ops;

}  // namespace ideep

#endif