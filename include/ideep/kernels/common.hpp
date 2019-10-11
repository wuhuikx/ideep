#ifndef IDEEP_KERNELS_COMMON_HPP
#define IDEEP_KERNELS_COMMON_HPP

#include "../abstract_types.hpp"
#include "../tensor.hpp"
#include "../utils.hpp"

namespace ideep {

using tdims_t = tensor::dims;
using tdesc_t = tensor::desc;
using tdtype_t = tensor::data_type;
using attr_t = dnnl::primitive_attr;
using computation = dnnl::primitive;
using tag = dnnl::memory::format_tag;
using post_ops = dnnl::post_ops;

}  // namespace ideep

#endif