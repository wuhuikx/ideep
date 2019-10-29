#ifndef IDEEP_TENSOR_HPP
#define IDEEP_TENSOR_HPP

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>

#include "abstract_types.hpp"
#include "allocators.hpp"
#include "attributes.hpp"
#include "utils.hpp"

namespace ideep {

class tensor : public dnnl::memory {
 public:
  using dims = dnnl::memory::dims;
  using dim_t = dims::value_type;
  using dims_t = dnnl_dims_t;
  using format_kind_t = dnnl_format_kind_t;
  using blocking_desc_t = dnnl_blocking_desc_t;

  struct desc : public dnnl::memory::desc {
    desc() : dnnl::memory::desc(){};

    desc(const dnnl_memory_desc_t &adata) : dnnl::memory::desc(adata){};

    desc(const dims &adims, data_type adata_type, format_tag aformat_tag)
        : dnnl::memory::desc(adims, adata_type, aformat_tag) {}

    desc(const dims &adims, data_type adata_type)
        : dnnl::memory::desc(adims, adata_type, get_default_format(adims)) {}

    desc(const dims &adims, data_type adata_type, const dims &astrides)
        : dnnl::memory::desc(adims, adata_type, astrides) {}

    /// Returns number of dimensions
    inline int ndims() const { return data.ndims; }

    const dims_t &padded_dims() const { return data.padded_dims; }

    const dims_t &padded_offsets() const { return data.padded_offsets; }

    dim_t offset0() const { return data.offset0; }

    inline format_kind_t format_kind() const { return data.format_kind; }

    /// Return size of specified dimension
    inline dim_t get_dim(int index) const {
      if (index < 0 || index >= ndims()) return static_cast<dim_t>(0);
      return data.dims[index];
    }

    /// Returns dimension vector
    inline dims get_dims() const {
      return dims(data.dims, data.dims + data.ndims);
    }

    /// Returns descriptor data type
    inline data_type get_data_type() const {
      return static_cast<data_type>(data.data_type);
    }

    /** returns true if memory descriptor is zero */
    bool is_zero() const { return ndims() == 0; }

    /** returns the number of elements including padding if \param with_padding
     * is true, and the number of data elements otherwise */
    inline dim_t nelems(bool with_padding = false) const {
      if (is_zero()) return 0;
      auto dims = with_padding ? data.padded_dims : data.dims;
      return std::accumulate(dims, dims + data.ndims, 1,
                             std::multiplies<dim_t>());
    }

    /** returns true if memory descriptor contains zero as one of its dim */
    bool has_zero_dim() const { return nelems() == 0; }

    const blocking_desc_t &blocking_desc() const {
      IDEEP_ENFORCE(is_blocking_desc(),
                    "Cannot get blocking desc on a non-blocking desc");
      return data.format_desc.blocking;
    }

    bool is_blocking_desc() const { return format_kind() == dnnl_blocked; }

    bool is_wino_desc() const { return format_kind() == dnnl_format_kind_wino; }

    bool is_rnn_packed_desc() const {
      return format_kind() == dnnl_format_kind_rnn_packed;
    }

    inline bool is_plain() {
      return is_blocking_desc() && blocking_desc().inner_nblks == 0;
    };

    inline bool is_nhwc() {
      if (!is_plain() || ndims() != 4) return false;
      const auto &dims = data.dims;
      const auto &strides = data.format_desc.blocking.strides;
      return strides[0] == dims[1] * dims[2] * dims[3]  // stride_n = c * h * w
             && strides[1] == 1                         // stride_c = 1
             && strides[2] == dims[3] * dims[1]         // stride_h = w * c
             && strides[3] == dims[3];                  // stride_w = c
    };

    inline bool is_nchw() {
      if (!is_plain() || ndims() != 4) return false;
      const auto &dims = data.dims;
      const auto &strides = data.format_desc.blocking.strides;
      return strides[0] == dims[1] * dims[2] * dims[3]  // stride_n = c * h * w
             && strides[1] == dims[2] * dims[3]         // stride_c = 1
             && strides[2] == dims[3]                   // stride_h = w * c
             && strides[3] == 1;                        // stride_w = c
    };

    /** returns true if data is dense in memory */
    bool is_dense(bool with_padding = false) const {
      if (format_kind() == dnnl_format_kind_undef ||
          format_kind() == dnnl_format_kind_any)
        return false;

      auto type_to_size = [](data_type data_type) {
        switch (data_type) {
          case dnnl::memory::data_type::f16:
          case dnnl::memory::data_type::bf16:
            return 2;
          case dnnl::memory::data_type::f32:
          case dnnl::memory::data_type::s32:
            return 4;
          case dnnl::memory::data_type::s8:
          case dnnl::memory::data_type::u8:
            return 1;
          case dnnl::memory::data_type::undef:
          default:
            IDEEP_ENFORCE(0, "unknown data type");
        }
      };

      return nelems(with_padding) * type_to_size(get_data_type()) == get_size();
    }

    /** returns physical offset by logical one. logical offset is represented by
     * an array \param pos. if \param is_pos_padded is true \param pos
     * represents the position in already padded area */
    dim_t off_v(const dims_t pos, bool is_pos_padded = false) const {
      const blocking_desc_t &blk = blocking_desc();

      dims_t pos_copy = {0};
      for (int d = 0; d < ndims(); ++d)
        pos_copy[d] = pos[d] + (is_pos_padded ? 0 : padded_offsets()[d]);

      dim_t phys_offset = offset0();

      if (blk.inner_nblks > 0) {
        dim_t blk_stride = 1;
        for (int iblk = blk.inner_nblks - 1; iblk >= 0; --iblk) {
          const int d = blk.inner_idxs[iblk];

          dim_t p;
          /* switch to faster 32-bit division when possible.
           * inner blocks always fit 32-bit. */
          if (pos_copy[d] <= INT32_MAX) {
            p = (int32_t)pos_copy[d] % (int32_t)blk.inner_blks[iblk];
            pos_copy[d] = (int32_t)pos_copy[d] / (int32_t)blk.inner_blks[iblk];
          } else {
            p = pos_copy[d] % blk.inner_blks[iblk];
            pos_copy[d] /= blk.inner_blks[iblk];
          }

          phys_offset += p * blk_stride;

          blk_stride *= blk.inner_blks[iblk];
        }
      }

      for (int d = 0; d < ndims(); ++d) {
        const dim_t p = pos_copy[d];
        phys_offset += p * blk.strides[d];
      }

      return phys_offset;
    }

    /** returns physical offset by logical one. logical offset is represented by
     * a scalar \param l_offset. if \param is_pos_padded is true, \param
     * l_offset represents logical offset in already padded area */
    dim_t off_l(dim_t l_offset, bool is_pos_padded = false) const {
      dims_t pos;
      for (int rd = 0; rd < ndims(); ++rd) {
        const int d = ndims() - 1 - rd;
        const dim_t cur_dim = is_pos_padded ? padded_dims()[d] : dims()[d];
        /* switch to faster 32-bit division when possible. */
        if (l_offset <= INT32_MAX && cur_dim <= INT32_MAX) {
          pos[d] = (int32_t)l_offset % (int32_t)cur_dim;
          l_offset = (int32_t)l_offset / (int32_t)cur_dim;
        } else {
          pos[d] = l_offset % cur_dim;
          l_offset /= cur_dim;
        }
      }
      return off_v(pos, is_pos_padded);
    }

    bool is_public_format() const {
      return is_blocking_desc() && blocking_desc().inner_nblks == 0;
    }

    desc to_format_any() const {
      return desc(get_dims(), get_data_type(), format_tag::any);
    }

    desc clone() const {
      dnnl_memory_desc_t clone_data;
      memcpy(&clone_data, &data, sizeof(dnnl_memory_desc_t));
      return desc(clone_data);
    }

    desc to_type() const {
      return desc(get_dims(), get_data_type(), format_tag::any);
    }

    desc to_grouped(int groups) const {
      auto dims = get_dims();
      dims.insert(dims.begin(), groups);
      dims[1] /= groups;
      return desc(dims, get_data_type(), format_tag::goihw);
    }
  };

  desc get_desc() const {
    const dnnl_memory_desc_t *cdesc;
    error::wrap_c_api(dnnl_memory_get_memory_desc(get(), &cdesc),
                      "could not get memory descriptor from a memory");
    return desc(*cdesc);
  }

  desc dup_desc() const {
    return get_desc().clone();
  }

  // For backward compatibility. Will be deprecated.
  desc dup_descriptor() const {
    return dup_desc();
  }

  // For backward compatibility. Will be deprecated.
  desc get_descriptor() const { return get_desc(); }

  // Constructs an tensor with no buffer and zero memory description
  tensor()
      : dnnl::memory({dims(0), data_type::undef, format_tag::undef},
                     engine::cpu_engine(), nullptr) {}

  /// Constructs a tensor.
  ///
  /// @param desc tensor descriptor.
  /// @param aengine Engine.
  /// @param ahandle handle.
  tensor(const dnnl::memory::desc &adesc, void *ahandle,
         const dnnl::engine &aengine = engine::cpu_engine()) {
    reinit(adesc, ahandle, aengine);
  }

  /// Constructs a memory.
  ///
  /// @param desc tensor descriptor.
  /// @param aengine Engine.
  tensor(const dnnl::memory::desc &adesc,
         const dnnl::engine &aengine = engine::cpu_engine()) {
    reinit(adesc, aengine);
  }

  // XPZ: sugar: unpack desc to top level to avoid nested implicit conversion

  // format_tag, buffer
  tensor(const dims &adims, data_type adata_type, format_tag aformat_tag,
         void *ahandle, const dnnl::engine &aengine = engine::cpu_engine()) {
    reinit(adims, adata_type, aformat_tag, ahandle, aengine);
  }

  // format_tag, no buffer
  tensor(const dims &adims, data_type adata_type, format_tag aformat_tag,
         const dnnl::engine &aengine = engine::cpu_engine()) {
    reinit(adims, adata_type, aformat_tag, aengine);
  }

  // no format_tag, buffer
  tensor(const dims &adims, data_type adata_type, void *ahandle,
         const dnnl::engine &aengine = engine::cpu_engine()) {
    reinit(adims, adata_type, ahandle, aengine);
  }

  // no format_tag, no buffer
  tensor(const dims &adims, data_type adata_type,
         const dnnl::engine &aengine = engine::cpu_engine()) {
    reinit(adims, adata_type, aengine);
  }

  /// Function that refill tensor with new description. Specifiy extra buffer.
  void reinit(const dnnl::memory::desc &adesc, void *ahandle,
              const dnnl::engine &aengine = engine::cpu_engine()) {
    buffer_.reset();
    workspace_.reset();
    scale_.reset();

    dnnl_memory_t result;
    error::wrap_c_api(
        dnnl_memory_create(&result, &adesc.data, aengine.get(), ahandle),
        "could not create a memory");
    reset(result);
  }

  /// Function that refill tensor with new description or buffer
  void reinit(const dnnl::memory::desc &adesc,
              const dnnl::engine &aengine = engine::cpu_engine()) {
    // XPZ: TODO: use engine allocator
    buffer_.reset(utils::allocator::malloc(adesc.get_size()),
                  utils::allocator::free);
    workspace_.reset();
    scale_.reset();

    dnnl_memory_t result;
    error::wrap_c_api(
        dnnl_memory_create(&result, &adesc.data, aengine.get(), buffer_.get()),
        "could not create a memory");
    reset(result);
  }

  // format_tag, buffer
  void reinit(const dims &adims, data_type adata_type, format_tag aformat_tag,
              void *ahandle,
              const dnnl::engine &aengine = engine::cpu_engine()) {
    reinit({adims, adata_type, aformat_tag}, ahandle, aengine);
  }

  // format_tag, no buffer
  void reinit(const dims &adims, data_type adata_type, format_tag aformat_tag,
              const dnnl::engine &aengine = engine::cpu_engine()) {
    reinit({adims, adata_type, aformat_tag}, aengine);
  }

  // no format_tag, buffer
  void reinit(const dims &adims, data_type adata_type, void *ahandle,
              const dnnl::engine &aengine = engine::cpu_engine()) {
    reinit({adims, adata_type, get_default_format(adims)}, ahandle, aengine);
  }

  // no format_tag, no buffer
  void reinit(const dims &adims, data_type adata_type,
              const dnnl::engine &aengine = engine::cpu_engine()) {
    reinit({adims, adata_type, get_default_format(adims)}, aengine);
  }

  void reinit_like(const tensor &t) {
    reinit(t.get_desc(), t.get_engine());
  }

  void reinit_like(const tensor &t, void *ahandle) {
    reinit(t.get_desc(), ahandle, t.get_engine());
  }

  void reinit_if_necessary(const dnnl::memory::desc &expected_desc) {
    if (expected_desc != get_desc() || !get_data_handle()) {
      reinit(expected_desc, get_engine());
    }
  }

  /// Copy constructor
  tensor(const tensor &t) : dnnl::memory(t) {
    // std::cout << "tensor copy ctor" << std::endl;
    buffer_ = t.buffer_;
    scale_ = t.scale_;
    workspace_ = t.workspace_;
  }

  /// Move constructor
  tensor(tensor &&t) : dnnl::memory(std::move(t)) {
    // std::cout << "tensor move ctor" << std::endl;
    buffer_ = std::move(t.buffer_);
    scale_ = std::move(t.scale_);
    workspace_ = std::move(t.workspace_);
  }

  /// Assignment operator
  tensor &operator=(const tensor &t) {
    // std::cout << "tensor copy assign" << std::endl;
    dnnl::memory::operator=(t);
    buffer_ = t.buffer_;
    scale_ = t.scale_;
    workspace_ = t.workspace_;
    return *this;
  }

  /// Move assignment operator
  tensor &operator=(tensor &&t) {
    // std::cout << "tensor move assign" << std::endl;
    dnnl::memory::operator=(std::move(t));
    buffer_ = std::move(t.buffer_);
    scale_ = std::move(t.scale_);
    workspace_ = std::move(t.workspace_);
    return *this;
  }

  /// Returns number of dimensions
  inline int ndims() const { return get_desc().ndims(); }

  /// Return size of specified dimension
  inline dim_t get_dim(int index) const { return get_desc().get_dim(index); }

  /// Returns dimension vector
  inline dims get_dims() const { return get_desc().get_dims(); }

  /// Return element number of the param.
  /// The number is the meaning values for a tensor, instead of whole buffer.
  /// It is the number without counting in paddings.
  inline dim_t get_nelems() const { return get_desc().nelems(); }

  /// Returns descriptor data type
  inline data_type get_data_type() const { return get_desc().get_data_type(); }

  inline size_t get_size() const { return get_desc().get_size(); }

  /// Return whether the tensor is empty
  inline bool is_empty() const {
    return get_desc().is_zero() && get_data_handle() == nullptr;
  }

  inline bool is_public_format() const {
    return get_desc().is_public_format();
  }

  inline static format_tag get_default_format(const dims &adims) {
    switch (adims.size()) {
      case 1:
        return format_tag::x;
      case 2:
        return format_tag::nc;
      case 3:
        return format_tag::ncw;
      case 4:
        return format_tag::nchw;
      case 5:
        return format_tag::ncdhw;
      default:
        return format_tag::undef;
    }
  }

  // For debugging only
  inline void peek_first_four_elems() const {
    auto data = reinterpret_cast<float *>(get_data_handle());
    std::cout << data[0] << std::endl;
    std::cout << data[1] << std::endl;
    std::cout << data[2] << std::endl;
    std::cout << data[3] << std::endl;
    std::cout << std::endl;
  }

  tensor reorder_if_necessary(const dnnl::memory::desc &expected_desc) const {
    if (expected_desc == get_desc()) {
      return *this;
    } else {
      tensor dst{expected_desc};
      this->reorder_to(dst);
      return dst;
    }
  }

  // no data copy
  tensor make_tmp_grouped_weights_if_necessary(int groups) const {
    if (groups > 1) {
      // XPZ: TODO: any other check?
      auto grouped_desc = get_desc().to_grouped(groups);
      auto this_copy = *this;
      this_copy.replace_desc(grouped_desc);
      return this_copy;
    } else {
      return *this;
    }
  }

  /// Recreate a param with completely different content from old one
  /// but reuse the param shell. Notice that after resize, its format
  /// is undefined
  /// XPZ: For caffe2
  void resize(const dims &adims, data_type adata_type) {
    auto new_desc = get_desc().reshape(adims);
    reinit(new_desc, get_engine());
  }

  /// Return an new tensor with new shape
  tensor &reshape(const dims &adims) {
    if (!has_same_volume(adims)) {
      throw error(dnnl_runtime_error, "reshape to incompatible shape");
    }
    if (adims != get_dims()) {
      if (!is_public_format()) {
        throw error(dnnl_runtime_error, "XPZ: TODO: reorder");
      }
      // XPZ: TODO: keep format structure
      replace_desc({adims, get_data_type()});
    }
    return *this;
  }

  inline void reorder_from(const tensor &src) {
    // https://github.com/intel/mkl-dnn/issues/571
    dnnl::reorder(src, *this)
        .execute(stream::default_stream(), const_cast<tensor &>(src), *this);
  }

  inline void reorder_to(tensor &dst, const attr_t &aattr = attr_t()) const {
    auto pd = dnnl::reorder::primitive_desc(*this, dst, aattr);
    dnnl::reorder(pd).execute(stream::default_stream(),
                              const_cast<tensor &>(*this), dst);
  }

  /// Convert the tensor to public format, and f32 data type by default
  inline tensor to_public(void *array = nullptr, bool scale_out = true) const {
    tensor dst{get_dims(), get_data_type(), array};
    this->reorder_to(dst);
    return dst;
  }

  /// Fill the tensor with a src tensor
  void feed_from(const tensor &src) { this->reorder_from(src); }

  // For backward compatibility. Will be deprecated.
  void feed_from(const dims &adims, data_type adata_type, const void *array) {
    feed_from({adims, adata_type, const_cast<void *>(array)});
  }

  void init_workspace(desc &desc) {
    auto workspace = new tensor(desc, get_engine());
    workspace_.reset(workspace);
  }

  /// Return extra packed tensor
  tensor &get_workspace() const { return *workspace_; }

  /// Decide wether there is an extra tensor packed in
  bool has_workspace() const { return workspace_ != nullptr; }

  /// Return the scale of this param.
  const scale_t &get_scale() const { return *scale_.get(); }

  /// Set new scale into param
  void set_scale(const scale_t &ascale) { scale_.reset(new scale_t(ascale)); }

  /// Return whether the param has a scale
  bool has_scale() const { return scale_ != nullptr && !scale_->empty(); }

  /// Need reorder if current param used by non DNNL routines.
  /// XPZ: TODO: will be removed
  inline bool need_reorder() const {
    return (!is_public_format() || get_data_type() != data_type::f32);
  }

  void transpose_from(const tensor &src, const std::vector<int> &axes = {}) {
    throw error(dnnl_runtime_error, "not implemented");
  }

 protected:
  bool has_same_volume(const dims &new_dims) const {
    auto old_dims = get_dims();
    auto volume_old = std::accumulate(old_dims.begin(), old_dims.end(), 1,
                                      std::multiplies<dim_t>());
    auto volume_new = std::accumulate(new_dims.begin(), new_dims.end(), 1,
                                      std::multiplies<dim_t>());
    return volume_old == volume_new;
  }

  /// Set a descriptor into tensor to replace the older one, keep buffer
  /// It is caller's responsibility to make sure the original buffer is large
  /// enough for specified descriptor
  void replace_desc(const desc &new_desc) {
    // Keep the original management
    auto buf = std::move(buffer_);
    auto ws = std::move(workspace_);
    auto scale = std::move(scale_);
    reinit(new_desc, get_data_handle(), get_engine());
    buffer_ = std::move(buf);
    workspace_ = std::move(ws);
    scale_ = std::move(scale);
  }

  std::shared_ptr<tensor> workspace_;
  std::shared_ptr<scale_t> scale_;
  std::shared_ptr<void> buffer_;
};

////////////////////////////////////////////////////////////////////////////////
/////////////////////////// Reference code below ///////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// /// Param class handles operands to computations' internal, it wrappers DNNL
// /// memory primitive and provides utilities to manipulate underlying object.
// /// It's also the base class of tensor, handles major tensor services.
// class param: public c_wrapper<dnnl_primitive_t> {
// public:
//   using super = c_wrapper<dnnl_primitive_t>;
//   using dims = dnnl::memory::dims;
//   using dim_t = dims::value_type;
//   using data_type = dnnl::memory::data_type;

//   /// Param descriptor class wrappers DNNL memory primitive descriptor
//   /// and provides utilities to manipulate underlying object
//   struct descriptor : public c_wrapper<dnnl_primitive_desc_t> {
//     friend class param;
//     inline static dnnl_primitive_kind_t convert_to_c(kind akind) {
//       return static_cast<dnnl_primitive_kind_t>(akind);
//     }
//     inline static dnnl_data_type_t convert_to_c(data_type adata_type) {
//       return static_cast<dnnl_data_type_t>(adata_type);
//     }
//     inline static dnnl_memory_format_t convert_to_c(format aformat) {
//       return static_cast<dnnl_memory_format_t>(aformat);
//     }
//     inline static std::vector<const_dnnl_primitive_desc_t> convert_to_c(const
//     std::vector<descriptor>& inputs) {
//       std::vector<const_dnnl_primitive_desc_t> c_api_inputs;
//       c_api_inputs.reserve(inputs.size());
//       auto convert_to_c = [](const descriptor& d) { return d.get(); };
//       std::transform(inputs.begin(), inputs.end(),
//       std::back_inserter(c_api_inputs), convert_to_c); return c_api_inputs;
//     }

//     static inline void fill_param(dnnl_memory_desc_t& md, const dims& adims,
//     data_type adata_type, format aformat) {
//       md.primitive_kind = convert_to_c(kind::memory);
//       md.ndims = static_cast<int>(adims.size());
//       std::copy(adims.begin(), adims.end(), md.dims);
//       md.data_type = convert_to_c(adata_type);
//       md.format = convert_to_c(aformat);
//     }

//     static inline void set_default_strides(dims& strides, const dims& adims,
//     const int* perm = NULL) {
//       static const int id_perm[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
//       if (perm == NULL) perm = id_perm;

//       auto ndims = adims.size();
//       strides[(unsigned)perm[ndims - 1]] = 1;
//       for (unsigned d = 1; d < ndims; ++d) {
//           const int prev_idx = perm[ndims - d];
//           const int curr_idx = perm[ndims - 1 - d];
//           strides[(unsigned)curr_idx] = adims[(unsigned)curr_idx] ==
//             0 ? 1 : strides[(unsigned)prev_idx] * std::max(1,
//             adims[(unsigned)prev_idx]);
//       }
//     }

//     static inline void fill_blocking(dnnl_memory_desc_t& md, const dims&
//     adims,
//         const dims& block_dims, const dims& stride, const dims& stride_inner)
//         {
//       dnnl_blocking_desc_t& blk = md.layout_desc.blocking;
//       std::copy(block_dims.begin(), block_dims.end(), blk.block_dims);
//       std::copy(stride.begin(), stride.end(), &blk.strides[0][0]);
//       std::copy(stride_inner.begin(), stride_inner.end(),
//       &blk.strides[1][0]); std::copy(adims.begin(), adims.end(),
//       blk.padding_dims); auto ndims = adims.size();
//       std::fill(blk.offset_padding_to_data, blk.offset_padding_to_data +
//       ndims, 0); blk.offset_padding = 0;
//     }

//   public:
//     /// Initiate a param descriptor, specifying blocking details.
//     descriptor(const dims& adims, data_type adata_type, const dims& stride,
//         const dims block_dims = dims(12, 1), const dims stride_inner =
//         dims(12, 1))
//       : c_wrapper([&adims, adata_type, &block_dims, &stride, &stride_inner] {
//       dnnl_memory_desc_t data;
//       fill_param(data, adims, adata_type, format::blocked);
//       fill_blocking(data, adims, block_dims, stride, stride_inner);

//       dnnl_primitive_desc_t result;
//       error::wrap_c_api(dnnl_memory_primitive_desc_create(
//             &result, &data, engine::cpu_engine().get()),
//           "could not initialize a memory descriptor");
//       return result;
//     }()), public_format_(format::blocked) {}

//     /// Initiate a param descriptor, using format for blocking
//     initialization. descriptor(const dims& adims, data_type adata_type,
//     format aformat)
//       :c_wrapper([&adims, adata_type, aformat]() {
//         dnnl::memory::validate_dims(adims);

//         // XXX: out of range enum might result unspecified behavior
//         dnnl_memory_desc_t data;
//         error::wrap_c_api(dnnl_memory_desc_init(
//               &data, (int)adims.size(), adims.size() == 0 ? nullptr :
//               &adims[0], convert_to_c(adata_type), convert_to_c(aformat)),
//             "could not initialize a memory descriptor");

//         dnnl_primitive_desc_t result;
//         error::wrap_c_api(dnnl_memory_primitive_desc_create(
//               &result, &data, engine::cpu_engine().get()),
//             "could not initialize a memory descriptor");
//         return result;
//       }()), public_format_(public_format(aformat)) {}

//     /// Initiate a param descriptor, assume nature format.
//     descriptor(const dims& adims, data_type adata_type)
//       : descriptor(adims, adata_type,
//       engine::default_format((int)adims.size())) {}

//     /// Initiate a descriptor from primitive_desc_t struct
//     descriptor(dnnl_primitive_desc_t adesc, format aformat)
//       : c_wrapper(adesc), public_format_(aformat) {}

//     /// Initiate a descriptor from primitive_desc_t struct
//     descriptor(dnnl_primitive_desc_t adesc) : descriptor(adesc,
//       public_format(convert_to_public_format(dnnl_primitive_desc_query_memory_d(adesc)->format)))
//       {}

//     /// Initiate a descriptor from primitive_desc_t struct
//     descriptor(const_dnnl_primitive_desc_t adesc, format aformat)
//       : c_wrapper(const_cast<dnnl_primitive_desc_t>(adesc), true),
//       public_format_(aformat) {}

//     /// Initiate a descriptor from primitive_desc_t struct
//     descriptor(const_dnnl_primitive_desc_t adesc)
//       : descriptor(adesc, public_format(convert_to_public_format(
//           dnnl_primitive_desc_query_memory_d(adesc)->format))) {}

//     /// Initiate a descriptor from another, share resource
//     descriptor(const descriptor& adesc)
//       : c_wrapper(adesc), public_format_ (adesc.public_format_) {}

//     /// Empty decriptor constructor
//     descriptor():descriptor(dims(0), data_type::data_undef,
//     format::format_undef) {}

//     /// Share a descriptor from another, share resource
//     descriptor& operator=(const descriptor& adesc) {
//       c_wrapper::operator=(adesc);
//       public_format_ = adesc.public_format_;
//       return *this;
//     }

//     inline void to_bytes(utils::bytestring& bytes) const {
//       auto* desc = get_dnnl_memory_desc_t();
//       utils::to_bytes(bytes, desc->data_type);
//       utils::to_bytes(bytes, desc->format);
//       for (int i = 0; i < desc->ndims; i++) {
//         utils::to_bytes(bytes, desc->dims[i]);
//       }

//       if (desc->format == format::blocked) {
//         for (int i = 0; i < desc->ndims; i++) {
//           utils::to_bytes(bytes,
//           static_cast<uint64_t>(desc->layout_desc.blocking.strides[0][i]));
//           utils::to_bytes(bytes,
//           static_cast<uint64_t>(desc->layout_desc.blocking.strides[1][i]));
//           utils::to_bytes(bytes, desc->layout_desc.blocking.block_dims[i]);
//           utils::to_bytes(bytes, desc->layout_desc.blocking.padding_dims[i]);
//           utils::to_bytes(bytes,
//           desc->layout_desc.blocking.offset_padding_to_data[i]);
//         }
//         utils::to_bytes(bytes,
//         static_cast<uint64_t>(desc->layout_desc.blocking.offset_padding));
//       }
//     }

//     /// Returns the number of bytes required to allocate the memory
//     /// described including the padding area.
//     inline size_t get_size() const {
//       return dnnl_memory_primitive_desc_get_size(get());
//     }

//     /// Returns number of dimensions
//     inline int ndims() const {
//       return get_dnnl_memory_desc_t()->ndims;
//     }

//   /// Return size of specified dimension
//     inline dim_t get_dim(int index) const {
//       if (index < 0 || index >= ndims()) return static_cast<dim_t>(0);
//       auto* internal = get_dnnl_memory_desc_t();
//       return internal->dims[index];
//     }

//     /// Returns dimension vector
//     inline dims get_dims() const {
//       auto* internal = get_dnnl_memory_desc_t();
//       return dims(internal->dims, &internal->dims[internal->ndims]);
//     }

//     /// Returns descriptor data type
//     inline data_type get_data_type() const {
//       auto* internal = get_dnnl_memory_desc_t();
//       return static_cast<data_type>(internal->data_type);
//     }

//     /// Returns C API dnnl_memory_desc_t structure which had same
//     /// dimension and data type but without format constrain.
//     dnnl_memory_desc_t format_any() const {
//       dnnl_memory_desc_t any;
//       const dnnl_memory_desc_t* origin = get_dnnl_memory_desc_t();

//       error::wrap_c_api(dnnl_memory_desc_init(
//             &any, origin->ndims, origin->dims, origin->data_type,
//             convert_to_c(format::any)),
//           "could not initialize a memory descriptor");
//       return any;
//     }

//     /// Returns a new descriptor which had same dimension and data type
//     /// but different public format.
//     /// Format protocol:
//     /// pre-condition. 4-dimension only
//     /// 1. (format_undef, nchw) for all unknown format creation
//     /// 2. (format_undef, <internel>) compatible with all public
//     correspondent descriptor format_to(format expected) const {
//       dnnl_memory_desc_t adesc;
//       const dnnl_memory_desc_t* origin = get_dnnl_memory_desc_t();
//       auto aformat = static_cast<format>(origin->format);

//       if (public_format_ == format::format_undef) {
//         if (public_format(aformat) != format::format_undef) {
//           aformat = expected;
//         }
//       } else {
//         if (format_compatible_with(expected))
//           aformat = expected;
//         else
//           throw error(dnnl_runtime_error, "format_to errors");
//       }

//       error::wrap_c_api(dnnl_memory_desc_init(
//             &adesc, origin->ndims, origin->dims, origin->data_type,
//             convert_to_c(aformat)),
//           "could not initialize a memory descriptor");

//       dnnl_primitive_desc_t result;
//       error::wrap_c_api(dnnl_memory_primitive_desc_create(
//             &result, &adesc, engine::cpu_engine().get()),
//           "could not initialize a memory descriptor");
//       return descriptor(result, expected);
//     }

//     /// Change format from data representation to weights, only nature
//     formats
//     /// were supported. Example: from nchw to oihw
//     descriptor as_weights_format() const {
//       switch(get_internal_format()) {
//       case format::nc:
//         return format_to(format::oi);
//       case format::ncw:
//         return format_to(format::oiw);
//       case format::nchw:
//         return format_to(format::oihw);
//       case format::nhwc:
//         return format_to(format::ihwo);
//       case format::chwn:
//         return format_to(format::hwio);
//       case format::ncdhw:
//         return format_to(format::oidhw);
//       case format::ndhwc:
//         return format_to(format::dhwio);
//       default:
//         return *this;
//       }
//     }

//     // Change format to rnn
//     descriptor as_rnn_format(bool is_weight) const {
//       switch(ndims()) {
//       case 3:
//         return {get_dims(), get_data_type(), format::tnc};
//       case 4:
//         return {get_dims(), get_data_type(), format::ldgo};
//       case 5:
//         if (is_weight) {
//           return {get_dims(), get_data_type(), format::ldigo};
//         } else {
//           return {get_dims(), get_data_type(), format::ldsnc};
//         }
//       default:
//         return *this;
//       }
//     }

//     bool is_shape_compatible(const dims& next) const {
//       auto origin = get_dnnl_memory_desc_t();
//       auto volume_old = std::accumulate(
//           origin->dims, &origin->dims[origin->ndims], 1,
//           std::multiplies<int>());
//       auto volume_new = std::accumulate(
//           next.begin(), next.end(), 1, std::multiplies<dims::value_type>());
//       return volume_old == volume_new;
//     }

//     descriptor reshape(const dims& adims) {
//       if (!is_shape_compatible(adims)) {
//         throw error(dnnl_runtime_error, "reshape to incompatible shape");
//       }
//       const dnnl_memory_desc_t* origin = get_dnnl_memory_desc_t();
//       return descriptor(adims, static_cast<data_type>(origin->data_type));
//     }

//     /// Returns C API dnnl_memory_desc_t structure
//     const dnnl_memory_desc_t* get_dnnl_memory_desc_t() const {
//       return dnnl_primitive_desc_query_memory_d(get());
//     }

//     inline bool operator ==(const descriptor& other) const {
//       return dnnl_memory_primitive_desc_equal(get(), other.get());
//     }

//     inline bool operator !=(const descriptor& other) const {
//       return !operator==(other);
//     }

//     /// Return format generated by DNNL
//     // XXX: format might be out of range.
//     format get_internal_format() const {
//       return static_cast<format>(get_dnnl_memory_desc_t()->format);
//     }

//     static inline format convert_to_public_format(const dnnl_memory_format_t
//     mformat) {
//       format ret;
//       switch(mformat) {
//       case dnnl_x:
//         ret = format::x;
//         break;
//       case dnnl_oi:
//       case dnnl_io:
//         ret = format::oi;
//         break;
//       case dnnl_nc:
//         ret = format::nc;
//         break;
//       case dnnl_ncw:
//       case dnnl_nwc:
//         ret = format::ncw;
//         break;
//       case dnnl_nhwc:
//         ret = format::nhwc;
//         break;
//       case dnnl_nchw:
//       case dnnl_chwn:
//       case dnnl_nChw4c:
//       case dnnl_nChw8c:
//       case dnnl_nChw16c:
//         ret = format::nchw;
//         break;
//       case dnnl_ncdhw:
//       case dnnl_ndhwc:
//       case dnnl_nCdhw16c:
//         ret = format::ncdhw;
//         break;
//       case dnnl_oihw:
//       case dnnl_ihwo:
//       case dnnl_hwio:
//       case dnnl_OIhw8i8o:
//       case dnnl_OIhw16i16o:
//       case dnnl_OIhw8o8i:
//       case dnnl_OIhw16o16i:
//       case dnnl_OIhw8i16o2i:
//       case dnnl_OIhw8o16i2o:
//       case dnnl_Oihw8o:
//       case dnnl_Oihw16o:
//       case dnnl_Ohwi8o:
//       case dnnl_Ohwi16o:
//       case dnnl_OhIw16o4i:
//       case dnnl_OIhw4i16o4i:
//       case dnnl_IOhw16o16i:
//       case dnnl_OIhw4i16o4i_s8s8:
//         ret = format::oihw;
//         break;
//       case dnnl_oidhw:
//       case dnnl_dhwio:
//         ret = format::oidhw;
//         break;
//       case dnnl_goihw:
//       case dnnl_hwigo:
//       case dnnl_gOIhw8i8o:
//       case dnnl_gOIhw16i16o:
//       case dnnl_gOIhw4i16o4i:
//       case dnnl_gOIhw8i16o2i:
//       case dnnl_gOIhw8o16i2o:
//       case dnnl_gOIhw8o8i:
//       case dnnl_gOIhw16o16i:
//       case dnnl_gIOhw16o16i:
//       case dnnl_gOihw8o:
//       case dnnl_gOihw16o:
//       case dnnl_gOhwi8o:
//       case dnnl_gOhwi16o:
//       case dnnl_Goihw8g:
//       case dnnl_Goihw16g:
//       case dnnl_Goihw16g_s8s8:
//       case dnnl_gOhIw16o4i:
//       case dnnl_gOIhw2i8o4i:
//       case dnnl_gOIhw2i8o4i_s8s8:
//       case dnnl_gOIhw4o4i:
//       case dnnl_gOIhw4o4i_s8s8:
//       case dnnl_gOIhw4i4o:
//         ret = format::goihw;
//         break;
//       case dnnl_ntc:
//       case dnnl_tnc:
//         ret = format::tnc;
//         break;
//       case dnnl_ldigo:
//         ret = format::ldigo;
//         break;
//       case dnnl_ldgoi:
//         ret = format::ldgoi;
//         break;
//       case dnnl_ldgo:
//         ret = format::ldgo;
//         break;
//       case dnnl_ldsnc:
//         ret = format::ldsnc;
//         break;
//       case dnnl_rnn_packed:
//         ret = format::rnn_packed;
//         break;
//       case dnnl_blocked:
//       case dnnl_wino_fmt:
//       case dnnl_format_undef:
//         ret = format::format_undef;
//         break;
//       default:
//         // std::cout<<"Unsupported DNNL memory format: "<<mformat<<std::endl;
//         throw error(dnnl_runtime_error, "unsupported dnnl memory format!");
//       }
//       return ret;
//     }

//     // oi, nc, oihw, nchw
//     // TODO: other public compatible format, eg. iohw, nhwc.
//     static inline format public_compatible_format(const descriptor& desc) {
//       return convert_to_public_format(desc.get_dnnl_memory_desc_t()->format);
//     }

//   private:
//     // format that perceived by user
//     format public_format_;

//     /// Helper function: if aformat is public format, then returns it, else
//     /// returns format_undef.
//     static inline format public_format(format aformat) {
//       switch(aformat) {
//         // Public format
//         case format::x:
//         case format::nc:
//         case format::io:
//         case format::oi:
//         case format::ncw:
//         case format::nwc:
//         case format::oiw:
//         case format::wio:
//         case format::tnc:
//         case format::ldigo:
//         case format::ldgoi:
//         case format::ldgo:
//         case format::ldsnc:
//         case format::rnn_packed:
//         case format::nchw:
//         case format::nhwc:
//         case format::chwn:
//         case format::oihw:
//         case format::ihwo:
//         case format::hwio:
//         case format::ncdhw:
//         case format::ndhwc:
//         case format::oidhw:
//         case format::dhwio:
//         case format::goihw:
//           return aformat;
//         default:
//           return format::format_undef;
//       }
//     }

//     inline bool format_compatible_with(format aformat) const {
//       if (public_format_ == format::format_undef && public_format_ == aformat
//       ) {
//           return true;
//       } else if (aformat == tnc || aformat == ldgo || aformat == ldsnc ||
//       aformat == ldigo ||
//           aformat == ldgoi || aformat == rnn_packed) {
//           return true;
//       } else {
//         switch(public_format_) {
//         case format::nc:
//           if (aformat == oi) return true;
//           break;
//         case format::ncw:
//           if (aformat == oiw) return true;
//           break;
//         case format::nchw:
//           if (aformat == oihw) return true;
//           break;
//         case format::nhwc:
//           if (aformat == ihwo) return true;
//           break;
//         case format::chwn:
//           if (aformat == hwio) return true;
//           break;
//         case format::ncdhw:
//           if (aformat == oidhw) return true;
//           break;
//         default:
//           break;
//         }
//       }
//       return false;
//     }
//   };

//   /// View is for describing a subregion from a param
//   struct view : public c_wrapper<dnnl_primitive_desc_t> {
//     /// Create view by specifying starting coordinate and size of each
//     dimension view (const descriptor& host, const dims& volume, const dims&
//     start) {
//       dnnl_primitive_desc_t result;
//       error::wrap_c_api(dnnl_view_primitive_desc_create(
//             &result, host.get(), &volume[0], &start[0]),
//           "could not create a view primitive descriptor");

//       auto desc_closer = [] (dnnl_primitive_desc_t res) {
//         dnnl_primitive_desc_destroy(res);
//       };

//       std::unique_ptr<std::remove_pointer<dnnl_primitive_desc_t>::type,
//       decltype(desc_closer)>
//         guard(result, desc_closer);

//       dnnl_primitive_desc_t cdesc;
//       const_dnnl_primitive_desc_t const_cdesc = dnnl_primitive_desc_query_pd(
//           result, dnnl::convert_to_c(query::dst_pd), 0);
//       error::wrap_c_api(dnnl_primitive_desc_clone(&cdesc, const_cdesc),
//           "could not clone a src primititve descriptor");
//       reset(cdesc);
//     }

//     descriptor expected_dst_descriptor() const {
//       auto internal = dnnl_primitive_desc_query_memory_d(get());
//       dims adims (internal->dims, &internal->dims[internal->ndims]);
//       data_type adata_type = static_cast<data_type>(internal->data_type);
//       // Care about 3D senario
//       format inner_format = static_cast<format>(internal->format);
//       return descriptor(adims, adata_type, inner_format);
//     }
//   };

//   /// The template initialize param with a descriptor.
//   template<class alloc = utils::allocator>
//   void init(const descriptor& adesc) {
//     dnnl_primitive_t result;
//     error::wrap_c_api(dnnl_primitive_create(&result, adesc.get(), nullptr,
//     nullptr),
//         "could not create a memory primitive");

//     reset(result);
//     // TODO: lazy buffer allocation
//     scale_.reset();
//     buffer_.reset(alloc::malloc(adesc.get_size()), alloc::free);
//     set_data_handle(buffer_.get());
//     public_format_ = adesc.public_format_;
//   }

//   /// The template initialize param with a descriptor. Specifiy extra buffer.
//   void init(const descriptor& adesc, void* ahandle) {
//     dnnl_primitive_t result;
//     error::wrap_c_api(dnnl_primitive_create(&result, adesc.get(), nullptr,
//     nullptr),
//         "could not create a memory primitive");

//     reset(result);
//     scale_.reset();
//     buffer_.reset();
//     set_data_handle(ahandle);
//     public_format_ = adesc.public_format_;
//   }

//   /// Function that refill tensor with new description or buffer
//   template<class alloc = utils::allocator>
//   void reinit(const descriptor& adesc) {
//     auto curr_size = get_size();
//     auto new_size = adesc.get_size();

//     if (curr_size >= new_size && buffer_.get() == get_data_handle()) {
//       // We don't have to allocate new buffer or we don't manage the buffer
//       // either way, we don't allocate new buffer
//       // People who manage buffer provide enough space
//       scale_.reset();
//       set_descriptor(adesc);
//     } else {
//       // re-allocate new room
//       init<alloc>(adesc);
//     }
//   }

//   template<class alloc = utils::allocator>
//   void reinit_like(const param& aparam) {
//     reinit<alloc>(aparam.get_descriptor());
//   }

//   /// Empty construction
//   param() {
//     init(descriptor(), nullptr);
//   }

//   /// Constructs a param and allocating internal buffer.
//   param(const descriptor& adesc, void* ahandle) {
//     init(adesc, ahandle);
//   }

//   /// Constructs a param and allocating internal buffer.
//   param(const descriptor& adesc, void* ahandle, const scale_t& ascale) {
//     init(adesc, ahandle);
//     scale_.reset(new scale_t(ascale));
//   }

//   /// Copy constructor
//   param(const param& p) : super(p) {
//     public_format_ = p.public_format_;
//     buffer_ = p.buffer_;
//     scale_ = p.scale_;
//   }

//   /// Move constructor
//   param(param&& movable) : super(std::move(movable)) {
//     public_format_ = movable.public_format_;
//     buffer_ = std::move(movable.buffer_);
//     scale_ = std::move(movable.scale_);
//   }

//   /// Assignment operator
//   param& operator = (const param& p) {
//     super::operator = (p);
//     public_format_ = p.public_format_;
//     buffer_ = p.buffer_;
//     scale_ = p.scale_;
//     return *this;
//   }

//   /// Move assignment operator
//   param& operator = (param&& movable) {
//     super::operator = (std::move(movable));
//     public_format_ = movable.public_format_;
//     buffer_ = std::move(movable.buffer_);
//     scale_ = std::move(movable.scale_);
//     return *this;
//   }

//   /// Operator "==" override
//   bool operator ==(const param& p) {
//     return get_descriptor() == p.get_descriptor() &&
//         get_data_handle() == p.get_data_handle() ? true : false;
//   }

//   inline void to_bytes(utils::bytestring& bytes) const {
//     get_descriptor().to_bytes(bytes);
//   }

//   /// Recreate a param with completely different content from old one
//   /// but reuse the param shell. Notice that after resize, its format
//   /// is undefined
//   template<class alloc = utils::allocator>
//   void resize(const dims& adims, data_type adata_type) {
//     descriptor adesc(adims, adata_type);
//     init<alloc>(adesc);
//   }

//   /// Returns pointer to structure of primitive descriptor.
//   const_dnnl_primitive_desc_t get_dnnl_primitive_desc_t() const {
//     const_dnnl_primitive_desc_t cdesc;
//     error::wrap_c_api(dnnl_primitive_get_primitive_desc(get(), &cdesc),
//             "could not get primitive descriptor from a memory primitive");
//     return cdesc;
//   }

//   /// Return pointer to memory descriptor structure
//   const dnnl_memory_desc_t* get_dnnl_memory_desc_t() const {
//     const_dnnl_primitive_desc_t cdesc;
//     error::wrap_c_api(dnnl_primitive_get_primitive_desc(get(), &cdesc),
//         "could not get primitive descriptor from a param");
//     return dnnl_primitive_desc_query_memory_d(cdesc);
//   }

//   descriptor get_descriptor() const {
//     return descriptor(get_dnnl_primitive_desc_t(), public_format_);
//   }

//   descriptor dup_descriptor() const {
//     dnnl_primitive_desc_t clone;
//     error::wrap_c_api(dnnl_primitive_desc_clone(&clone,
//     get_dnnl_primitive_desc_t()),
//         "could not clone a primitive descriptor");
//     return descriptor(clone, public_format_);
//   }

//   /// Set a descriptor into param to replace the older one, keep buffer
//   /// It is caller's responsibility to make sure the original buffer is large
//   /// enough for specified descriptor
//   void set_descriptor(const descriptor& new_desc) {
//     // Keep the original management
//     auto buf = std::move(buffer_);
//     auto scale = std::move(scale_);
//     init(new_desc, get_data_handle());
//     buffer_ = std::move(buf);
//     scale_ = std::move(scale);
//     public_format_ = new_desc.public_format_;
//   }

//   /// Create a view from current param
//   view create_view(const dims& view_dims, const dims& offsets) const {
//     return view(get_descriptor(), view_dims, offsets);
//   }

//   /// Reture param's data type
//   inline data_type get_data_type() const {
//     const dnnl_memory_desc_t* adesc = get_dnnl_memory_desc_t();
//     return static_cast<data_type>(adesc->data_type);
//   }

//   /// Return size of specified dimension
//   inline dim_t get_dim(int index) const {
//     if (index < 0 || index >= ndims()) return static_cast<dim_t>(0);
//     const dnnl_memory_desc_t* mdesc = get_dnnl_memory_desc_t();
//     return mdesc->dims[index];
//   }

//   /// Return dimensions' size vector
//   inline dims get_dims() const {
//     const dnnl_memory_desc_t* mdesc = get_dnnl_memory_desc_t();
//     return dims (mdesc->dims, &mdesc->dims[mdesc->ndims]);
//   }

//   /// Return public format dimensions' size vector
//   inline dims get_public_format_dims() const {
//     if (public_format_ == format::iohw) {
//       return {get_dim(1), get_dim(0), get_dim(2), get_dim(3)};
//     } else if (public_format_ == format::nhwc) {
//       return {get_dim(0), get_dim(2), get_dim(3), get_dim(1)};
//     } else {
//       return get_dims();
//     }
//   }

//   /// Return number of dimensions
//   inline int ndims() const {
//     return get_dnnl_memory_desc_t()->ndims;
//   }

//   inline dims get_block_dims() const {
//     const dnnl_memory_desc_t* mdesc = get_dnnl_memory_desc_t();
//     const auto block_dims = mdesc->layout_desc.blocking.block_dims;
//     return dims (block_dims, &block_dims[mdesc->ndims]);
//   }

//   inline dims get_block_stride() const {
//     const dnnl_memory_desc_t* mdesc = get_dnnl_memory_desc_t();
//     const auto block_stride = mdesc->layout_desc.blocking.strides[0];
//     return dims (block_stride, &block_stride[mdesc->ndims]);
//   }

//   /// Return whether the tensor is empty
//   inline bool is_empty() const {
//     return ndims() == 0 && get_data_handle() == nullptr;
//   }

//   /// Return buffer size required by the param
//   inline size_t get_size() const {
//     return dnnl_memory_primitive_desc_get_size(get_dnnl_primitive_desc_t());
//   }

//   /// Return element number of the param.
//   /// The number is the meaning values for a tensor, instead of whole buffer.
//   /// It is the number without counting in paddings.
//   inline dim_t get_nelems() const {
//     const dnnl_memory_desc_t* mdesc = get_dnnl_memory_desc_t();
//     return std::accumulate(mdesc->dims, &mdesc->dims[mdesc->ndims], 1,
//     std::multiplies<dim_t>());
//   }

//   /// Returns a handle of the data contained in the param. On
//   /// the CPU engine, this is a pointer to the allocated memory.
//   inline void* get_data_handle() const {
//     void* handle;
//     error::wrap_c_api(dnnl_memory_get_data_handle(get(), &handle), "could not
//     get native handle"); return handle;
//   }

//   /// Set new buffer handle into param
//   inline void set_data_handle(void* handle) {
//     if (buffer_.get() != handle && buffer_ != nullptr) buffer_.reset();
//     error::wrap_c_api(dnnl_memory_set_data_handle(get(), handle), "could not
//     set native handle");
//   }

//   /// Return the scale of this param.
//   const scale_t& get_scale() const {
//     return *scale_.get();
//   }

//   /// Set new scale into param
//   void set_scale(const scale_t& ascale) {
//     scale_.reset(new scale_t(ascale));
//   }

//   /// Return whether the param has a scale
//   bool has_scale() const {
//     return (scale_ != nullptr) && (!scale_->empty());
//   }

//   /// Materialize a param. For specific scenario param will allocate
//   /// internal buffer and manage it. As if it created with buffers.
//   /// Materialize a materialied param cause no effect at all.
//   void materialize() {
//     if (!materialized()) {
//       auto adesc = get_descriptor();
//       buffer_.reset(utils::allocator::malloc(adesc.get_size()),
//       utils::allocator::free);
//       // set_data_handle will generate exception if malloc fail
//       set_data_handle(buffer_.get());
//     }
//   }

//   /// Materialize API used internal only, we should deal with it
//   inline bool materialized() const {
//     return (get_data_handle() != nullptr);
//   }

//   void dematerialize() {
//     if (get_data_handle() != nullptr) {
//       buffer_.reset();
//       set_data_handle(nullptr);
//     }
//   }

//   // Must go away or be private:
//   static dnnl_data_type_t convert_to_c(data_type adata_type) {
//       return static_cast<dnnl_data_type_t>(adata_type);
//   }
//   static dnnl_memory_format_t convert_to_c(format aformat) {
//       return static_cast<dnnl_memory_format_t>(aformat);
//   }

//   /// Return internal format of the param
//   inline format get_internal_format() const {
//     return static_cast<format>(get_dnnl_memory_desc_t()->format);
//   }

//   inline format get_public_format() const {
//     return public_format_;
//   }

//   inline format set_public_format(format aformat) {
//     return public_format_ = aformat;
//   }

//   /// Need reorder if current param used by non DNNL routines.
//   inline bool need_reorder() const {
//     return (!is_public_format() || get_data_type() != data_type::f32);
//   }

//   inline int canonical_axis_index(int axis_index) const {
//     IDEEP_ENFORCE((axis_index >= -ndims()) && (axis_index < ndims()),
//     "Invalid axis index"); if (axis_index < 0) {
//       return axis_index + ndims();
//     }
//     return axis_index;
//   }

//   bool is_shape_compatible(const dims& next) const {
//     const dnnl_memory_desc_t* adesc =
//     dnnl_primitive_desc_query_memory_d(get_descriptor().get()); auto origin =
//     adesc->dims; auto volume_old = std::accumulate(origin,
//     &origin[adesc->ndims], 1, std::multiplies<int>()); auto volume_new =
//     std::accumulate(next.begin(), next.end(), 1,
//     std::multiplies<dims::value_type>());

//     // More check than just volume
//     return volume_old == volume_new;
//   }

//   inline bool is_public_format() const {
//     auto desc = get_descriptor();
//     return desc.get_dnnl_memory_desc_t()->format ==
//     convert_to_c(descriptor::public_compatible_format(desc));
//   }

//   inline bool is_weights() const {
//     auto fmt = convert_to_c(get_internal_format());
//     return (fmt >= dnnl_oi && fmt < dnnl_ntc)
//       || (fmt > dnnl_ldsnc && fmt < dnnl_nCw8c)
//       || fmt > dnnl_nCdhw16c;
//   }

//   inline bool is_grouped() const {
//     return public_format_ == format::goihw;
//   }

//   static inline void group_dims(dims& adims, const int group) {
//     adims.insert(adims.begin(), group);
//     adims[1] /= group;
//   }

//   static inline int ungroup_dims(dims& adims) {
//     int group = adims[0];
//     adims[1] *= group;
//     adims.erase(adims.begin());
//     return group;
//   }

//   void make_group(int group) {
//     if (group > 1 && !is_grouped()) {
//       IDEEP_ENFORCE(is_public_format(), "can not make grouped with internal
//       format"); auto adims = get_dims(); group_dims(adims, group);
//       set_descriptor({adims, get_data_type(), format::goihw});
//     }
//   }

//   void make_ungroup() {
//     if (is_grouped()) {
//       IDEEP_ENFORCE(is_public_format(), "can not make ungrouped with internal
//       format"); auto adims = get_dims(); ungroup_dims(adims);
//       set_descriptor({adims, get_data_type(), format::oihw});
//     }
//   }

//   inline std::shared_ptr<char> get_tensor_buffer() const { return buffer_; }

//   inline void set_tensor_buffer(const std::shared_ptr<char>& buffer) {buffer_
//   = buffer;}

// protected:
//   // mirror descriptor's same information
//   format public_format_;
//   std::shared_ptr<char> buffer_;
//   std::shared_ptr<scale_t> scale_;
// };

// /// Tensor that describes data buffer and its explanation.
// /// It also integrates an optional tensor as an intemediate results, used in
// /// Pooling/LRN
// class IDEEP_EXPORT tensor : public param {
// public:
//   using param::param;
//   using attr_t = descriptor_group::attr_t;

//   struct reorder: public primitive_group,
//       public utils::computation_cache<reorder> {
//     struct reorder_desc : public descriptor_group {
//       reorder_desc(const c_wrapper<dnnl_primitive_desc_t>& input,
//           const c_wrapper<dnnl_primitive_desc_t>& output, const attr_t& attr
//           = attr_t()) {
//         dnnl_primitive_desc_t result;
//         error::wrap_c_api(dnnl_reorder_primitive_desc_create_v2(
//               &result, input.get(), output.get(), attr.get()),
//             "could not create a reorder primitive reorder descriptor");
//         reset(result);
//       }
//     };

//   public:
//     reorder() = default;

//     void init(const reorder_desc& desc, const descriptor& src_desc, const
//     descriptor& dst_desc) {
//       in_.init(src_desc, nullptr);
//       out_.init(dst_desc, nullptr);

//       dnnl_primitive_at_t inputs[] = { {in_.get(), 0} };
//       const_dnnl_primitive_t outputs[] = { out_.get() };
//       create_primitive(desc, inputs, outputs);
//     }

//     reorder(const descriptor& src_desc, const descriptor& dst_desc, const
//     attr_t& attr = attr_t()) {
//       reorder_desc desc(src_desc, dst_desc, attr);
//       init(desc, src_desc, dst_desc);
//     }

//     reorder(const view& aview, const descriptor& src_desc, const descriptor&
//     dst_desc, const attr_t& attr = attr_t()) {
//       reorder_desc desc(aview, dst_desc, attr);
//       init(desc, src_desc, dst_desc);
//     }

//     reorder(const descriptor& src_desc, const view& aview, const descriptor&
//     dst_desc, const attr_t& attr = attr_t()) {
//       reorder_desc desc(src_desc, aview, attr);
//       init(desc, src_desc, dst_desc);
//     }

//     void operator() (const tensor& input, const tensor& output) {
//       IDEEP_ENFORCE(!(input.get_data_type() == data_type::s8 &&
//       output.get_data_type() == data_type::u8),
//           "Not support the reorder of s8 to u8 to avoid overflow.");
//       IDEEP_ENFORCE(input.get_descriptor() == in_.get_descriptor() &&
//       output.get_descriptor() == out_.get_descriptor(),
//           "Unmatch tensor reorder descriptor in reorder");

//       in_.set_data_handle(input.get_data_handle());
//       out_.set_data_handle(output.get_data_handle());

//       stream parallel_control = stream::default_stream();
//       primitive_group::execute(parallel_control);
//     }

//     static void compute(const dims& volume, const dims& offset, const tensor&
//     input,
//         tensor& output, const attr_t& attr = attr_t()) {
//       if (input.is_empty() || output.is_empty()) { return; }

//       key_t key;
//       check_or_create_k(key, volume, offset, input, output, attr);
//       auto view = input.create_view(volume, offset);
//       fetch_or_create_m(comp, key, view, input.get_descriptor(),
//       output.get_descriptor(), attr); comp(input, output);
//     }

//     static void compute(const tensor& input, const dims& volume, const dims&
//     offset,
//         tensor& output, const attr_t& attr = attr_t()) {
//       if (input.is_empty() || output.is_empty()) { return; }

//       key_t key;
//       check_or_create_k(key, input, volume, offset, output, attr);
//       auto view = output.create_view(volume, offset);
//       fetch_or_create_m(comp, key, input.get_descriptor(), view,
//       output.get_descriptor(), attr); comp(input, output);
//     }

//     static void compute(const tensor& input, tensor& output, const attr_t&
//     attr = attr_t()) {
//       if (input.is_empty() || output.is_empty()) { return; }

//       key_t key;
//       check_or_create_k(key, input, output, attr);
//       fetch_or_create_m(comp, key, input.get_descriptor(),
//       output.get_descriptor(), attr); comp(input, output);
//     }

//   protected:
//     param in_, out_;
//   };

//   /// Pack an extra tensor into current one, allocate buffer using specified
//   allocator. template<class alloc = utils::allocator> void init_extra(const
//   descriptor& workspace) {
//     auto twin = new tensor();
//     twin->init<alloc>(workspace);
//     twin_.reset(twin);
//   }

//   /// Pack an extra tensor into current one
//   void init_extra(const descriptor& workspace, void* handle) {
//     twin_.reset(new tensor(workspace, handle));
//   }

//   /// Pack an extra tensor into current one
//   void init_extra(const tensor& ws) {
//     twin_.reset();
//     twin_ = std::make_shared<tensor>(ws);
//   }

//   tensor() : param() {}

//   tensor(const descriptor& major, void* h_major) : param(major, h_major) {}

//   tensor(const descriptor& major, void* h_major, const scale_t& scale)
//     : param(major, h_major, scale) {}

//   tensor(const descriptor& major, void* h_major, const descriptor& workspace,
//   void* h_workspace,
//       const scale_t& scale)
//     : tensor(major, h_major, scale) {
//     init_extra(workspace, h_workspace);
//   }

//   tensor(const descriptor& major, void* h_major, const descriptor& workspace,
//   void* h_workspace)
//     : tensor(major, h_major) {
//     init_extra(workspace, h_workspace);
//   }

//   /// Copy constructor
//   tensor(const tensor& t) : param(t) {
//     twin_ = t.twin_;
//   }

//   /// Move constructor
//   tensor(tensor&& movable) : param(std::move(movable)) {
//     twin_ = std::move(movable.twin_);
//   }

//   /// Assignment operator
//   tensor& operator = (const tensor& t) {
//     param::operator = (t);
//     twin_ = t.twin_;
//     return *this;
//   }

//   /// Move assignment operator
//   tensor& operator = (tensor&& movable) {
//     param::operator = (std::move(movable));
//     twin_ = std::move(movable.twin_);
//     return *this;
//   }

//   template<class alloc = utils::allocator>
//   void init(const descriptor& adesc) {
//     param::init<alloc>(adesc);
//     twin_.reset();
//   }

//   void init(const descriptor& adesc, void* ahandle) {
//     param::init(adesc, ahandle);
//     twin_.reset();
//   }

//   template<class alloc = utils::allocator>
//   void reinit(const descriptor& adesc) {
//     param::reinit<alloc>(adesc);
//     twin_.reset();
//   }

//   template<class alloc = utils::allocator>
//   void reinit_like(const param& aparam) {
//     param::reinit<alloc>(aparam.get_descriptor());
//     twin_.reset();
//   }

//   /// Return extra packed tensor
//   tensor* get_extra() {
//     return twin_.get();
//   }

//   /// Return extra packed tensor
//   const tensor* get_extra() const {
//     return twin_.get();
//   }

//   /// Decide wether there is an extra tensor packed in
//   bool has_extra() const {
//     return twin_ != nullptr;
//   }

//   tensor as_weights() const {
//     tensor ret = *this;
//     if (!is_weights())
//       ret.set_descriptor(get_descriptor().as_weights_format());
//     return ret;
//   }

//   tensor as_rnn(bool is_weight = false) const {
//     tensor ret = *this;
//     ret.set_descriptor(get_descriptor().as_rnn_format(is_weight));
//     return ret;
//   }

//   /// Returns a handle of the data contained in the param. On
//   /// the CPU engine, this is a pointer to the allocated memory.
//   inline void* get_data_handle() const {
//     void* handle;
//     error::wrap_c_api(dnnl_memory_get_data_handle(get(), &handle), "could not
//     get native handle"); return handle;
//   }

//   /// Reshape a param, reorder might happen if its format is internal
//   template<class alloc = utils::allocator>
//   tensor& reshape(const dims& new_dims) {
//     if (!get_descriptor().is_shape_compatible(new_dims)) {
//       throw error(dnnl_runtime_error, "reshape to incompatible shape");
//     } else if (new_dims != get_dims()) {
//       if (!is_public_format()) {
//         tensor p;
//         p.init<alloc>({get_dims(), get_data_type()});
//         reorder::compute(*this, p);
//         set_data_handle(p.get_data_handle());
//         set_tensor_buffer(p.get_tensor_buffer());
//       }

//       set_descriptor({new_dims, get_data_type()});
//     }

//     return *this;
//   }

//   template<class alloc = utils::allocator>
//   inline tensor permute(const std::vector<int>& permute_axes = {}) const {
//     if (ndims() <= 1) {
//       return to_public_format<alloc>();
//     }

//     auto axes = permute_axes;
//     if (axes.empty()) {
//       axes.resize(ndims());
//       std::iota(axes.rbegin(), axes.rend(), 0);
//     } else {
//       IDEEP_ENFORCE(static_cast<int>(axes.size()) == ndims(),
//           "Axes should be size like source tensor.");
//       auto axes_sorted = axes;
//       std::sort(axes_sorted.begin(), axes_sorted.end());
//       for (auto i = 0; i < axes_sorted.size(); ++i) {
//         IDEEP_ENFORCE(static_cast<float>(axes_sorted[i]) == i,
//             "Axes should be a permutation of 0 to ndim.");
//       }
//       if (axes_sorted == axes) {
//         return to_public_format<alloc>();
//       }
//     }

//     auto src = *this;
//     if (!is_public_format()) {
//       src = to_public_format<alloc>();
//     }

//     auto src_dims = src.get_dims();
//     dims dst_dims(src_dims.size());
//     for (int i = 0; i < src_dims.size(); i++) {
//       dst_dims[i] = src_dims[axes[i]];
//     }

//     tensor dst;
//     dst.init<alloc>({dst_dims, src.get_data_type(),
//     src.get_public_format()}); auto dst_stride = dst.get_block_stride(); dims
//     stride (dst_stride.size(), 1); for (int i = stride.size() - 2; i >= 0;
//     i--) {
//       stride[axes[i]] = dst_stride[i];
//     }

//     tensor mask_dst;
//     mask_dst.init({src.get_dims(), src.get_data_type(), stride},
//     dst.get_data_handle()); reorder::compute(src, mask_dst); if
//     (src.has_scale()) {
//       dst.set_scale(src.get_scale());
//     }

//     return dst;
//   }

//   template<class alloc = utils::allocator>
//   void transpose_from(const tensor& src, const std::vector<int>& axes = {}) {
//     *this = src.permute<alloc>(axes);
//   }

//   /// Fill the tensor with a src tensor
//   inline void feed_from(const tensor& src) {
//     scale_t dst_scale, src_scale;
//     if (has_scale() && src.has_scale()) {
//       dst_scale = get_scale();
//       src_scale = src.get_scale();
//     } else if (has_scale()) {
//       dst_scale = get_scale();
//       src_scale.assign(dst_scale.size(), 1.0f);
//     } else if (src.has_scale()) {
//       src_scale = src.get_scale();
//       dst_scale.assign(src_scale.size(), 1.0f);
//     } else {
//       dst_scale = IDEEP_DEF_SCALE;
//       src_scale = IDEEP_DEF_SCALE;
//     }

//     IDEEP_ENFORCE(dst_scale.size() == src_scale.size(), "Invalid tensor
//     scales"); auto src_in = src; if (src_in.is_iohw_public_layout()) {
//       iohw_definedby_blocked(src_in);
//     } else {
//       IDEEP_ENFORCE(src_in.get_dims() == get_dims(), "Incorrect tesnor
//       dims");
//     }

//     scale_t scales(dst_scale.size());
//     for (int i = 0; i < dst_scale.size(); i++) {
//       scales[i] = dst_scale[i] / src_scale[i];
//     }
//     int mask = IDEEP_TENSOR_SCALE_MASK(src_scale.size(),
//     src_in.is_grouped()); reorder::compute(src_in, *this, {mask, scales});
//   }

//   /// Fill the tensor with parameters
//   inline void feed_from(const dims& adims, data_type adata_type, const void*
//   array) {
//     feed_from({{adims, adata_type, engine::default_format(adims.size())},
//     const_cast<void*>(array)});
//   }

//   /// Convert the tensor to public format, and f32 data type by default
//   template<class alloc = utils::allocator>
//   inline tensor to_public(void* array = nullptr, bool scale_out = true) const
//   {
//     tensor ret;
//     auto dst_dtype = scale_out ? data_type::f32 : get_data_type();
//     auto dst_format = ((public_format_ == format::format_undef) ||
//     (public_format_ == format::iohw))
//       ? engine::default_format(ndims()) : public_format_;

//     dims iohw_dims;
//     // TODO:it will be remove when deconvolution in dnnl support iohw format.
//     if (public_format_ == format::iohw) {
//       iohw_dims = get_public_format_dims();
//       if (array == nullptr)
//         ret.init<alloc>({iohw_dims, dst_dtype, format::oihw});
//       else
//         ret.init({iohw_dims, dst_dtype, format::oihw}, array);
//       iohw_definedby_blocked(ret);
//     } else {
//       if (array == nullptr)
//         ret.init<alloc>({get_dims(), dst_dtype, dst_format});
//       else
//         ret.init({get_dims(), dst_dtype, dst_format}, array);
//     }

//     if (scale_out && has_scale()) {
//       auto& src_scale = get_scale();
//       scale_t scales(src_scale.size());
//       for (int i = 0 ; i < src_scale.size(); i++) {
//         scales[i] = 1.0f / src_scale[i];
//       }
//       int mask = IDEEP_TENSOR_SCALE_MASK(src_scale.size(), is_grouped());
//       reorder::compute(*this, ret, {mask, scales});
//     } else {
//       reorder::compute(*this, ret);
//       if (has_scale()) {
//         ret.set_scale(get_scale());
//       }
//     }

//     // TODO:it will be remove when deconvolution in dnnl support iohw format.
//     if (!iohw_dims.empty()) {
//       ret.set_descriptor({iohw_dims, dst_dtype, dst_format});
//     }

//     return ret;
//   }

//   /// Convert the tensor to public format and keep original data type
//   template<class alloc = utils::allocator>
//   inline tensor to_public_format(void* array = nullptr) const {
//     return to_public<alloc>(array, false /* scale out */);
//   }

//   bool is_nchw_channel_blocking() const {
//     auto aformat = get_internal_format();
//     return aformat == static_cast<format>(dnnl_nchw)
//       || aformat == static_cast<format>(dnnl_nChw8c)
//       || aformat == static_cast<format>(dnnl_nChw16c);
//   }

//   bool is_nhwc_format() const {
//     auto aformat = get_internal_format();
//     return aformat == static_cast<format>(dnnl_nhwc);
//   }

//   bool is_iohw_public_layout() const {
//     return (get_public_format() == format::iohw && get_internal_format() !=
//     format::blocked);
//   }

//   bool is_limited_blockable() {
//     auto& blocking =
//     get_dnnl_memory_desc_t()->layout_desc.blocking.block_dims; for (auto i =
//     0; i < ndims(); i++) {
//       if (get_dim(i) < blocking[i]) continue;
//       if (get_dim(i) % blocking[i] == 0) continue;
//       return false;
//     }
//     return true;
//   }

//   // TODO:it will be remove when deconvolution in dnnl support iohw format.
//   static void iohw_definedby_blocked(tensor& atensor) {
//     IDEEP_ENFORCE(atensor.ndims() == 4, "Only support 4 dims tensor");
//     dims oihw_dims {atensor.get_dim(1), atensor.get_dim(0),
//     atensor.get_dim(2), atensor.get_dim(3)}; descriptor desc(oihw_dims,
//     atensor.get_data_type(), format::oihw);

//     auto oi_primitive_desc = desc.get_dnnl_memory_desc_t();
//     auto oi_blk = oi_primitive_desc->layout_desc.blocking;
//     oi_blk.strides[0][0] = oi_blk.strides[0][1];
//     oi_blk.strides[0][1] = oi_blk.strides[0][0] * oi_blk.padding_dims[0];

//     dims stride(oi_blk.strides[0], oi_blk.strides[0] +
//     oi_primitive_desc->ndims); dims stride_inner(oi_blk.strides[1],
//     oi_blk.strides[1] + oi_primitive_desc->ndims); dims
//     block_dims(oi_blk.block_dims, oi_blk.block_dims +
//     oi_primitive_desc->ndims);

//     descriptor io_desc(oihw_dims, atensor.get_data_type(), stride,
//     block_dims, stride_inner); atensor.set_descriptor(io_desc);
//   }

//   template<typename T>
//   inline void fill_all_elems(T val) {
//     utils::fast_memset(static_cast<T*>(get_data_handle()), val, get_size() /
//     sizeof(T));
//   }

// protected:
//   std::shared_ptr<tensor> twin_;
// };

}  // namespace ideep
#endif
