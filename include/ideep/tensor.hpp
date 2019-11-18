#ifndef IDEEP_TENSOR_HPP
#define IDEEP_TENSOR_HPP

#include "abstract_types.hpp"
#include "attributes.hpp"
#include "utils.hpp"

namespace ideep {

class tensor : public memory {
 public:
  using dim_t = dnnl_dim_t;
  using dims_t = dnnl_dims_t;
  using format_kind_t = dnnl_format_kind_t;
  using blocking_desc_t = dnnl_blocking_desc_t;
  using descriptor = desc; // for backward compatibility

  struct desc : public memory::desc {
    friend class tensor;

    desc() : memory::desc() {};

    desc(const memory::desc &adesc) : memory::desc(adesc.data) { init(); };

    desc(const dnnl_memory_desc_t &adata) : memory::desc(adata) { init(); };

    desc(const dims &adims, data_type adata_type, format_tag aformat_tag)
        : memory::desc(adims, adata_type, aformat_tag) { init(); }

    desc(const dims &adims, data_type adata_type)
        : memory::desc(adims, adata_type, get_default_format(adims)) { init(); }

    desc(const dims &adims, data_type adata_type, const dims &astrides)
        : memory::desc(adims, adata_type, astrides) { init(); }

    /// Returns number of dimensions
    inline int ndims() const { return data.ndims; }

    /// Return size of specified dimension
    inline dim_t get_dim(int index) const {
      if (index < 0 || index >= ndims()) return static_cast<dim_t>(0);
      return data.dims[index];
    }

    void print_desc() const {
      auto md = data;
      printf("\t\t");
      for (int i = 0; i < md.ndims; i++)
          printf("%c\t", 'A' + i);
      printf("\ndims\t\t");
      for (int i = 0; i < md.ndims; i++)
          printf("%ld\t", md.dims[i]);
      printf("\npadded_dims\t");
      for (int i = 0; i < md.ndims; i++)
          printf("%ld\t", md.padded_dims[i]);
      printf("\nstrides\t\t");
      for (int i = 0; i < md.ndims; i++)
          printf("%ld\t", md.format_desc.blocking.strides[i]);
      printf("\n\ninner_idxs\t");
      for (int i = 0; i < md.format_desc.blocking.inner_nblks; i++)
          printf("%ld\t", md.format_desc.blocking.inner_idxs[i]);
      printf("\ninner_blks\t");
      for (int i = 0; i < md.format_desc.blocking.inner_nblks; i++)
          printf("%ld\t", md.format_desc.blocking.inner_blks[i]);
      printf("\n");
    }

    /// Returns dimension vector
    inline dims get_dims() const {
      return dims(data.dims, data.dims + ndims());
    }

    /// Returns descriptor data type
    inline data_type get_data_type() const {
      return static_cast<data_type>(data.data_type);
    }

    inline dims get_strides() const {
      auto& strides = blocking_strides();
      return dims(strides, strides + ndims());
    }

    /** returns true if memory descriptor is zero */
    bool is_zero() const { return ndims() == 0; }

    /** returns the number of elements including padding if \param with_padding
     * is true, and the number of data elements otherwise */
    inline dim_t nelems(bool with_padding = false) const {
      if (is_zero()) return 0;
      auto dims = with_padding ? data.padded_dims : data.dims;
      return std::accumulate(dims, dims + ndims(), 1, std::multiplies<dim_t>());
    }

    inline bool is_plain() const {
      return is_blocking_desc() && blocking_desc().inner_nblks == 0;
    };

    inline bool is_nhwc() const {
      if (!is_plain() || ndims() != 4) return false;
      const auto &dims = data.dims;
      const auto &strides = blocking_strides();
      const auto n = 0, c = 1, h = 2, w = 3;
      return strides[n] == dims[h] * dims[w] * dims[c]
             && strides[h] == dims[w] * dims[c]
             && strides[w] == dims[c]
             && strides[c] == 1;
    };

    inline bool is_nchw() const {
      if (!is_plain() || ndims() != 4) return false;
      const auto &dims = data.dims;
      const auto &strides = blocking_strides();
      const auto n = 0, c = 1, h = 2, w = 3;
      return strides[n] == dims[c] * dims[h] * dims[w]
             && strides[c] == dims[h] * dims[w]
             && strides[h] == dims[w]
             && strides[w] == 1;
    };

    inline bool is_iohw() const {
      if (!is_plain() || ndims() != 4) return false;
      const auto &dims = data.dims;
      const auto &strides = blocking_strides();
      const auto o = 0, i = 1, h = 2, w = 3;
      return strides[i] == dims[o] * dims[h] * dims[w]
             && strides[o] == dims[h] * dims[w]
             && strides[h] == dims[w]
             && strides[w] == 1;
    };

    inline bool is_iodhw() const {
      if (!is_plain() || ndims() != 5) return false;
      const auto &dims = data.dims;
      const auto &strides = blocking_strides();
      const auto o = 0, i = 1, d = 2, h = 3, w = 4;
      return strides[i] == dims[o] * dims[d] * dims[h] * dims[w]
             && strides[o] == dims[d] * dims[h] * dims[w]
             && strides[d] == dims[h] * dims[w]
             && strides[h] == dims[w]
             && strides[w] == 1;
    };

    // workaround for issue intel/mkl-dnn#588
    bool is_4c_blocked() {
      auto& blk = blocking_desc();
      return blk.inner_nblks == 1
          && blk.inner_idxs[0] == 1 && blk.inner_blks[0] == 4;
    }

    desc to_format(format_tag aformat_tag) const {
      return desc(get_dims(), get_data_type(), aformat_tag);
    }

    desc to_format_any() const {
      return desc(get_dims(), get_data_type(), format_tag::any);
    }

    desc to_default_format() const {
      return desc(get_dims(), get_data_type());
    }

    desc clone() const {
      dnnl_memory_desc_t clone_data = data;
      return desc(clone_data);
    }

    desc to_type(data_type atype) const {
      auto ret = clone();
      ret.data.data_type = static_cast<dnnl_data_type_t>(atype);
      return ret;
    }

    desc to_grouped(int groups) const {
      auto ret = clone();

      auto &dims = ret.data.dims;
      auto &paddim = ret.data.padded_dims;
      auto &blk = ret.data.format_desc.blocking;
      auto &strides = blk.strides;

      auto second_dim_blocks = 1;

      for (size_t i = 0; i < blk.inner_nblks; i++) {
        // assume most significant dim g is not blocked
        blk.inner_idxs[i] += 1;
        if (blk.inner_idxs[i] == 1)
          second_dim_blocks *= blk.inner_blks[i];
      }

      for (size_t i = ret.data.ndims; i >= 2; i--) {
        dims[i] = dims[i - 1];
        paddim[i] = paddim[i - 1];
        strides[i] = strides[i - 1];
      }

      ret.data.ndims += 1;

      dims[1] = dims[0] / groups;
      paddim[1] = paddim[0] / groups;
      strides[1] = strides[0];

      dims[0] = groups;
      paddim[0] = groups;
      strides[0] = strides[1] * paddim[1] / second_dim_blocks;

      return ret;
    }

    // blocking structure reserving
    desc to_ungrouped() const {
      IDEEP_ENFORCE(ndims() >= 2, "Cannot ungroup a descriptor with ndims < 2");

      auto ret = clone();

      auto &dims = ret.data.dims;
      auto &paddim = ret.data.padded_dims;
      auto &blk = ret.data.format_desc.blocking;
      auto &strides = blk.strides;

      // implicitly store group info
      auto g = dims[0];
      ret.set_groups(g);

      // merge top two dims
      dims[0] *= dims[1];
      paddim[0] *= paddim[1];
      strides[0] = strides[1];

      // move each dim to the left 
      for (size_t i = 2; i < ret.data.ndims; i++) {
        dims[i - 1] = dims[i];
        paddim[i - 1] = paddim[i];
        strides[i - 1] = strides[i];
      }

      ret.data.ndims -= 1;

      for (size_t i = 0; i < blk.inner_nblks; i++) {
        // assume most significant dim g is not blocked
        blk.inner_idxs[i] -= 1;
      }

      return ret;
    }

    desc permute(const std::vector<int> &permute_axes = {}) const {
      if (ndims() <= 1) {
        return clone();
      }

      auto perms = permute_axes;
      if (perms.empty()) {
        perms.resize(ndims());
        std::iota(perms.rbegin(), perms.rend(), 0);
      } else {
        IDEEP_ENFORCE(perms.size() == ndims(),
                      "Axes should be size like source tensor.");
        auto perms_sorted = perms;
        std::sort(perms_sorted.begin(), perms_sorted.end());
        for (auto i = 0; i < perms_sorted.size(); ++i) {
          IDEEP_ENFORCE(perms_sorted[i] == i,
                        "Axes should be a permutation of 0 to ndim.");
        }
        if (perms_sorted == perms) {
          return clone();
        }
      }

      desc new_desc{};
      auto ndims = data.ndims;
      new_desc.data.ndims = data.ndims;
      new_desc.data.data_type = data.data_type;
      new_desc.data.format_kind = data.format_kind;
      new_desc.data.offset0 = data.offset0;

      // permute dims, padded_dims, padded_offsets, strides
      auto& new_dims = new_desc.data.dims;
      auto& old_dims = data.dims;
      auto& new_stride = new_desc.data.format_desc.blocking.strides;
      auto& old_stride = data.format_desc.blocking.strides;
      auto& new_paddim = new_desc.data.padded_dims;
      auto& old_paddim = data.padded_dims;
      auto& new_padoff = new_desc.data.padded_offsets;
      auto& old_padoff = data.padded_offsets;
      for (int i = 0; i < ndims; i++) {
        new_dims[i] = old_dims[perms[i]];
        new_stride[i] = old_stride[perms[i]];
        new_paddim[i] = old_paddim[perms[i]];
        new_padoff[i] = old_padoff[perms[i]];
      }

      // permute blocking
      auto inner_nblks = data.format_desc.blocking.inner_nblks;
      new_desc.data.format_desc.blocking.inner_nblks = inner_nblks;
      auto& old_inner_idxs = data.format_desc.blocking.inner_idxs;
      auto& new_inner_idxs = new_desc.data.format_desc.blocking.inner_idxs;
      auto& old_inner_blks = data.format_desc.blocking.inner_blks;
      auto& new_inner_blks = new_desc.data.format_desc.blocking.inner_blks;
      for (int i = 0; i < inner_nblks; i++) {
        new_inner_idxs[i] = perms[old_inner_idxs[i]];
        new_inner_blks[i] = old_inner_blks[i];
      }

      return new_desc;
    }

    desc transpose(dim dim0, dim dim1) const {
      std::vector<int> axes(ndims());
      std::iota(axes.begin(), axes.end(), 0);
      std::swap(axes[dim0], axes[dim1]);
      return permute(axes);
    }

    /** inits descriptor with logical dimensions adims and keep the current
     * blocking structure
     */
    desc to_dims(const dims &adims) const {
      IDEEP_ENFORCE(adims.size() == ndims(), "Rank mismatched.");

      dnnl_memory_desc_t md;
      md.ndims = data.ndims;
      md.data_type = data.data_type;

      auto& blk = blocking_desc();

      dims_t blocks;
      for (size_t i = 0; i < ndims(); i++)
        blocks[i] = 1;

      dim_t block_size = 1;
      for (int iblk = 0; iblk < blk.inner_nblks; ++iblk) {
        blocks[blk.inner_idxs[iblk]] *= blk.inner_blks[iblk];
        block_size *= blk.inner_blks[iblk];
      }

      for (int d = 0; d < ndims(); ++d) {
        md.dims[d] = adims[d];
        md.padded_dims[d] = utils::rnd_up(adims[d], blocks[d]);
        md.padded_offsets[d] = 0;
      }
      md.offset0 = 0;

      md.format_kind = dnnl_blocked;
      auto &mblk = md.format_desc.blocking;
      mblk = blk;

      for (size_t i = 0; i < ndims(); i++)
        mblk.strides[i] = blk.strides[i];

      int perm[DNNL_MAX_NDIMS];
      for (int d = 0; d < ndims(); ++d)
        perm[d] = d;

      utils::simultaneous_sort(mblk.strides, perm, ndims(),
                               [](dim_t a, dim_t b) { return b - a; });

      dim_t stride = block_size;
      for (int _d = ndims() - 1; _d >= 0; --_d) {
        const int d = perm[_d];
        md.format_desc.blocking.strides[d] = stride;
        stride *= md.padded_dims[d] / blocks[d];
      }

      md.extra = dnnl_memory_extra_desc_t {};

      return desc(md);
    }

   private:

    void init() {
      set_groups(1);
    }

    const dims_t &padded_dims() const { return data.padded_dims; }

    const dims_t &padded_offsets() const { return data.padded_offsets; }

    dim_t offset0() const { return data.offset0; }

    inline format_kind_t format_kind() const { return data.format_kind; }

    bool is_blocking_desc() const { return format_kind() == dnnl_blocked; }

    bool is_wino_desc() const { return format_kind() == dnnl_format_kind_wino; }

    bool is_rnn_packed_desc() const {
      return format_kind() == dnnl_format_kind_rnn_packed;
    }

    const blocking_desc_t &blocking_desc() const {
      IDEEP_ENFORCE(is_blocking_desc(),
                    "Cannot get blocking desc on a non-blocking desc");
      return data.format_desc.blocking;
    }

    dims_t& blocking_strides() const {
      IDEEP_ENFORCE(is_blocking_desc(),
                    "Cannot get blocking desc on a non-blocking desc");
      return const_cast<dnnl_memory_desc_t&>(data).format_desc.blocking.strides;
    }

    void set_groups(dim groups) {
      auto reserved_size = 64;
      auto offset = reserved_size / sizeof(dim) - 1;
      reinterpret_cast<dim *>(data.extra.reserved)[offset] = groups;
    }

    dim get_groups() {
      auto reserved_size = 64;
      auto offset = reserved_size / sizeof(dim) - 1;
      return reinterpret_cast<dim *>(data.extra.reserved)[offset];
    }
  };

  desc get_desc() const {
    const dnnl_memory_desc_t *cdesc;
    error::wrap_c_api(dnnl_memory_get_memory_desc(get(), &cdesc),
                      "could not get memory descriptor from a memory");
    return desc(*cdesc);
  }

  void print_desc() const {
    get_desc().print_desc();
  }

  // For backward compatibility. Will be deprecated.
  desc get_descriptor() const { return get_desc(); }

  desc dup_desc() const { return get_desc().clone(); }

  // For backward compatibility. Will be deprecated.
  desc dup_descriptor() const { return dup_desc(); }

  // Constructs an tensor with no buffer and zero memory description
  tensor() {
    reinit({dims(0), data_type::undef, format_tag::undef}, nullptr,
           engine::cpu_engine());
  }

  /// Constructs a tensor.
  ///
  /// @param desc tensor descriptor.
  /// @param aengine Engine.
  /// @param ahandle handle.
  tensor(const memory::desc &adesc, void *ahandle,
         const engine &aengine = engine::cpu_engine()) {
    reinit(adesc, ahandle, aengine);
  }

  /// Constructs a memory.
  ///
  /// @param desc tensor descriptor.
  /// @param aengine Engine.
  tensor(const memory::desc &adesc,
         const engine &aengine = engine::cpu_engine()) {
    reinit(adesc, aengine);
  }

  // XPZ: sugar: unpack desc to top level to avoid nested implicit conversion

  // format_tag, buffer
  tensor(const dims &adims, data_type adata_type, format_tag aformat_tag,
         void *ahandle, const engine &aengine = engine::cpu_engine()) {
    reinit(adims, adata_type, aformat_tag, ahandle, aengine);
  }

  // format_tag, no buffer
  tensor(const dims &adims, data_type adata_type, format_tag aformat_tag,
         const engine &aengine = engine::cpu_engine()) {
    reinit(adims, adata_type, aformat_tag, aengine);
  }

  // no format_tag, buffer
  tensor(const dims &adims, data_type adata_type, void *ahandle,
         const engine &aengine = engine::cpu_engine()) {
    reinit(adims, adata_type, ahandle, aengine);
  }

  // no format_tag, no buffer
  tensor(const dims &adims, data_type adata_type,
         const engine &aengine = engine::cpu_engine()) {
    reinit(adims, adata_type, aengine);
  }

  /// Function that refill tensor with new description. Specifiy extra buffer.
  void reinit(const memory::desc &adesc, void *ahandle,
              const engine &aengine = engine::cpu_engine()) {
    buffer_.reset();
    scale_.reset();
    eng_ = aengine;
    reset_internal(adesc, aengine, ahandle);
  }

  /// Function that refill tensor with new description or buffer
  void reinit(const memory::desc &adesc,
              const engine &aengine = engine::cpu_engine()) {
    buffer_.reset(aengine.malloc(adesc.get_size()), aengine.free);
    scale_.reset();
    eng_ = aengine;
    reset_internal(adesc, aengine, buffer_.get());
  }

  // format_tag, buffer
  void reinit(const dims &adims, data_type adata_type, format_tag aformat_tag,
              void *ahandle, const engine &aengine = engine::cpu_engine()) {
    reinit({adims, adata_type, aformat_tag}, ahandle, aengine);
  }

  // format_tag, no buffer
  void reinit(const dims &adims, data_type adata_type, format_tag aformat_tag,
              const engine &aengine = engine::cpu_engine()) {
    reinit({adims, adata_type, aformat_tag}, aengine);
  }

  // no format_tag, buffer
  void reinit(const dims &adims, data_type adata_type, void *ahandle,
              const engine &aengine = engine::cpu_engine()) {
    reinit({adims, adata_type, get_default_format(adims)}, ahandle, aengine);
  }

  // no format_tag, no buffer
  void reinit(const dims &adims, data_type adata_type,
              const engine &aengine = engine::cpu_engine()) {
    reinit({adims, adata_type, get_default_format(adims)}, aengine);
  }

  void reinit_like(const tensor &t) {
    reinit(t.get_desc(), t.get_engine());
  }

  void reinit_like(const tensor &t, void *ahandle) {
    reinit(t.get_desc(), ahandle, t.get_engine());
  }

  void reinit_if_necessary(const memory::desc &expected_desc) {
    if (expected_desc != get_desc() || !get_data_handle()) {
      reinit(expected_desc, get_engine());
    }
  }

  /// Copy constructor
  tensor(const tensor &t) : memory(t) {
    buffer_ = t.buffer_;
    scale_ = t.scale_;
    workspace_ = t.workspace_;
    eng_ = t.eng_;
  }

  /// Move constructor
  tensor(tensor &&t) : memory(std::move(t)) {
    buffer_ = std::move(t.buffer_);
    scale_ = std::move(t.scale_);
    workspace_ = std::move(t.workspace_);
    eng_ = std::move(t.eng_);
  }

  /// Assignment operator
  tensor &operator=(const tensor &t) {
    memory::operator=(t);
    buffer_ = t.buffer_;
    scale_ = t.scale_;
    workspace_ = t.workspace_;
    eng_ = t.eng_;
    return *this;
  }

  /// Move assignment operator
  tensor &operator=(tensor &&t) {
    memory::operator=(std::move(t));
    buffer_ = std::move(t.buffer_);
    scale_ = std::move(t.scale_);
    workspace_ = std::move(t.workspace_);
    eng_ = std::move(t.eng_);
    return *this;
  }

  /// Returns the engine of the tensor
  const engine& get_engine() const {
    return eng_;
  }

  /// Returns number of dimensions
  inline int ndims() const { return get_desc().ndims(); }

  /// Return size of specified dimension
  inline dim_t get_dim(int index) const { return get_desc().get_dim(index); }

  /// Returns dimension vector
  inline dims get_dims() const { return get_desc().get_dims(); }

  inline dims get_strides() const { return get_desc().get_strides(); }

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

  // "public format" has the same semantic as DNNL's "plain format"
  inline bool is_public_format() const {
    return get_desc().is_plain();
  }

  inline static format_tag get_default_format(const dims &adims) {
    switch (adims.size()) {
      case 1:
        return format_tag::a;
      case 2:
        return format_tag::ab;
      case 3:
        return format_tag::abc;
      case 4:
        return format_tag::abcd;
      case 5:
        return format_tag::abcde;
      case 6:
        return format_tag::abcdef;
      default:
        return format_tag::undef;
    }
  }

  // legacy API for caffe2
  dims get_public_format_dims() const {
    auto nchw_dims = get_dims();
    if (get_desc().is_nhwc()) {
      dims nhwc_dims(ndims());
      nhwc_dims[0] = nchw_dims[0];
      nhwc_dims[1] = nchw_dims[2];
      nhwc_dims[2] = nchw_dims[3];
      nhwc_dims[3] = nchw_dims[1];
      return nhwc_dims;
    }
    return nchw_dims;
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

  tensor reorder_if_differ_in(const memory::desc &expected_desc) const {
    if (expected_desc == get_desc()) {
      return *this;
    } else {
      tensor dst{expected_desc};
      this->reorder_to(dst);
      return dst;
    }
  }

  // workaround for issue intel/mkl-dnn#588
  desc _get_unblocked_desc_if_4c_blocked() const {
    if (get_desc().is_4c_blocked()) {
      return desc(get_dims(), get_data_type());
    } else {
      return get_desc();
    }
  }

  // no data copy
  tensor make_grouped_weights(int groups) const {
    if (groups <= 1) {
      return *this;
    } else {
      auto old_desc = get_desc();
      auto grouped_desc =
          old_desc.is_iohw() || old_desc.is_iodhw()
              ? old_desc.transpose(0, 1).to_grouped(groups).transpose(1, 2)
              : old_desc.to_grouped(groups);
      auto this_copy = *this;
      this_copy.set_desc(grouped_desc);
      return this_copy;
    }
  }

  /// Recreate a param with completely different content from old one
  /// but reuse the param shell. Notice that after resize, its format
  /// is undefined
  /// XPZ: For caffe2
  void resize(const dims &adims, data_type adata_type) {
    reinit(adims, adata_type, get_engine());
  }

  /// Return an new tensor with new shape
  tensor &reshape(const dims &adims) {
    if (!has_same_volume(adims)) {
      throw error(dnnl_runtime_error, "reshape to incompatible shape");
    }
    if (adims != get_dims()) {
      if (!is_public_format()) {
        *this = std::move(to_public());
      }
      // XPZ: TODO: keep format structure
      set_desc({adims, get_data_type()});
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
    dnnl::reorder(pd)
        .execute(stream::default_stream(), const_cast<tensor &>(*this), dst);
  }

  /// Convert the tensor to public format, and f32 data type by default
  // XPZ: TODO: scale_out ??
  tensor to_public(void *buffer = nullptr, bool scale_out = true) const {
    auto public_desc =
        is_public_format() ? get_desc() : desc(get_dims(), get_data_type());
    auto dst = buffer ? tensor(public_desc, buffer) : tensor(public_desc);

    auto groups = 1;
    if ((groups = get_desc().get_groups()) > 1) {
      auto mask_src = this->make_grouped_weights(groups);
      auto mask_dst = dst.make_grouped_weights(groups);
      mask_src.reorder_to(mask_dst);
    } else
      this->reorder_to(dst);

    return dst;
  }

  inline void to_format(format_tag aformat_tag) {
    auto dst = tensor(get_desc().to_format(aformat_tag));
    this->reorder_to(dst);
    *this = std::move(dst);
  }

  /// Fill the tensor with a src tensor
  void feed_from(const tensor &src) {
    auto groups = 1;
    if ((groups = get_desc().get_groups()) > 1 ||
        (groups = src.get_desc().get_groups()) > 1) {
      auto mask_dst = this->make_grouped_weights(groups);
      auto mask_src = src.make_grouped_weights(groups);
      mask_src.reorder_to(mask_dst);
    } else
      src.reorder_to(*this);
  }

  // For backward compatibility. Will be deprecated.
  void feed_from(const dims &adims, data_type adata_type, const void *array) {
    feed_from({adims, adata_type, const_cast<void *>(array)});
  }

  // reorder src to part of this tensor
  void insert_submemory(const tensor &src, const dims &adims,
                        const dims &offsets, const attr_t &attr = attr_t()) {
    auto view = get_desc().submemory_desc(adims, offsets);
    dnnl::reorder({src.get_engine(), src.get_desc(), get_engine(), view, attr})
        .execute(stream::default_stream(), const_cast<tensor &>(src), *this);
  }

  // reorder part of this tensor to dst
  void extract_submemory(tensor &dst, const dims &adims, const dims &offsets,
                         const attr_t &attr = attr_t()) const {
    auto view = get_desc().submemory_desc(adims, offsets);
    dnnl::reorder({get_engine(), view, dst.get_engine(), dst.get_desc(), attr})
        .execute(stream::default_stream(), const_cast<tensor &>(*this), dst);
  }

  // simple api for extract_submemory
  tensor extract_submemory(const dims &adims, const dims &offsets,
                           const attr_t &attr = attr_t()) const {
    tensor dst{adims, get_data_type(), get_engine()};
    extract_submemory(dst, adims, offsets, attr);
    return dst;
  }

  // data copy
  tensor clone() const {
    tensor dst(get_desc());
    this->reorder_to(dst);
    return dst;
  }

  void init_workspace(const memory::desc &desc) {
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

  tensor permute_(const std::vector<int> &permute_axes = {}) {
    set_desc(get_desc().permute(permute_axes));
    return *this;
  }

  tensor permute(const std::vector<int> &permute_axes = {}) const {
    return clone().permute_(permute_axes);
  }

  tensor transpose_(dim dim0, dim dim1) {
    set_desc(get_desc().transpose(dim0, dim1));
    return *this;
  }

  tensor transpose(dim dim0, dim dim1) const {
    return clone().transpose_(dim0, dim1);
  }

  // For backward compatibility. Will be deprecated
  void transpose_from(const tensor &src, const std::vector<int> &perms = {}) {
    *this = std::move(src.permute(perms));
  }

  /// Set a descriptor into tensor to replace the older one, keep buffer
  /// It is caller's responsibility to make sure the original buffer is large
  /// enough for specified descriptor
  void set_desc(const desc &new_desc) {
    // Keep the original management
    auto buf = std::move(buffer_);
    auto ws = std::move(workspace_);
    auto scale = std::move(scale_);
    reinit(new_desc, get_data_handle(), get_engine());
    buffer_ = std::move(buf);
    workspace_ = std::move(ws);
    scale_ = std::move(scale);
  }

 private:
  void reset_internal(const desc &adesc, const engine &aengine, void *ahandle) {
    dnnl_memory_t result;
    error::wrap_c_api(
        dnnl_memory_create(&result, &adesc.data, aengine.get(), ahandle),
        "could not create a memory");
    reset(result);
  }

  bool has_same_volume(const dims &new_dims) const {
    auto old_dims = get_dims();
    auto volume_old = std::accumulate(old_dims.begin(), old_dims.end(), 1,
                                      std::multiplies<dim_t>());
    auto volume_new = std::accumulate(new_dims.begin(), new_dims.end(), 1,
                                      std::multiplies<dim_t>());
    return volume_old == volume_new;
  }

  std::shared_ptr<tensor> workspace_;
  std::shared_ptr<scale_t> scale_;
  std::shared_ptr<void> buffer_;
  engine eng_;
};

}  // namespace ideep
#endif
