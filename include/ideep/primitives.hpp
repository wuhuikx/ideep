#ifndef IDEEP_PRIMITIVES_HPP
#define IDEEP_PRIMITIVES_HPP

#include "abstract_types.hpp"
#include "utils.hpp"

namespace ideep {

/// A group of primitive descriptors, pack related reorder descriptors
/// with computational descriptor.
struct descriptor_group: public c_wrapper<dnnl_primitive_desc_t> {
public:
  /// Post ops for fusion operations
  struct post_ops : public c_wrapper<dnnl_post_ops_t> {
  public:
    post_ops() : c_wrapper([]() {
      dnnl_post_ops_t result;
      error::wrap_c_api(dnnl_post_ops_create(&result), "could not create post operation sequence");
      return result;
    }()) {}

    int num_ops() const {
      return dnnl_post_ops_len(get());
    }

    kind op_kind(int index) const {
      IDEEP_ENFORCE(index < num_ops(), "post_ops index is out of range");
      return static_cast<kind>(dnnl_post_ops_get_kind(get(), index));
    }

    bool has_op_kind(kind op_kind) const {
      for (int i = 0; i < num_ops(); i++) {
        if (op_kind == this->op_kind(i)) {
          return true;
        }
      }
      return false;
    }

    bool non_negitive_output() const {
      auto last = num_ops() - 1;
      if (last < 0) {
        return false;
      }

      auto params = get_params(last);
      if (std::get<0>(params) != kind::eltwise
          || std::get<1>(params) <= 0.f || std::get<2>(params) != 0.f
          || std::get<3>(params) != 0.f || std::get<4>(params) != algorithm::eltwise_relu)
        return false;

      return true;
    }

    void append(kind op_kind, float scale, float alpha, float beta, algorithm alg) {
      switch(op_kind) {
        case kind::sum:
          error::wrap_c_api(dnnl_post_ops_append_sum(get(), scale), "could not append sum");
          break;
        case kind::eltwise:
          error::wrap_c_api(dnnl_post_ops_append_eltwise(
                get(), scale, convert_to_c(alg), alpha, beta), "could not append eltwise");
          break;
        default:
          error::wrap_c_api(dnnl_invalid_arguments, "Unsupport op kind");
      }
    }

    std::tuple<kind, float, float, float, algorithm> get_params(int index) const {
      dnnl_alg_kind_t c_alg = dnnl_eltwise_relu;
      float scale = 1.0, alpha = 1.0, beta = 0.0;

      auto akind = op_kind(index);
      switch(akind) {
        case kind::sum:
          error::wrap_c_api(dnnl_post_ops_get_params_sum(get(), index, &scale),
              "could not get sum params");
          break;
        case kind::eltwise:
          error::wrap_c_api(dnnl_post_ops_get_params_eltwise(
                get(), index, &scale, &c_alg, &alpha, &beta),
              "could not get eltwise params");
          break;
        default:
          error::wrap_c_api(dnnl_invalid_arguments, "could not get params");
          break;
      }

      return std::make_tuple(akind, scale, alpha, beta, static_cast<algorithm>(c_alg));
    }

    void to_bytes(utils::bytestring& bytes) const {

      for (int i = 0; i < num_ops(); i ++) {
        kind akind;
        algorithm alg;
        float scale = 1.0, alpha = 1.0, beta = 0.0;
        std::tie(akind, scale, alpha, beta, alg) = get_params(i);

        switch(akind) {
          case kind::sum:
            utils::to_bytes(bytes, akind);
            bytes.append(1, '.');
            utils::to_bytes(bytes, scale);
            break;
          case kind::eltwise:
            utils::to_bytes(bytes, akind);
            bytes.append(1, '.');
            utils::to_bytes(bytes, scale);
            bytes.append(1, '.');
            utils::to_bytes(bytes, alpha);
            bytes.append(1, '.');
            utils::to_bytes(bytes, beta);
            bytes.append(1, '.');
            utils::to_bytes(bytes, alg);
          default:
            break;
        }
      }
    }

  public:
    // Helper factory
    static post_ops sum(float scale = 1.0) {
      post_ops ret;
      ret.append(kind::sum, scale, 1.0, 0.0, algorithm::eltwise_relu);
      return ret;
    }

    static post_ops relu(float scale = 1.f, float alpha = 0.f, float beta = 0.f) {
      post_ops ret;
      ret.append(kind::eltwise, scale, alpha, beta, algorithm::eltwise_relu);
      return ret;
    }

    static post_ops residual(
        float sum_scale = 1.0, float relu_scale = 1.0, float alpha = 0.f, float beta = 0.f) {
      post_ops ret;
      ret.append(kind::sum, sum_scale, 1.0, 0.0, algorithm::eltwise_relu);
      ret.append(kind::eltwise, relu_scale, alpha, beta, algorithm::eltwise_relu);
      return ret;
    }
  };

  /// Attribute class for extra information into computations, including
  /// post operations, rounding mode, etc.
  struct attr_t : public c_wrapper<dnnl_primitive_attr_t> {
  public:
    attr_t() : c_wrapper([]() {
      dnnl_primitive_attr_t result;
      error::wrap_c_api(dnnl_primitive_attr_create(&result), "could not create a primitive attr");
      return result;
    }()) {}

    attr_t(int mask, scale_t &scales, round_mode mode = round_mode::round_nearest)
      : c_wrapper([]() {
      dnnl_primitive_attr_t result;
      error::wrap_c_api(dnnl_primitive_attr_create(&result), "could not create a primitive attr");
      return result; }()) {
      set_output_scales(mask, scales);
      set_int_output_round_mode(round_mode::round_nearest);
    }

    round_mode get_int_output_round_mode() const {
      dnnl_round_mode_t result;
      error::wrap_c_api(dnnl_primitive_attr_get_int_output_round_mode(get(), &result),
          "could not get int output round mode");
      return round_mode(result);
    }

    void set_int_output_round_mode(round_mode mode) {
      error::wrap_c_api(dnnl_primitive_attr_set_int_output_round_mode(
            get(), dnnl::convert_to_c(mode)),
          "could not set int output round mode");
    }

    std::pair<scale_t, int> get_output_scales() const {
      int count, c_mask;
      const float* c_scales;
      error::wrap_c_api(dnnl_primitive_attr_get_output_scales(get(), &count, &c_mask, &c_scales),
          "could not get int output scales");
      return std::make_pair(scale_t(c_scales, c_scales + count), c_mask);
    }

    void set_output_scales(int mask, const scale_t& scales) {
      error::wrap_c_api(dnnl_primitive_attr_set_output_scales(
            get(), (int)scales.size(), mask, &scales[0]),
          "could not set int output scales");
    }

    const post_ops get_post_ops() const {
      const_dnnl_post_ops_t c_result;
      error::wrap_c_api(dnnl_primitive_attr_get_post_ops(get(), &c_result),
          "could not get post operatoion sequence");
      post_ops result;
      result.reset(const_cast<dnnl_post_ops_t>(c_result), true);
      return result;
    }

    void set_post_ops(post_ops ops) {
      error::wrap_c_api(dnnl_primitive_attr_set_post_ops(get(), ops.get()),
          "could not set post operation sequence");
    }

    void to_bytes(utils::bytestring& bytes) const {
      get_post_ops().to_bytes(bytes);
      auto scales = get_output_scales();
      utils::to_bytes(bytes, scales.first);
      utils::to_bytes(bytes, scales.second);
    }

  public:
    // Helper factory
    static inline attr_t fuse_sum(float scale = 1.0) {
      attr_t attr;
      attr.set_post_ops(post_ops::sum(scale));
      return attr;
    }

    static inline attr_t fuse_relu(float scale = 1.0, float alpha = 0.f, float beta = 0.f) {
      attr_t attr;
      attr.set_post_ops(post_ops::relu(scale, alpha, beta));
      return attr;
    }

    static inline attr_t residual(float sum_scale = 1.0, float relu_scale = 1.0,
        float alpha = 0.f, float beta = 0.f) {
      attr_t attr;
      attr.set_post_ops(post_ops::residual(sum_scale, relu_scale, alpha, beta));
      return attr;
    }

    static inline attr_t attr_post_ops(post_ops post) {
      attr_t attr;
      attr.set_post_ops(post);
      return attr;
    }
  };

public:
  descriptor_group() = default;

  template<typename T>
  void create_primitive_desc(const T& desc, const_dnnl_primitive_desc_t hint = nullptr) {
    dnnl_primitive_desc_t result;
    error::wrap_c_api(dnnl_primitive_desc_create(
          &result, &desc, engine::cpu_engine().get(), hint),
        "could not create a primitive descriptor");
    reset(result);
  }

  template<typename T>
  void create_primitive_desc_v2(const T& desc, const attr_t attr = attr_t(), const_dnnl_primitive_desc_t hint = nullptr) {
    dnnl_primitive_desc_t result;
    error::wrap_c_api(dnnl_primitive_desc_create_v2(
          &result, &desc, attr.get(), engine::cpu_engine().get(), hint),
        "could not create a primitive descriptor");
    reset(result);
  }

  template<typename T>
  void create_primitive_desc_by_info_str_v2(std::string info_str, const T& desc,
      const attr_t attr = attr_t(), const_dnnl_primitive_desc_t hint = nullptr) {
    const char *query_info_str;
    dnnl_primitive_desc_t result;
    dnnl_primitive_desc_iterator_t iterator = nullptr;
    error::wrap_c_api(dnnl_primitive_desc_iterator_create_v2(
          &iterator, &desc, attr.get(), engine::cpu_engine().get(), hint),
            "could not create a primitive descriptor iterator");
    do {
      result = dnnl_primitive_desc_iterator_fetch(iterator);
      error::wrap_c_api(result != nullptr ? dnnl_success : dnnl_runtime_error,
              "could not fetch a primitive descriptor from the iterator");
      error::wrap_c_api(dnnl_primitive_desc_query(result, dnnl_query_impl_info_str, 0, &query_info_str),
              "could not query implementation info string");
      if (std::string(query_info_str).find(info_str) != std::string::npos) {
        reset(result);
        return;
      }
    } while(dnnl_primitive_desc_iterator_next(iterator) != dnnl_iterator_ends);
    error::wrap_c_api(dnnl_runtime_error, "could not fetch a primitive descriptor by info_str");
  }

  /// Query interface
  const_dnnl_primitive_desc_t expected_descriptor_of(query q, int index = 0) const {
      return dnnl_primitive_desc_query_pd(get(), dnnl::convert_to_c(q), index);
  }

  /// Query number of inputs
  int num_of_inputs() const {
      return dnnl_primitive_desc_query_s32(get(),
         dnnl::convert_to_c(dnnl::num_of_inputs_s32), 0);
  }

  /// Query number of outputs
  int num_of_outputs() const {
      return dnnl_primitive_desc_query_s32(get(),
         dnnl::convert_to_c(dnnl::num_of_outputs_s32), 0);
  }
};

/// A group of primitives, pack related reorder with computation.
/// It serves as a base class of computation
struct primitive_group: public c_wrapper<dnnl_primitive_t> {
public:
  primitive_group() = default;

  /// Returns the internal structure of primitive descriptor.
  const_dnnl_primitive_desc_t get_dnnl_primitive_desc_t() const {
    const_dnnl_primitive_desc_t cdesc;
    error::wrap_c_api(dnnl_primitive_get_primitive_desc(get(), &cdesc),
        "could not get primitive descriptor from a memory primitive");
    return cdesc;
  }

  /// Query interface
  const_dnnl_primitive_desc_t expected_descriptor_of(query q, int index = 0) const {
    return dnnl_primitive_desc_query_pd(get_dnnl_primitive_desc_t(), dnnl::convert_to_c(q), index);
  }

  /// Query interface
  dnnl_primitive_desc_t dup_descriptor_of(query q, int index = 0) const {
    dnnl_primitive_desc_t cdesc;
    const_dnnl_primitive_desc_t const_cdesc = dnnl_primitive_desc_query_pd(
        get_dnnl_primitive_desc_t(), dnnl::convert_to_c(q), index);
    error::wrap_c_api(dnnl_primitive_desc_clone(&cdesc, const_cdesc),
        "could not clone a src primititve descriptor");
    return cdesc;
  }

protected:
  /// Specific query interface, not valid for all computations.
  const_dnnl_primitive_desc_t expected_dst_descriptor() const {
    return expected_descriptor_of(query::dst_pd, 0);
  }

  const_dnnl_primitive_desc_t expected_workspace_descriptor() const {
    return expected_descriptor_of(query::workspace_pd, 0);
  }

  const_dnnl_primitive_desc_t expected_gradx_descriptor() const {
    return expected_descriptor_of(query::diff_src_pd, 0);
  }

  const_dnnl_primitive_desc_t expected_gradw_descriptor() const {
    return expected_descriptor_of(query::diff_weights_pd, 0);
  }

  const_dnnl_primitive_desc_t expected_gradb_descriptor() const {
    return expected_descriptor_of(query::diff_weights_pd, 1);
  }

  void create_primitive(const descriptor_group& desc, dnnl_primitive_at_t* inputs,
      const_dnnl_primitive_t* outputs) {
    dnnl_primitive_t result;
    error::wrap_c_api(dnnl_primitive_create(&result, desc.get(), inputs, outputs),
        "could not create a primitive");
    reset(result);
  }

  void execute(stream& parallel_control) {
    std::vector<dnnl_primitive_t> execution_sequence;
    dnnl_primitive_t c_api_error_primitive;

    execution_sequence.push_back(get());
    error::wrap_c_api(dnnl_stream_submit(
          parallel_control.get(), execution_sequence.size(),
          &execution_sequence[0], &c_api_error_primitive),
        "could not execute the computation");
  }
};

} // namespace ideep

#endif
