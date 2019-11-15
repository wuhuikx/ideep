#ifndef IDEEP_OPERATORS_CONCAT_HPP
#define IDEEP_OPERATORS_CONCAT_HPP

namespace ideep {

struct concat : public dnnl::concat {

  using super = dnnl::concat;

  static void compute(const std::vector<tensor>& inputs,
                      int axis,
                      tensor& output,
                      const engine& aengine = engine::cpu_engine()) {
    auto input_descs = utils::fmap(inputs, [](const tensor& t) {
      // "upcast" vector<tensor::desc> to vector<memory::desc>
      return static_cast<memory::desc>(t.get_desc());
    });

    // XPZ: TODO: reorder to same format

    auto pd = primitive_desc(axis, input_descs, aengine);

    output.reinit_if_necessary(pd.dst_desc());

    exec_args args {{DNNL_ARG_DST, output}};
    for (int i = 0; i < inputs.size(); ++i) {
      args.insert({DNNL_ARG_MULTIPLE_SRC + i, inputs[i]});
    }

    super(pd).execute(stream::default_stream(), args);
  }

  // for caffe2
  static std::vector<int32_t> compute(
      std::vector<tensor>& inputs,
      int axis,
      bool add_axis,
      tensor& dst,
      const engine& aengine = engine::cpu_engine()) {
    IDEEP_ENFORCE(axis < (inputs[0].ndims() + add_axis),
                  "invalid axis in concat");
    for (int i = 0; i < inputs[0].ndims(); i++) {
      if (i == axis && !add_axis) continue;
      for (unsigned j = 1; j < inputs.size(); j++) {
        IDEEP_ENFORCE(inputs[j].get_dim(i) == inputs[0].get_dim(i),
                      "invalid input dims in concat");
      }
    }

    int32_t dst_channels = 0;
    std::vector<int32_t> axis_info(inputs.size(), 0);
    for (unsigned k = 0; k < inputs.size(); k++) {
      axis_info[k] = add_axis ? 1 : inputs[k].get_dim(axis);
      dst_channels += axis_info[k];
    }

    dims dst_dims(inputs[0].get_dims());
    if (add_axis)
      dst_dims.insert(dst_dims.begin() + axis, dst_channels);
    else
      dst_dims[axis] = dst_channels;

    auto dst_data_type = inputs[0].get_data_type();
    scale_t min_scale(IDEEP_DEF_SCALE);
    if (dst_data_type != data_type::f32) {
      min_scale[0] = std::numeric_limits<float>::max();
      for (auto i : inputs) {
        if (i.get_data_type() != dst_data_type) {
          min_scale = IDEEP_DEF_SCALE;
          dst_data_type = data_type::f32;
          break;
        }
        if (i.has_scale() && (min_scale[0] > i.get_scale()[0])) {
          IDEEP_ENFORCE(i.get_scale().size() == 1, "incorrect scale size");
          min_scale[0] = i.get_scale()[0];
        }
      }
    }

    dims offset_dims(dst_dims.size(), 0);
    dst.reinit(dst_dims, dst_data_type);
    if (dst_data_type != data_type::f32)
      dst.set_scale(min_scale);

    scale_t scales(1);
    // FIXME: To avoid view issue in dnnl
    // NOTE: In dnnl concat, dim 3 and 6+ are not supported.
    // Morewhile, the tensor shape must be blockable to create a view.
    // if (!add_axis && dst_dims.size() != 3 && dst_dims.size() < 6) {
    //   for (unsigned k = 0; k < inputs.size(); k++) {
    //     if (!inputs[k].is_limited_blockable()) {
    //       for (int i = 0; i < inputs.size(); ++i) {
    //         float input_scale =
    //             inputs[i].has_scale() ? inputs[i].get_scale()[0] : 1.0f;
    //         if (inputs[i].get_data_type() != dst_data_type ||
    //             input_scale - min_scale[0] != 0) {
    //           scales[0] = min_scale[0] / input_scale;
    //           tensor input_fp(inputs[i].get_desc().to_type(dst_data_type));
    //           inputs[i].reorder_to(input_fp, {0, scales});
    //           inputs[i] = input_fp;
    //         }
    //       }
    //       compute(inputs, axis, dst);
    //       return axis_info;
    //     }
    //   }
    // }

    for (unsigned i = 0; i < inputs.size(); ++i) {
      auto input_i = inputs[i];
      auto in_dims = inputs[i].get_dims();
      scales[0] = min_scale[0] /
          (input_i.has_scale() ? input_i.get_scale()[0] : 1.0f);
      if (add_axis) {
        in_dims.insert(in_dims.begin() + axis, 1);
        input_i = input_i.reshape(in_dims);
      }
      dst.insert_submemory(input_i, in_dims, offset_dims, {0, scales});
      offset_dims[axis] += axis_info[i];
    }

    return axis_info;
  }
};

}  // namespace ideep

#endif