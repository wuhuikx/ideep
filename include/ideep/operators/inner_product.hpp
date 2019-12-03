#ifndef IDEEP_OPERATORS_INNER_PRODUCT_HPP
#define IDEEP_OPERATORS_INNER_PRODUCT_HPP

namespace ideep {

struct inner_product_forward : public dnnl::matmul {

  using super = dnnl::matmul;
  template <bool with_bias = true>
  static void compute(
      const tensor& src,
      const tensor& weights,
      const tensor& bias,
      tensor& dst,
      const scale_t& src_scales = scale_t(),
      const scale_t& weights_scales = scale_t(),
      const scale_t& dst_scales = scale_t(),
      const attr_t& attr = attr_t(),
      const prop_kind aprop_kind = prop_kind::forward,
      const lowp_kind alowp_kind = LOWP_U8S8,
      const engine& aengine = engine::cpu_engine()) {
    compute_impl<with_bias>(
        src,
        weights,
        bias,
        dst,
        src_scales,
        weights_scales,
        dst_scales,
        attr,
        aprop_kind,
        alowp_kind,
        aengine);
  }

  static void compute(
      const tensor& src,
      const tensor& weights,
      tensor& dst,
      const scale_t& src_scales = scale_t(),
      const scale_t& weights_scales = scale_t(),
      const scale_t& dst_scales = scale_t(),
      const attr_t& attr = attr_t(),
      const prop_kind aprop_kind = prop_kind::forward,
      const lowp_kind alowp_kind = LOWP_U8S8,
      const engine& aengine = engine::cpu_engine()) {
    static tensor dummy_bias;
    compute<false>(
        src,
        weights,
        dummy_bias,
        dst,
        src_scales,
        weights_scales,
        dst_scales,
        attr,
        aprop_kind,
        alowp_kind,
        aengine);
  }

  static memory::desc expected_weights_descriptor(
      const dims& weights_dims,
      tensor::data_type dtype = tensor::data_type::f32,
      tensor::data_type x_dtype = tensor::data_type::f32,
      prop_kind aprop_kind = prop_kind::forward,
      const engine& aengine = engine::cpu_engine()) {
    auto ndims = weights_dims.size();
    auto x_dims = weights_dims;
    x_dims[ndims-2] = 1;
    x_dims[ndims-1] = weights_dims[0];
    auto y_dims = {x_dims[0], weights_dims[1]};
    if (ndims == 3) 
        y_dims = {x_dims[0], x_dims[1], weights_dims[2]};

    auto y_dtype =
        (dtype != tensor::data_type::s8) ? dtype : tensor::data_type::s32;
    IDEEP_ENFORCE(
        x_dims.size() == weights_dims.size(),
        "Invalid dims for data and weights");
    tensor::desc x_desc(
        x_dims, x_dtype, ndims == 2 ? format_tag::ab : format_tag::abc);
    tensor::desc y_desc(y_dims, y_dtype, ndims == 2 ? format_tag::ab : format_tag::abc);
    tensor::desc weights_desc(
        weights_dims , dtype, ndims == 2 ? format_tag::ab : format_tag::abc);
    auto pd =
        primitive_desc({x_desc, weights_desc, y_desc}, aengine);
    return pd.weights_desc();
  }

private:
  template <bool with_bias = true>
 static void compute_impl(
     const tensor& src,
     const tensor& weights,
     const tensor& bias,
     tensor& dst,
     const scale_t& src_scales = scale_t(),
     const scale_t& weights_scales = scale_t(),
     const scale_t& dst_scales = scale_t(),
     const attr_t& attr = attr_t(),
     const prop_kind aprop_kind = prop_kind::forward,
     const lowp_kind alowp_kind = LOWP_U8S8,
     const engine& aengine = engine::cpu_engine()) {
   IDEEP_ENFORCE(src.ndims() == weights.ndims(), "Invalid dims in src or weights");

   tensor::desc src_desc, weights_desc, bias_desc;
   attr_t op_attr, src_attr, weights_attr, bias_attr;
   scale_t dst_scales_in;
   auto dst_data_type = data_type::f32;
   tensor::dims dst_dims;

   auto weights_scales_in =
       weights.has_scale() ? weights.get_scale() : weights_scales;

   if (!weights_scales_in.empty()) {
     IDEEP_ENFORCE(
         alowp_kind == LOWP_U8S8 || alowp_kind == LOWP_S8S8,
         "Unsupported lowp kind");

     auto src_scales_in = src.has_scale()
         ? src.get_scale()
         : (src_scales.empty() ? IDEEP_DEF_SCALE : src_scales);
     src_desc = {src.get_dims(),
                 alowp_kind == LOWP_U8S8 ? data_type::u8 : data_type::s8,
                 format_tag::any};
     if (src.get_data_type() == data_type::f32) {
       src_attr = {0, src_scales_in};
     }
     
     dst_dims = {src_desc.get_dim(0), weights.get_dim(1)};
     int scale_size = (weights_scales_in.size() > 1) ? dst_dims[1] : 1;
     weights_desc = {weights.get_dims(), data_type::s8, format_tag::any};
     if (weights.get_data_type() == data_type::f32) {
       weights_attr = {IDEEP_TENSOR_SCALE_MASK(scale_size, false), weights_scales_in};
     }
     
     // determine dst data type
     if (dst_scales.empty() || dst_scales == IDEEP_DEF_SCALE) {
       dst_data_type = data_type::f32;
     } else {
       dst_data_type = data_type::u8;
     }

     // fill primitive attr
     scale_t op_scales(scale_size), bias_scales(scale_size);
     dst_scales_in = (dst_scales.empty() || dst_data_type == data_type::f32) 
	     ? IDEEP_DEF_SCALE : dst_scales;
     for (int i = 0; i < scale_size; i++) {
       bias_scales[i] = src_scales_in[0] * weights_scales_in[i];
       op_scales[i] = dst_scales_in[0] / bias_scales[i];
     }
     op_attr.set_output_scales(IDEEP_OP_SCALE_MASK(scale_size), op_scales);
     // op_attr.set_int_output_round_mode(round_mode::round_nearest);
     
     auto src_zero_point = src.has_zero_point()
         ? src.get_zero_point() : std::vector<int32_t>(1);
     auto wei_zero_point = weights.has_zero_point()
         ? weights.get_zero_point() : std::vector<int32_t>(1);
     IDEEP_ENFORCE(
         src_zero_point.size() == 1 && wei_zero_point.size() == 1, 
	 "DNNL only support 1-dim zero_point");
     op_attr.set_zero_points(DNNL_ARG_SRC, 
		     IDEEP_OP_ZP_MASK(src_zero_point.size()), src_zero_point);
     op_attr.set_zero_points(DNNL_ARG_WEIGHTS, 
		     IDEEP_OP_ZP_MASK(wei_zero_point.size()), wei_zero_point);
     
     if (dst_data_type != data_type::f32) {
       auto dst_zero_point = dst.has_zero_point()
           ? dst.get_zero_point() : std::vector<int32_t>(1);
       IDEEP_ENFORCE(
           dst_zero_point.size() == 1, 
	   "DNNL only support 1-dim zero_point");
       op_attr.set_zero_points(DNNL_ARG_DST, 
  		     IDEEP_OP_ZP_MASK(dst_zero_point.size()), dst_zero_point);
     }

     if (with_bias) {
       bias_desc = {bias.get_dims(), data_type::s32, format_tag::any};
       if (bias.get_data_type() == data_type::f32) {
         bias_attr = {IDEEP_TENSOR_SCALE_MASK(scale_size, false), bias_scales};
       }
     }
   } else {
     op_attr = attr;
     src_desc = {src.get_dims(), data_type::f32, format_tag::any};
     if (src.has_scale()) {
       auto src_scale = src.get_scale();
       src_scale[0] = 1.0f / src_scale[0];
       src_attr = {0, src_scale};
     }
     dst_dims = {src_desc.get_dim(0), weights.get_dim(1)};
     weights_desc = weights.get_desc().to_format_any();
     IDEEP_ENFORCE(weights.get_data_type() == data_type::f32, "Incorrect data type in weights");
     if (with_bias) {
       IDEEP_ENFORCE(bias.get_data_type() == data_type::f32, "Incorrect data type in bias");
       bias_desc = bias.get_desc().to_format_any();
     }
   }
   //  auto src_desc_in = src_desc.to_format_any();
   //  tensor::desc weights_desc_in;
   //   if (weights_desc.get_data_type() == data_type::s8 ||
   //       weights_desc.get_data_type() == data_type::u8)
   //     weights_desc_in = weights_desc.to_format_any();
   //   else
   //     weights_desc_in = weights_desc;
   //  auto bias_desc_in =
   //      with_bias ? bias_desc.to_format_any() : tensor::desc();
   tensor::desc dst_desc(dst_dims, dst_data_type, format_tag::any);
   auto pd = with_bias
       ? primitive_desc(
             {src_desc, weights_desc, bias_desc, dst_desc}, op_attr, aengine)
       : primitive_desc(
             {src_desc, weights_desc, dst_desc}, op_attr, aengine);
    auto expected_src = src.reorder_if_differ_in(pd.src_desc(), src_attr);
    auto expected_weights = weights.reorder_if_differ_in(pd.weights_desc(), weights_attr);
    dst.reinit_if_necessary(pd.dst_desc());
   if (!dst_scales.empty() && dst_data_type != data_type::f32) {
     dst.set_scale(dst_scales_in);
   }
   if (with_bias){
     auto expected_bias = bias.reorder_if_differ_in(pd.bias_desc(), bias_attr);
     super(pd).execute(stream::default_stream(),
                       {{DNNL_ARG_SRC, expected_src},
                        {DNNL_ARG_WEIGHTS, expected_weights},
                        {DNNL_ARG_BIAS, expected_bias},
                        {DNNL_ARG_DST, dst}});
   } else {
     super(pd).execute(stream::default_stream(),
                       {{DNNL_ARG_SRC, expected_src},
                        {DNNL_ARG_WEIGHTS, expected_weights},
                        {DNNL_ARG_DST, dst}});
   }
  }
};

struct inner_product_forward_origin : public dnnl::inner_product_forward {

  using super = dnnl::inner_product_forward;
  template <bool with_bias = true>
  static void compute(
      const tensor& src,
      const tensor& weights,
      const tensor& bias,
      tensor& dst,
      const scale_t& src_scales = scale_t(),
      const scale_t& weights_scales = scale_t(),
      const scale_t& dst_scales = scale_t(),
      const attr_t& attr = attr_t(),
      const prop_kind aprop_kind = prop_kind::forward,
      const lowp_kind alowp_kind = LOWP_U8S8,
      const engine& aengine = engine::cpu_engine()) {
    compute_impl<with_bias>(
        src,
        weights,
        bias,
        dst,
        src_scales,
        weights_scales,
        dst_scales,
        attr,
        aprop_kind,
        alowp_kind,
        aengine);
  }

  static void compute(
      const tensor& src,
      const tensor& weights,
      tensor& dst,
      const scale_t& src_scales = scale_t(),
      const scale_t& weights_scales = scale_t(),
      const scale_t& dst_scales = scale_t(),
      const attr_t& attr = attr_t(),
      const prop_kind aprop_kind = prop_kind::forward,
      const lowp_kind alowp_kind = LOWP_U8S8,
      const engine& aengine = engine::cpu_engine()) {
    static tensor dummy_bias;
    compute<false>(
        src,
        weights,
        dummy_bias,
        dst,
        src_scales,
        weights_scales,
        dst_scales,
        attr,
        aprop_kind,
        alowp_kind,
        aengine);
  }

  static memory::desc expected_weights_descriptor(
      const dims& weights_dims,
      tensor::data_type dtype = tensor::data_type::f32,
      tensor::data_type x_dtype = tensor::data_type::f32,
      prop_kind aprop_kind = prop_kind::forward,
      const engine& aengine = engine::cpu_engine()) {
    auto x_dims = weights_dims;
    x_dims[0] = 1;
    auto y_dims = {x_dims[0], weights_dims[0]};
    auto ndims = weights_dims.size();
    auto y_dtype =
        (dtype != tensor::data_type::s8) ? dtype : tensor::data_type::s32;

    IDEEP_ENFORCE(
        x_dims.size() == weights_dims.size(),
        "Invalid dims for data and weights");
    tensor::desc x_desc(
        x_dims, x_dtype, ndims == 2 ? format_tag::nc : format_tag::nchw);
    tensor::desc y_desc(y_dims, y_dtype, format_tag::nc);
    tensor::desc weights_desc(
        weights_dims, dtype, ndims == 2 ? format_tag::oi : format_tag::oihw);

    auto pd =
        primitive_desc({aprop_kind, x_desc, weights_desc, y_desc}, aengine);
    return pd.weights_desc();
  }

private:
  template <bool with_bias = true>
 static void compute_impl(
     const tensor& src,
     const tensor& weights,
     const tensor& bias,
     tensor& dst,
     const scale_t& src_scales = scale_t(),
     const scale_t& weights_scales = scale_t(),
     const scale_t& dst_scales = scale_t(),
     const attr_t& attr = attr_t(),
     const prop_kind aprop_kind = prop_kind::forward,
     const lowp_kind alowp_kind = LOWP_U8S8,
     const engine& aengine = engine::cpu_engine()) {
   IDEEP_ENFORCE(src.ndims() == weights.ndims(), "Invalid dims in src or weights");

   tensor::desc src_desc, weights_desc, bias_desc;
   attr_t op_attr, src_attr, weights_attr, bias_attr;
   scale_t dst_scales_in;
   auto dst_data_type = data_type::f32;
   tensor::dims dst_dims;

   auto weights_scales_in =
       weights.has_scale() ? weights.get_scale() : weights_scales;

   if (!weights_scales_in.empty()) {
     IDEEP_ENFORCE(
         alowp_kind == LOWP_U8S8 || alowp_kind == LOWP_S8S8,
         "Unsupported lowp kind");

     auto src_scales_in = src.has_scale()
         ? src.get_scale()
         : (src_scales.empty() ? IDEEP_DEF_SCALE : src_scales);

     src_desc = {src.get_dims(),
                 alowp_kind == LOWP_U8S8 ? data_type::u8 : data_type::s8,
                 format_tag::any};
     if (src.get_data_type() == data_type::f32) {
       src_attr = {0, src_scales_in};
     }

     dst_dims = {src_desc.get_dim(0), weights.get_dim(0)};
     int scale_size = (weights_scales_in.size() > 1) ? dst_dims[1] : 1;

     weights_desc = {weights.get_dims(), data_type::s8, format_tag::any};
     if (weights.get_data_type() == data_type::f32) {
       weights_attr = {IDEEP_TENSOR_SCALE_MASK(scale_size, false), weights_scales_in};
     }

     // determine dst data type
     if (dst_scales.empty() || dst_scales == IDEEP_DEF_SCALE) {
       dst_data_type = data_type::f32;
     } else if (attr.non_negitive_output()){
       dst_data_type = data_type::u8;
     } else {
       dst_data_type = data_type::s8;
     }

     // fill primitive attr
     scale_t op_scales(scale_size), bias_scales(scale_size);
     dst_scales_in = (dst_scales.empty() || dst_data_type == data_type::f32) ? IDEEP_DEF_SCALE : dst_scales;
     for (int i = 0; i < scale_size; i++) {
       bias_scales[i] = src_scales_in[0] * weights_scales_in[i];
       op_scales[i] = dst_scales_in[0] / bias_scales[i];
     }
     op_attr.set_output_scales(IDEEP_OP_SCALE_MASK(scale_size), op_scales);
    //  op_attr.set_int_output_round_mode(round_mode::round_nearest);

     if (with_bias) {
       bias_desc = {bias.get_dims(), data_type::s32, format_tag::any};
       if (bias.get_data_type() == data_type::f32) {
         bias_attr = {IDEEP_TENSOR_SCALE_MASK(scale_size, false), bias_scales};
       }
     }
   } else {
     op_attr = attr;
     src_desc = {src.get_dims(), data_type::f32, format_tag::any};
     if (src.has_scale()) {
       auto src_scale = src.get_scale();
       src_scale[0] = 1.0f / src_scale[0];
       src_attr = {0, src_scale};
     }
     dst_dims = {src_desc.get_dim(0), weights.get_dim(0)};
     weights_desc = weights.get_desc().to_format_any();
     IDEEP_ENFORCE(weights.get_data_type() == data_type::f32, "Incorrect data type in weights");
     if (with_bias) {
       IDEEP_ENFORCE(bias.get_data_type() == data_type::f32, "Incorrect data type in bias");
       bias_desc = bias.get_desc().to_format_any();
     }
   }
  //  auto src_desc_in = src_desc.to_format_any();
  //  tensor::desc weights_desc_in;
  //   if (weights_desc.get_data_type() == data_type::s8 ||
  //       weights_desc.get_data_type() == data_type::u8)
  //     weights_desc_in = weights_desc.to_format_any();
  //   else
  //     weights_desc_in = weights_desc;
  //  auto bias_desc_in =
  //      with_bias ? bias_desc.to_format_any() : tensor::desc();
   tensor::desc dst_desc(dst_dims, dst_data_type, format_tag::any);
   auto pd = with_bias
       ? primitive_desc(
             {aprop_kind, src_desc, weights_desc, bias_desc, dst_desc}, op_attr, aengine)
       : primitive_desc(
             {aprop_kind, src_desc, weights_desc, dst_desc}, op_attr, aengine);
    auto expected_src = src.reorder_if_differ_in(pd.src_desc(), src_attr);
    auto expected_weights = weights.reorder_if_differ_in(pd.weights_desc(), weights_attr);
    dst.reinit_if_necessary(pd.dst_desc());
   if (!dst_scales.empty() && dst_data_type != data_type::f32) {
     dst.set_scale(dst_scales_in);
   }
    if (with_bias){
      auto expected_bias = bias.reorder_if_differ_in(pd.bias_desc(), bias_attr);
      super(pd).execute(stream::default_stream(),
                        {{DNNL_ARG_SRC, expected_src},
                         {DNNL_ARG_WEIGHTS, expected_weights},
                         {DNNL_ARG_BIAS, expected_bias},
                         {DNNL_ARG_DST, dst}});
    } else {
      super(pd).execute(stream::default_stream(),
                        {{DNNL_ARG_SRC, expected_src},
                         {DNNL_ARG_WEIGHTS, expected_weights},
                         {DNNL_ARG_DST, dst}});
    }

    if (attr.non_negitive_output() && dst.get_data_type() == data_type::s8) {
     tensor::desc dst_u8_desc = dst.get_desc().to_type(data_type::u8);
     dst.set_desc(dst_u8_desc);
   }
  }
};

// TODO: parameter sequence adjust?
struct inner_product_backward_data : public dnnl::inner_product_backward_data {
  using super = dnnl::inner_product_backward_data;
  static void compute(const tensor& diff_dst,
                      const tensor& weights,
                      const dims& diff_src_dims,
                      tensor& diff_src,
                      const engine& aengine = engine::cpu_engine()) {
    tensor::desc diff_src_desc = tensor::desc(diff_src_dims, diff_dst.get_data_type());
    auto weights_desc = weights.get_desc();
    auto diff_dst_desc = diff_dst.get_desc();
    auto forward_hints = inner_product_forward_origin::primitive_desc(
        {prop_kind::forward, diff_src_desc, weights_desc, diff_dst_desc}, aengine);
    auto pd =primitive_desc(
        {diff_src_desc, weights_desc, diff_dst_desc}, aengine, forward_hints);
    auto expected_diff_dst = diff_dst.reorder_if_differ_in(pd.diff_dst_desc());
    auto expected_weights = weights.reorder_if_differ_in(pd.weights_desc());
    diff_src.reinit_if_necessary(pd.diff_src_desc());
    super(pd).execute(stream::default_stream(),
                      {{DNNL_ARG_DIFF_DST, expected_diff_dst},
                       {DNNL_ARG_WEIGHTS, expected_weights},
                       {DNNL_ARG_DIFF_SRC, diff_src}});
  }
};

struct inner_product_backward_weights : public dnnl::inner_product_backward_weights {
  using super = dnnl::inner_product_backward_weights;
  template<bool with_diff_bias = true>
  static void compute(const tensor& src,
                      const tensor& diff_dst,
                      tensor& diff_weights,
                      tensor& diff_bias,
                      const engine& aengine = engine::cpu_engine()) {
    compute_impl<with_diff_bias>(src, diff_dst, diff_weights, diff_bias);
  }

  static void compute(const tensor& src,
                      const tensor& diff_dst,
                      tensor& diff_weights,
                      const engine& aengine = engine::cpu_engine()) {
   static tensor dummy_diff_bias;
   compute<false>(src, diff_dst, diff_weights, dummy_diff_bias);
  }

private:
  template<bool with_diff_bias = true>
    static void compute_impl(const tensor& src,
                             const tensor& diff_dst,
                             tensor& diff_weights,
                             tensor& diff_bias,
                             const engine& aengine = engine::cpu_engine()) {
      auto src_desc = src.get_desc();
      auto diff_dst_desc = diff_dst.get_desc();
      auto diff_weights_dims = src.get_dims();
      diff_weights_dims[0] = diff_dst.get_dim(1);
      tensor::desc diff_weights_desc = tensor::desc(diff_weights_dims, diff_dst.get_data_type());
      tensor::desc diff_bias_desc = with_diff_bias
          ? tensor::desc({diff_dst.get_dim(1)}, diff_dst.get_data_type())
          : tensor::desc();
      auto forward_hints = with_diff_bias
          ? inner_product_forward_origin::primitive_desc({prop_kind::forward, src_desc,
              diff_weights_desc, diff_bias_desc, diff_dst_desc}, aengine)
          : inner_product_forward_origin::primitive_desc({prop_kind::forward, src_desc,
              diff_weights_desc, diff_dst_desc}, aengine);
      auto pd = with_diff_bias
          ? primitive_desc({src_desc, diff_weights_desc, diff_bias_desc, diff_dst_desc}, aengine, forward_hints)
          : primitive_desc({src_desc, diff_weights_desc, diff_dst_desc}, aengine, forward_hints);
      auto expected_diff_dst = diff_dst.reorder_if_differ_in(pd.diff_dst_desc());
      auto expected_src = src.reorder_if_differ_in(pd.src_desc());
      diff_weights.reinit_if_necessary(pd.diff_weights_desc());
      if (with_diff_bias) {
        diff_bias.reinit_if_necessary(pd.diff_bias_desc());
        super(pd).execute(stream::default_stream(),
                          {{DNNL_ARG_DIFF_DST, expected_diff_dst},
                           {DNNL_ARG_SRC, expected_src},
                           {DNNL_ARG_DIFF_WEIGHTS ,diff_weights},
                           {DNNL_ARG_DIFF_BIAS, diff_bias}});
      } else {
        super(pd).execute(stream::default_stream(),
                          {{DNNL_ARG_DIFF_DST, expected_diff_dst},
                           {DNNL_ARG_SRC, expected_src},
                           {DNNL_ARG_DIFF_WEIGHTS ,diff_weights},
                           {DNNL_ARG_DIFF_BIAS, diff_bias}});
      }
    }
};

}  // namespace ideep

#endif
