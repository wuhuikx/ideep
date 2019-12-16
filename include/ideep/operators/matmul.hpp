#ifndef IDEEP_OPERATORS_INNER_PRODUCT_MATMUL_HPP
#define IDEEP_OPERATORS_INNER_PRODUCT_MATMUL_HPP

namespace ideep {

struct matmul_forward : public dnnl::matmul {

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

   tensor scales_m, src_zero_point_m, wei_zero_point_m, dst_zero_point_m;
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
     
     bool flag_runtime = true;
     if (flag_runtime) {
         op_attr.set_output_scales(IDEEP_OP_SCALE_MASK(scale_size), {DNNL_RUNTIME_F32_VAL});
         tensor::desc scales_desc = {{scale_size}, memory::data_type::f32, {1}};
         scales_m.reinit(scales_desc, aengine);
         auto s = reinterpret_cast<float *>(scales_m.get_data_handle());
         for (memory::dim i = 0; i < scale_size; ++i) {
             bias_scales[i] = src_scales_in[0] * weights_scales_in[i];
             s[i] = dst_scales_in[0] / bias_scales[i];
	 }
     } else {
         for (int i = 0; i < scale_size; i++) {
           bias_scales[i] = src_scales_in[0] * weights_scales_in[i];
           op_scales[i] = dst_scales_in[0] / bias_scales[i];
         }
         op_attr.set_output_scales(IDEEP_OP_SCALE_MASK(scale_size), op_scales);
     }
     
     auto src_zero_point = src.has_zero_point()
         ? src.get_zero_point() : std::vector<int32_t>(1);
     auto src_zero_point_size = src_zero_point.size();
     IDEEP_ENFORCE(
         src_zero_point_size == 1, 
         "DNNL only support 1-dim zero_point");
     if (flag_runtime) {
         op_attr.set_zero_points(DNNL_ARG_SRC, IDEEP_OP_ZP_MASK(1), {DNNL_RUNTIME_S32_VAL});
         tensor::desc src_zero_point_desc = {{src_zero_point_size}, memory::data_type::s32, {1}};
         src_zero_point_m.reinit(src_zero_point_desc, aengine);
         auto z = reinterpret_cast<int32_t *>(src_zero_point_m.get_data_handle());
         for (memory::dim i = 0; i < src_zero_point_size; ++i)
             z[i] = src_zero_point[i];
     } else { 
        op_attr.set_zero_points(DNNL_ARG_SRC, 
            IDEEP_OP_ZP_MASK(src_zero_point.size()), src_zero_point);
     } 

     auto wei_zero_point = weights.has_zero_point()
         ? weights.get_zero_point() : std::vector<int32_t>(1);
     size_t wei_zero_point_size = 1;
     if (flag_runtime) {
         op_attr.set_zero_points(DNNL_ARG_WEIGHTS, IDEEP_OP_ZP_MASK(1), {DNNL_RUNTIME_S32_VAL});
         tensor::desc wei_zero_point_desc = {{wei_zero_point_size}, memory::data_type::s32, {1}};
         wei_zero_point_m.reinit(wei_zero_point_desc, aengine);
         auto z = reinterpret_cast<int32_t *>(wei_zero_point_m.get_data_handle());
         for (memory::dim i = 0; i < wei_zero_point_size; ++i)
             z[i] = wei_zero_point[i];
     } else { 
         op_attr.set_zero_points(DNNL_ARG_WEIGHTS, 
             IDEEP_OP_ZP_MASK(1), std::vector<int32_t>(1,wei_zero_point[0]));
     }

     auto dst_zero_point = dst.has_zero_point()
         ? dst.get_zero_point() : std::vector<int32_t>(1);
     IDEEP_ENFORCE(
         dst_zero_point.size() == 1, 
         "DNNL only support 1-dim zero_point");
     auto dst_zero_point_size = dst_zero_point.size();
     if (dst_data_type != data_type::f32) {
         if (flag_runtime) {
             op_attr.set_zero_points(DNNL_ARG_DST, IDEEP_OP_ZP_MASK(1), {DNNL_RUNTIME_S32_VAL});
             tensor::desc dst_zero_point_desc = {{dst_zero_point_size}, memory::data_type::s32, {1}};
             dst_zero_point_m.reinit(dst_zero_point_desc, aengine);
             auto z = reinterpret_cast<int32_t *>(dst_zero_point_m.get_data_handle());
             for (memory::dim i = 0; i < dst_zero_point_size; ++i)
                 z[i] = dst_zero_point[i];
         } else { 
             op_attr.set_zero_points(DNNL_ARG_DST, 
  	  	     IDEEP_OP_ZP_MASK(dst_zero_point.size()), dst_zero_point);
	 }
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
                        {DNNL_ARG_DST, dst},
                        {DNNL_ARG_ATTR_OUTPUT_SCALES, scales_m},
                        {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC, src_zero_point_m},
                        {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS, wei_zero_point_m},
                        {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST, dst_zero_point_m}});
   } else {
     super(pd).execute(stream::default_stream(),
                       {{DNNL_ARG_SRC, expected_src},
                        {DNNL_ARG_WEIGHTS, expected_weights},
                        {DNNL_ARG_DST, dst},
                        {DNNL_ARG_ATTR_OUTPUT_SCALES, scales_m},
                        {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC, src_zero_point_m},
                        {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS, wei_zero_point_m},
                        {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST, dst_zero_point_m}});
   }
  }
};

}  // namespace ideep

#endif
