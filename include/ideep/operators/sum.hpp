#ifndef IDEEP_OPERATORS_SUM_HPP
#define IDEEP_OPERATORS_SUM_HPP

namespace ideep {

struct sum : public dnnl::sum {

  using super = dnnl::sum;

  static void compute(const scale_t& scales,
                      const std::vector<tensor>& inputs,
                      tensor& output,
                      const engine& aengine = engine::cpu_engine()) {
    auto input_descs = utils::fmap(inputs, [](const tensor& t) {
      // "upcast" vector<tensor::desc> to vector<dnnl::memory::desc>
      return static_cast<dnnl::memory::desc>(t.get_desc());
    });
    auto pd = primitive_desc(scales, input_descs, aengine);

    output.reinit_if_necessary(pd.dst_desc());

    std::unordered_map<int, dnnl::memory> args {{DNNL_ARG_DST, output}};
    for (int i = 0; i < inputs.size(); ++i) {
      args.insert({DNNL_ARG_MULTIPLE_SRC + i, inputs[i]});
    }

    super(pd).execute(stream::default_stream(), args);
  }
};

}  // namespace ideep

#endif