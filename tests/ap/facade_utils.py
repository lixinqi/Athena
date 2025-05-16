def quant_infer_symbolic(infer_ctx, inputs, attrs):
    print("inputs:", inputs)
    print("type(inputs):", type(inputs))
    return [inputs[0], inputs[0]]


def quant_infer_meta(inputs, attrs, mut_outputs):
    mut_outputs[0].dims = inputs[0].dims
    mut_outputs[0].dtype = inputs[0].dtype
    mut_outputs[1].dims = inputs[0].dims
    mut_outputs[1].dtype = inputs[0].dtype
