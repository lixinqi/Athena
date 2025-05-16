def infer_symbolic(infer_ctx, inputs, attrs):
    return inputs


def infer_meta(inputs, attrs, mut_outputs):
    mut_outputs[0].dims = inputs[0].dims
    mut_outputs[0].dtype = inputs[0].dtype
