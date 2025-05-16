def infer_symbolic(infer_ctx, inputs, attrs):
    return inputs


def infer_meta(inputs, attrs, mut_outputs):
    def copy_meta(i):
        mut_outputs[i].dims = inputs[i].dims
        mut_outputs[i].dtype = inputs[i].dtype

    map(copy_meta, range(len(inputs)))
