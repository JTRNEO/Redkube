"""
Make rois for mask branch.
"""

import mxnet as mx
import pdb

class Debug2Operator(mx.operator.CustomOp):
    def __init__(self):
        super(Debug2Operator, self).__init__()

    def forward(self, is_train, req, in_data, out_data, aux):
        fmask_pred = in_data[0]
        print 'input shape: ', fmask_pred.shape
        pdb.set_trace()
        for i in range(len(out_data)):
            self.assign(out_data[i], req[i], in_data[i])

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        for i in range(len(in_grad)):
            self.assign(in_grad[i], req[i], 0)

@mx.operator.register("debug2")
class Debug2Prop(mx.operator.CustomOpProp):
    def __init__(self):
        super(Debug2Prop, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['data']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        print 'in infer_shape'
        # return [rois_shape, bbox_deltas_shape, data_shape, label_shape], [rois_shape]
        return in_shape, in_shape
    def create_operator(self, ctx, shapes, dtypes):
        return Debug2Operator()