"""
Make rois for mask branch.
"""

import mxnet as mx
import pdb

class DebugOperator(mx.operator.CustomOp):
    def __init__(self):
        super(DebugOperator, self).__init__()

    def forward(self, is_train, req, in_data, out_data, aux):
        mask_rois = in_data[0]
        mask_polys = in_data[1]
        mask_ids = in_data[2]
        mask_pred = in_data[3]
        print 'mask_rois shape: ', mask_rois.shape
        print 'mask_polys shape: ', mask_polys.shape
        print 'mask_ids: ', mask_ids.shape
        print 'mask_pred: ', mask_pred.shape

        pdb.set_trace()
        for i in range(len(out_data)):
            self.assign(out_data[i], req[i], in_data[i])

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        for i in range(len(in_grad)):
            self.assign(in_grad[i], req[i], 0)

@mx.operator.register("debug")
class DebugProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(DebugProp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['rois', 'mask_polys', 'mask_ids', 'mask_pred']

    def list_outputs(self):
        return ['rois', 'mask_polys', 'mask_ids', 'mask_pred']

    def infer_shape(self, in_shape):
        # rois_shape = in_shape[0]
        # bbox_deltas_shape = in_shape[1]
        # data_shape = in_shape[2]
        print 'in infer_shape'
        pdb.set_trace()
        # return [rois_shape, bbox_deltas_shape, data_shape, label_shape], [rois_shape]
        return in_shape, in_shape
    def create_operator(self, ctx, shapes, dtypes):
        return DebugOperator()