"""
Make rois for mask branch.
"""

import mxnet as mx
import pdb
import os

workdir = os.getcwd()
# Switch directory to reach get_config hook
os.chdir('/sniper')
from main_test_mask import get_config
cfg = get_config()
num_classes = cfg.dataset.NUM_CLASSES
# Switch back
os.chdir(workdir)



class ExpandClsOperator(mx.operator.CustomOp):
    def __init__(self):
        super(ExpandClsOperator, self).__init__()

    def forward(self, is_train, req, in_data, out_data, aux):
        # background cls is not included
        cls_prob = in_data[0]
        cls_prob_ids = mx.nd.argmax(cls_prob[:, 1: num_classes-1], axis=1, keepdims=True)
        # pdb.set_trace()
        cls_prob_ids_expand = mx.nd.expand_dims(cls_prob_ids, 2)
        mask_ncls_pred_ids = mx.nd.broadcast_to(cls_prob_ids_expand,
                              shape=(cls_prob_ids.shape[0], 28, 28))

        self.assign(out_data[0], req[0], mask_ncls_pred_ids)
        # for i in range(len(out_data)):
        #     self.assign(out_data[i], req[i], in_data[i])

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        for i in range(len(in_grad)):
            self.assign(in_grad[i], req[i], 0)

@mx.operator.register("expand_cls")
class ExpandClsProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(ExpandClsProp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['cls_prob']

    def list_outputs(self):
        return ['cls_prob_expand']

    def infer_shape(self, in_shape):
        cls_prob_shape = in_shape[0]
        output_shape = [cls_prob_shape[0], 28, 28]
        #num_classes = [1,1]
        # pdb.set_trace()
        return in_shape, [output_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return ExpandClsOperator()