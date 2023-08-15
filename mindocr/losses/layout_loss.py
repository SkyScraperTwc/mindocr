import mindspore as ms
from mindspore import nn, ops
import numpy as np

__all__ = ["VQASerTokenLayoutLMLoss", "LossFromOutput"]


class VQASerTokenLayoutLMLoss(nn.LossBase):
    def __init__(self, num_classes, key=None):
        super().__init__()
        self.loss_class = nn.CrossEntropyLoss(reduction="none")
        self.num_classes = num_classes
        self.ignore_index = self.loss_class.ignore_index
        self.key = key

    def construct(self, predicts, attention_mask, labels):
        if isinstance(predicts, dict) and self.key is not None:
            predicts = predicts[self.key]
        # print("predicts----", predicts, flush=True)
        # print("attention_mask----", attention_mask, flush=True)
        if attention_mask is not None:
            # print("attention_mask----shape---", attention_mask.shape, flush=True)
            # print("predicts----shape---", predicts.shape, flush=True)
            # print("labels----shape---", labels.shape, flush=True)
            # attention_mask = attention_mask.reshape((-1))
            # active_label = ops.mul(labels.reshape((-1,)), attention_mask)
            #
            # attention_mask = ops.broadcast_to(attention_mask, predicts.shape)
            # active_output = ops.mul(predicts.reshape((-1, self.num_classes)), attention_mask)

            loss = self.loss_class(predicts.reshape((-1, self.num_classes)), labels.reshape((-1,)).astype(ms.int32))
            # print("loss---shape--", loss.shape)

            attention_mask = attention_mask.reshape((-1,))
            loss = ops.mul(loss, attention_mask)
            loss = loss[loss > 0]
            # print("loss----shape--", loss.shape, flush=True)
        else:
            loss = self.loss_class(predicts.reshape((-1, self.num_classes)), labels.reshape((-1,)).astype(ms.int32))
        # print("loss----", loss, flush=True)
        return ops.reduce_mean(loss)


class LossFromOutput(nn.LossBase):
    def __init__(self, key="loss", reduction="none"):
        super().__init__()
        self.key = key
        self.reduction = reduction

    def construct(self, predicts, batch):
        loss = predicts
        if self.key is not None and isinstance(predicts, dict):
            loss = loss[self.key]
        if self.reduction == "mean":
            loss = ops.mean(loss)
        elif self.reduction == "sum":
            loss = ops.sum(loss)
        return loss
