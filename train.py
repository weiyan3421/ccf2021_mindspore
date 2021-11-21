import os

from dataset import train_data, test_data
from lr_scheduler import CosineWarmupRestart_2, CosineRestart
from mindspore import dtype as mstype
from mindspore import Model, context, FixedLossScaleManager
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore import load_checkpoint, load_param_into_net
from L_SE import L_SE_resnet50 as net
import mindspore.nn as nn
import numpy as np
from mindspore.common.tensor import Tensor
from mindspore.train.callback import Callback, TimeMonitor

from mindspore.nn import Accuracy
import argparse
import logging


parser = argparse.ArgumentParser(description='model Training')

parser.add_argument('--num_classes', default=2388, type=int)
parser.add_argument('--num_workers', default=2, type=int)
parser.add_argument("--lr", "--learningRate", default=1e-4, type=float)
parser.add_argument("--momentum", default=0.9, type=float)
parser.add_argument("--total_epochs", default=100, type=int)
parser.add_argument("--warmup_epoch", default=4, type=int)
parser.add_argument("--warmup_ratio", default=0.3, type=float)
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--device", default="Ascend", type=str, choices=['Ascend', 'GPU', 'CPU'])
parser.add_argument("--device_id", default=0, type=int)
parser.add_argument("--sink_mode", default=True, type=bool)
args = parser.parse_args()


class MyLossMonitor(Callback):
    def __init__(self, logger, per_print_times=1):
        super(MyLossMonitor, self).__init__()
        self._per_print_times = per_print_times
        self.logger = logger

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs
        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1
        if self._per_print_times != 0 and cb_params.cur_step_num % self._per_print_times == 0:
            str_loss = "epoch: %s step: %s, loss is %s" % (cb_params.cur_epoch_num, cur_step_in_epoch, loss)
            self.logger.info(str_loss)
            print(str_loss, flush=True)


def get_logger(LEVEL, log_file):
    head = '[%(asctime)-15s] [%(levelname)s] %(message)s'
    if LEVEL == 'info':
        logging.basicConfig(level=logging.INFO, format=head, filename=log_file)
    elif LEVEL == 'debug':
        logging.basicConfig(level=logging.DEBUG, format=head, filename=log_file)
    logger = logging.getLogger()
    return logger


if __name__ == '__main__':
    # context.set_context(mode=context.GRAPH_MODE, device_target=args.device)
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device, device_id=args.device_id)
    
    net = net(args.num_classes)
    # param_dict = load_checkpoint("ckpt_bigse/bigse_1-16_5373.ckpt")
    # load_param_into_net(net, param_dict)
    train_data = train_data(args)
    steps_per_epoch = train_data.get_dataset_size()

    # 保存模型
    config_ck = CheckpointConfig(save_checkpoint_steps=steps_per_epoch,
                                 keep_checkpoint_max=args.total_epochs)
    ckpoint = ModelCheckpoint(prefix="bigse", directory="ckpt_bigse", config=config_ck)

    # 定义损失函数
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    # 初始化学习率和定义优化器
    lr_scheduler = Tensor(CosineWarmupRestart_2(args, steps_per_epoch, 8), mstype.float32)
    opt = nn.SGD(net.trainable_params(), learning_rate=lr_scheduler, momentum=0.9, weight_decay=5e-2, loss_scale=1.0)
    
    # 构建模型
    metrics={"Accuracy": Accuracy()}
    
    loss_scale_manager = FixedLossScaleManager(128.0, drop_overflow_update=False)
    if args.device == "Ascend":
        model = Model(net, loss_fn=loss, optimizer=opt, metrics=metrics, amp_level="O2", keep_batchnorm_fp32=False,
                      loss_scale_manager=loss_scale_manager)
    elif args.device == "GPU":
        model = Model(net, loss_fn=loss, optimizer=opt, metrics=metrics, amp_level="O2",
                      loss_scale_manager=loss_scale_manager)
    else:
        raise ValueError("Unsupported platform.")  

    # 定义保存日志和callback函数 
    loss_logger = get_logger('info', "loss/bigse_loss.txt")
    loss_cb = MyLossMonitor(loss_logger, 1)
    time_cb=TimeMonitor(data_size=steps_per_epoch)

    # 训练模型
    print("strat train!")
    model.train(args.total_epochs, train_data, callbacks=[ckpoint, loss_cb, time_cb], dataset_sink_mode=args.sink_mode)

    # 测试集准确率
    test_data = test_data(args)
    acc = model.eval(test_data, dataset_sink_mode=False)
    acc_str = "训练{}个epoch后，最后得到{}".format(args.total_epochs, acc)
    loss_logger.info(acc_str)
    print(acc_str)
 