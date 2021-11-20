from bigse import se_resnet50 as net
from mindspore.nn import Accuracy
from mindspore import load_checkpoint, load_param_into_net
from mindspore import Model, context
from dataset import test_data
import mindspore.nn as nn
from mindspore.train.callback import LossMonitor
import os
import argparse
import logging


def get_logger(LEVEL, log_file):
    # head = '[%(asctime)-15s] [%(levelname)s] %(message)s'
    head = ''
    if LEVEL == 'info':
        logging.basicConfig(level=logging.INFO, format=head, filename=log_file)
    elif LEVEL == 'debug':
        logging.basicConfig(level=logging.DEBUG, format=head, filename=log_file)
    logger = logging.getLogger()
    return logger


parser = argparse.ArgumentParser(description='model Training')
parser.add_argument("--batch_size", default=640)
parser.add_argument("--device", default="GPU", type=str, choices=['Ascend', 'GPU', 'CPU'])
parser.add_argument("--device_id", default=2, type=int)
parser.add_argument('--num_workers', default=2, type=int)
parser.add_argument('--num_classes', default=2388, type=int)
args = parser.parse_args()

if __name__ == '__main__':
    net = net(2388)
    acc_logger = get_logger('info', "acc/bigse_acc.txt")
    context.set_context(device_target=args.device, device_id=args.device_id)
    for idx, filename in enumerate(os.listdir("ckpt_bigse")):
        # if idx > 5:
        #     break
        if filename.endswith("ckpt"):
            file = os.path.join("ckpt_bigse", filename)
            print(file)
            # file = "ckpt_resnet50/smallse-2_10777.ckpt"
            param_dict = load_checkpoint(file)
            load_param_into_net(net, param_dict)

            net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

            model = Model(net, net_loss, metrics={"Accuracy": Accuracy()})

            test_dataset = test_data(args)
            acc = model.eval(test_dataset, dataset_sink_mode=False)
            acc_str =filename + "  {}".format(acc)
            acc_logger.info(file)
            acc_logger.info(acc_str)
            print(acc_str)


