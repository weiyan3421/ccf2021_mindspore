import os
import numpy as np
import mindspore.dataset as ds
from PIL import Image
import mindspore.common.dtype as mstype
from mindspore.dataset.vision import Inter, Border
from mindspore.dataset.transforms.py_transforms import Compose
from mindspore.dataset.transforms import c_transforms
import mindspore.dataset.vision.c_transforms as c_vision
import mindspore.dataset.vision.py_transforms as py_vision

# ds.config.set_seed(58)

# def get_goods_class():
#     gas = GAS("Accesskey-f58826765f344863cf230eb3b5db9665")
#     dataset = Dataset("RP2K", gas)
#
#     info = dataset.catalog.classification.categories
#     goods_class = [i.name for i in info]
#
#     return goods_class

# def get_goods_class():
#     with open("label.txt", "r", encoding="gbk") as f:
#         goods_class = [n.strip("\n\r") for n in f.readlines()]
#     return goods_class


def get_goods_class():
    train_path = "./all/train"
    goods_class = []
    for filename in os.listdir(train_path):
        goods_class.append(filename)
    return goods_class


def get_data_path(str):
    path = os.path.join("./all", str)
    data_path = []
    label = []
    for filename in os.listdir(path):
        for file in os.listdir(os.path.join(path, filename)):
            if file is None:
                break
            image_path = os.path.join(path, filename, file)
            data_path.append(image_path)
            label.append(filename)
    return data_path, label


class dataset():
    def __init__(self, tag="train"):
        super(dataset, self).__init__()
        self.goods_class = get_goods_class()

        data, label = get_data_path(tag)
        self.data_path = data
        self.label = label
      

    def __len__(self):
        return len(self.data_path)
    
    def __getitem__(self, item):
        image_path = self.data_path[item]
    
        try:
            img_RGB = Image.open(image_path).convert('RGB')
        except BaseException:
            print("图片读取发生错误，错误的图片为{},请检查该图片！".format(image_path))


        label_str = self.label[item]
        label_idx = self.goods_class.index(label_str)
        return img_RGB, label_idx


def train_data(args, tag="train"):
    train_set = dataset(tag)
    train_data = ds.GeneratorDataset(source=train_set, column_names=["image", "label"],
                                     num_parallel_workers=args.num_workers, shuffle=True)

#    onehot_op = c_transforms.OneHot(num_classes=args.num_classes)
#    train_data = train_data.map(operations=onehot_op, input_columns=["label"])
    op = c_transforms.TypeCast(mstype.int32)
    train_data = train_data.map(operations=op, input_columns=["label"])

    c_compose = c_transforms.Compose([#c_vision.Decode(),
        c_vision.RandomHorizontalFlip(0.5),
        c_vision.RandomRotation(degrees=15.0, resample=Inter.NEAREST, expand=True),
        c_vision.Rescale(1.0 / 255.0, 0.0), 
        c_vision.Resize([224, 224], Inter.BICUBIC),  
        c_vision.HWC2CHW()
        ])
    train_data = train_data.map(operations=c_compose, input_columns=["image"])

    erasre_op = py_vision.RandomErasing(prob=0.5, scale=(0.02, 0.22), ratio=(0.2, 2.2))
    train_data = train_data.map(operations=erasre_op, input_columns="image")
    train_data = train_data.batch(args.batch_size, drop_remainder=args.sink_mode, num_parallel_workers=args.num_workers)

    return train_data


def test_data(args, tag="test"):
    test_set = dataset(tag)
    test_data = ds.GeneratorDataset(source=test_set, column_names=["image", "label"],
                                    num_parallel_workers=args.num_workers, shuffle=False)

    op = c_transforms.TypeCast(mstype.int32)
    test_data = test_data.map(operations=op, input_columns=["label"])

    c_compose = c_transforms.Compose([  # c_vision.Decode(),
        c_vision.Rescale(1.0 / 255.0, 0.0),
        c_vision.Resize([224, 224], Inter.BICUBIC),
        c_vision.HWC2CHW()
    ])
    test_data = test_data.map(operations=c_compose, input_columns=["image"])

    test_data = test_data.batch(args.batch_size*10, drop_remainder=False, num_parallel_workers=args.num_workers)

    return test_data

