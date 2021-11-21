# ccf2021
基于MindSpore AI框架实现零售商品识别比赛提交代码

# 1.代码目录结构说明
.
├── all                                      // 比赛数据
│   ├── test                               // 测试集数据
│   │   ├── 555（冰炫）
│   │   │   ├── 0205_20357.jpg
│   │   │   ├── 0205_57146.jpg
│   │   │   └── 0205_88207.jpg
│   │   ├── 555（金）
│   │   │   └── 截屏2019-12-09下午7.49.21.png
│   │   └── ...
│   └── train                               // 训练集数据 
│       ├── 555（冰炫）
│       │   ├── 0205_10676.jpg
│       │   ├── 0205_22525.jpg
│       │   └── ...
│       ├── 555（金）
│       │   ├── 截屏2019-12-09下午7.48.26.png
│       │   ├── 截屏2019-12-09下午7.48.46.png
│       │   └── ...
│       └── ...
├── ckpt_bigse                             // 模型保存文件夹
│   ├── ckpt-1-5388.ckpt
│   ├── ckpt-2-5388.ckpt
│   └── ...
├── acc                                    // 准确率保存文件夹
│   └── L-SE_acc.txt
├── loss                                   // loss保存文件夹
│   └── L-SE_loss.txt
├── README.md                          // 代码解释说明
├── lr_scheduler.py                       // 动态学习率脚本
├── dataset.py                           // 数据处理脚本
├── L_SE.py                              //  L_SE模型脚本
├── train.py                              // 模型训练脚本
└── eval.py                              // 验证每一个保存的模型的准确率

# 2.自验环境
华为云modelarts平台，Ascend910（mindspore1.5）和GPU V100 （mindspore1.5），以及3090 GPU（mindspore1.5）环境

Python版本：python3.7.5
Python所需环境：
    scipy==1.6.3
    numpy==1.20.3
    mindspore-gpu==1.5.0
    pillow==6.2.1

# 3.--num_classes      分类数目，默认2388
--num_workers      数据处理多线程数，默认4
--lr               基础学习率，默认1e-4
--momentum         梯度下降动量值，默认0.9
--total_epochs     训练总epoch数，默认100
--warmup_epoch     使用warmup预热模型的epoch数，默认4
--warmup_ratio     warmup初始学习率比例，默认0.3
--batch_size       一个batch中样本数量，默认64
--device           训练设备类型，支持Ascend,GPU,和CPU，默认Ascend
--set_device       是否指定设备代号，默认False
--device_id        使用设备的代号，只有在set_device为True时有效，默认0
--sink_mode        训练中是否使用dataset_sink，默认True

# 4.启动训练脚本
训练如何启动：
Ascend训练：
    1、不指定使用几号卡情况：
    python train.py --device=Ascend 

    2、指定使用几号卡时训练情况：
    python train.py --device=Ascend --set_device=True --device_id=0
    (device_id 为卡号)


GPU V100训练：
    1、不指定使用几号卡情况：
    python train.py --device=GPU

    2、指定使用几号卡时训练情况：
    python train.py --device=GPU --set_device=True --device_id=0
    (device_id 为卡号)