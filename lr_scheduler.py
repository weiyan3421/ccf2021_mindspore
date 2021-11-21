import math
import numpy as np

def linear_warmup_learning_rate(current_step, warmup_steps, base_lr, init_lr):
    lr_inc = (float(base_lr) - float(init_lr)) / float(warmup_steps)
    learning_rate = float(init_lr) + lr_inc * current_step
    return learning_rate


def a_cosine_learning_rate(current_step, base_lr, warmup_steps, decay_steps):
    base = float(current_step - warmup_steps) / float(decay_steps)
    learning_rate = (1 + math.cos(base * math.pi)) / 2 * base_lr
    return learning_rate


def cosine_learning_rate(current_step, base_lr, Tmax_steps):
    base = float(current_step) / float(Tmax_steps)
    learning_rate = (1 + math.cos(base * math.pi)) / 2 * base_lr
    return learning_rate


def CosineWarmupRestart(args, steps_per_epoch, T_max):
    """dynamic learning rate generator"""
    base_lr = args.lr
    warmup_steps = steps_per_epoch * args.warmup_epoch
    Tmax_steps = steps_per_epoch * T_max
    T_num = args.total_epochs // T_max
    lr_T = []
    for i in range(Tmax_steps):
        lr_T.append(cosine_learning_rate(i, base_lr, Tmax_steps))
    lr = []
    for i in range(warmup_steps):
        lr.append(linear_warmup_learning_rate(i, warmup_steps, base_lr, base_lr * args.warmup_ratio))
    lr.extend(lr_T * T_num)
    return lr


def CosineRestart(args, steps_per_epoch, T_max):
    """dynamic learning rate generator"""
    base_lr = args.lr
    Tmax_steps = steps_per_epoch * T_max
    T_num = args.total_epochs // T_max
    lr_T = []
    for i in range(Tmax_steps):
        lr_T.append(cosine_learning_rate(i, base_lr, Tmax_steps))
    return lr_T*T_num


def CosineWarmupRestart_2(args, steps_per_epoch, T_max):
    """dynamic learning rate generator"""
    base_lr = args.lr
    warmup_steps = steps_per_epoch * args.warmup_epoch
    Tmax_steps = steps_per_epoch * T_max
    T_num = args.total_epochs // T_max
    ratio = np.linspace(1, 0.3, num=T_num, endpoint=True, retstep=False, dtype=float)
    lr = []
    for i in range(warmup_steps):
        lr.append(linear_warmup_learning_rate(i, warmup_steps, base_lr, base_lr * args.warmup_ratio))

    for n in range(T_num):
        for i in range(Tmax_steps):
            lr.append(cosine_learning_rate(i, base_lr*ratio[n], Tmax_steps))
    return lr
