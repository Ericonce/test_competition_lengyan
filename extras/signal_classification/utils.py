import numpy as np
import models

# import scipy.io as scio
# import matplotlib.pyplot as plt

from functools import reduce

import torch
# from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use('Agg')

def to_onehot(yy):
    yy1 = np.zeros([len(yy), max(yy) + 1])
    yy1[np.arange(len(yy)), yy] = 1
    return yy1


# def load_data(data_set, ratio, seed, mod='train', task='', SNR='2'):
#     np.random.seed(seed)
#     if data_set == 'cemee':
#         if mod == 'train':
#             data = h5py.File('/media/yc/rushb/data/cemee/Task_1_Train.mat', 'r')
#             f = list(data.keys())

#             x = data[f[0]][()].transpose((2, 0, 1))
#             x = (x - np.min(x)) / (np.max(x) - np.min(x))
#             y = data[f[1]][()].squeeze()

#             data, label = [], []
#             length, step = 600, 150
#             for i in range(x.shape[0]):
#                 for j in range(int((x.shape[2]-length)/step)):
#                     data.append(x[i:i+1, :, j*step:length+j*step])
#                     label.append([i, int(y[i])])
#             x, y = np.vstack(data), np.array(label)

#             X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.8)
#             X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5)

#             if ratio >= .8:
#                 return X_train, X_val, X_test, y_train, y_val, y_test, np.unique(y[:, 1])
#             else:
#                 if task != 'semi':
#                     X_train, _, y_train, _ = train_test_split(X_train, y_train, train_size=ratio * 1.25)
#                     return X_train, X_val, X_test, y_train, y_val, y_test, np.unique(y[:, 1])
#                 else:
#                     X_train, X_train_semi, y_train, y_train_semi = train_test_split(X_train, y_train, train_size=ratio*1.25)
#                     return X_train, X_train_semi, X_val, X_test, y_train, y_train_semi, y_val, y_test, np.unique(y[:, 1])

#         elif mod == 'test':
#             data = h5py.File('/media/yc/rushb/data/cemee/Task_1_Test.mat', 'r')
#             f = list(data.keys())

#             x = data[f[0]][()].transpose((2, 0, 1))
#             x = (x - np.min(x)) / (np.max(x) - np.min(x))
#             y = data[f[1]][()].squeeze()

#             data, label = [], []
#             length, step = 600, 150
#             for i in range(x.shape[0]):
#                 for j in range(int((x.shape[2] - length) / step)):
#                     data.append(x[i:i + 1, :, j * step:length + j * step])
#                     label.append([i, int(y[i])])
#             x, y = np.vstack(data), np.array(label)

#             return [], [], x, [], [], y, np.unique(y[:, 1])
#     else:
#         raise ValueError("Invalid dataset name!")


def _backend_available(backend: str) -> bool:
    backend = backend.lower()
    if backend == "npu":
        try:
            import torch_npu  # noqa: F401
        except ImportError:
            pass
        return hasattr(torch, "npu") and torch.npu.is_available()
    if backend == "cuda":
        return torch.cuda.is_available()
    return True


def resolve_device(requested: str = "auto", device_type: str = "auto") -> torch.device:
    """Resolve cpu/cuda/Ascend NPU device from CLI-friendly values."""
    requested = str(requested or "auto").strip().lower()
    device_type = str(device_type or "auto").strip().lower()
    if requested in {"cpu", "cuda", "npu"}:
        backend = requested if requested != "cpu" else "cpu"
        index = 0
    elif ":" in requested:
        backend, index_text = requested.split(":", 1)
        index = int(index_text)
    elif requested.isdigit():
        backend = device_type if device_type != "auto" else (
            "npu" if _backend_available("npu") else "cuda" if _backend_available("cuda") else "cpu"
        )
        index = int(requested)
    else:
        backend = device_type if device_type != "auto" else (
            "npu" if _backend_available("npu") else "cuda" if _backend_available("cuda") else "cpu"
        )
        index = 0

    if backend not in {"cpu", "cuda", "npu"}:
        raise ValueError(f"Unsupported device: {requested}")
    if backend != "cpu" and not _backend_available(backend):
        raise RuntimeError(f"Requested {backend}:{index}, but this backend is unavailable")
    device = torch.device(backend if backend == "cpu" else f"{backend}:{index}")
    if backend == "npu" and hasattr(torch, "npu"):
        torch.npu.set_device(device)
    return device

def load_model(model_name, class_num=100, in_channel=1, dropout=.5, length=128, device=None):
    """Build a registered model and move it to the requested accelerator.

    ``length`` is used by sequence models whose convolution kernel depends on
    the input length. The legacy CNN keeps its original fixed-size head.
    """
    name = str(model_name).strip().lower()

    if name in {"resnet18", "resnet-18"}:
        model = models.ResNet18(class_num, in_channel)
    elif name in {"cnn", "convnet"}:
        model = models.CNN(class_num, in_channel)
    elif name in {"efficientnet", "efficientnetb0", "efficientnet-b0"}:
        model = models.EfficientNetB0(class_num, in_channel)
    elif name in {"cnn1d", "multigrain_cnn", "multigraincnn"}:
        model = models.MultiGrainCNN(
            {"in_channels": in_channel, "dropout": dropout},
            num_classes=class_num,
        )
    elif name in {"atenet", "atenet1d"}:
        model = models.ATENetClassifier(
            {"in_channels": in_channel, "dropout": dropout},
            num_classes=class_num,
        )
    elif name in {"invconv", "invconv1d"}:
        model = models.InvConvClassifier(
            {
                "in_channels": in_channel,
                "dropout": dropout,
                "sequence_length": length,
            },
            num_classes=class_num,
        )
    elif name in {"mambasl", "mamba_sl", "mamba"}:
        model = models.MambaSLClassifier(
            {"in_channels": in_channel, "dropout": dropout},
            num_classes=class_num,
        )
    elif name in {"modern_tcn", "moderntcn"}:
        model = models.ModernTCNClassifier(
            {"in_channels": in_channel, "dropout": dropout},
            num_classes=class_num,
        )
    elif name in {"mptsnet", "mptsnet1d"}:
        model = models.MPTSNetClassifier(
            {"in_channels": in_channel, "dropout": dropout},
            num_classes=class_num,
        )
    else:
        available = (
            "ResNet18, CNN, EfficientNetB0, MultiGrainCNN, ATENet, "
            "InvConv, MambaSL, ModernTCN, MPTSNet"
        )
        raise ValueError(f"Invalid model name {model_name!r}; available models: {available}")
    target = resolve_device() if device is None else torch.device(device)
    return model.to(target)
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    '''
    :param unknown_idx_bank_current: 当前epoch检出的待确定未知样本
    :param unknown_idx_bank: 未知样本索引队列
    :param idx_bank: 未知样本库
    :param test_true_class_labels: 所有未知样本的标签
    :param T: 超参数，连续T次被检测为未知样本
    :param writer: tensorboard
    :param epoch: 当前epoch
    :return:
    '''
    unknown_idx_bank_current = list(unknown_idx_bank_current)
    idx_current = []
    if epoch + 1 > class_epoch:
        if len(unknown_idx_bank_current) == 0:
            if_restruct = False
            print('未发现新样本')
            pass
        else:
            if len(unknown_idx_bank) + 1 < T:
                unknown_idx_bank.append(unknown_idx_bank_current)
                print(f'发现{len(unknown_idx_bank_current)}个新样本')
                if_restruct = False
            elif len(unknown_idx_bank) + 1 == T:
                unknown_idx_bank.append(unknown_idx_bank_current)
                idx_current = reduce(np.intersect1d, [unknown_idx_bank[i] for i in range(T)])
                idx_current = np.setdiff1d(idx_current, idx_bank)
                for i in idx_current:
                    idx_bank.append(i)
                if len(idx_current) != 0:
                    print(f'发现{len(unknown_idx_bank_current)}个新样本，其中{len(idx_current)}个第一次检测到，重构训练集')
                    if_restruct = True
                    writer.add_scalar('current_bank_num', len(idx_current), epoch)
                    writer.add_scalar('current_bank_acc',
                                      np.where(np.array(test_true_class_labels)[idx_current] == 3)[0].shape[0] / len(
                                          idx_current), epoch)
                    writer.add_scalar('bank_acc',
                                      np.where(np.array(test_true_class_labels)[idx_bank] == 3)[0].shape[0] / len(idx_bank),
                                      epoch)
                    writer.add_scalar('bank_num', len(idx_bank), epoch)
                else:
                    print(f'发现{len(unknown_idx_bank_current)}个新样本，其中{len(idx_current)}个第一次检测到，不重构训练集')
                    if_restruct = False

            elif len(unknown_idx_bank) == T:
                unknown_idx_bank.pop(0)
                unknown_idx_bank.append(unknown_idx_bank_current)
                #  连续T次被检测为未知样本的加入银行中
                idx_current = reduce(np.intersect1d, [unknown_idx_bank[i] for i in range(len(unknown_idx_bank))])
                idx_current = np.setdiff1d(idx_current, idx_bank)
                new_num = idx_current.shape[0]
                for i in idx_current:
                    idx_bank.append(i)
                if new_num != 0:
                    print(f'发现{len(unknown_idx_bank_current)}个新样本，其中{new_num}个第一次检测到，重构训练集')
                    if_restruct = True
                    writer.add_scalar('current_bank_num', len(idx_current), epoch)
                    writer.add_scalar('current_bank_acc',
                                      np.where(np.array(test_true_class_labels)[idx_current] >= 3)[0].shape[0] / len(
                                          idx_current), epoch)
                    writer.add_scalar('bank_acc',
                                      np.where(np.array(test_true_class_labels)[idx_bank] >= 3)[0].shape[0] / len(idx_bank),
                                      epoch)
                    writer.add_scalar('bank_num', len(idx_bank), epoch)
                else:
                    print(f'发现{len(unknown_idx_bank_current)}个新样本，其中{new_num}个第一次检测到，不重构训练集')
                    if_restruct = False
    else:
        if_restruct = False

    return idx_current, unknown_idx_bank, idx_bank, if_restruct