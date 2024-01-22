import os
import itertools
import numpy as np
import torch
from bpsr.encoding import LCSampler
from typing import Optional, Literal
from math import floor


class MIT_BIH(torch.utils.data.Dataset):
    def __init__(self, root: str, T: Literal[5, 18], cl: int = 18, delta: float = 0.1, ds: float = 0.1,
                 window: Literal['floating', 'fixed'] = 'floating',
                 mlii: bool = False, norm: bool = False, remove_paced: bool = False,
                 align: Literal['left', 'center'] = 'left', fold: Optional[int] = None,
                 test_ratio: float = 0.2, seed: Optional[int] = None,
                 print_info: bool = False, plot_hist: bool = False):
        """
        经过LC采样后的MIT-BIH心电信号异常检测数据集。

        :param T: 心电序列时间长度，会进行截断或填充处理，如设置为None则直接输出固有长度
        :type T: int, None

        :param cl: 异常检测分类数量，可选值包括5/18
        :type cl: int

        :param delta: LC采样的delta值（阈值）
        :type delta: float

        :param ds: 原心电信号的降采样比例，取值范围(0,1)
        :type ds: float

        :param window: LC采样方法，包括浮动窗口/固定，即'floating'或'fixed'
        :type window: str

        :param mlii: 是否只保留心电信号中的MLII导联
        :type mlii: bool

        :param norm: 是否对每个导联进行归一化处理，限制其最大/最小值为±1
        :type norm: bool

        :param remove_paced: 是否去除平静信号
        :type remove_paced: bool

        :param align: 时间序列进行截断或填充时的对齐方式，包括中心对齐/左对齐，即'center'或‘left’。T设置为None时该项无效
        :type align: str

        :param fold: 进行折叠的次数，设置为None时不进行折叠。仅在T不为None时有效
        :type fold: int, None

        :param test_ratio: 训练集与测试集的划分比例，取值范围(0,1)
        :type test_ratio: float

        :param seed: 划分数据集时的随机种子，使数据集可复现。为None时不进行种子固定
        :type seed: int, None

        :param print_info: 打印数据集信息，包括：分类标签、每类样本数量、源文件名称、LC采样的MSE误差
        :type print_info: bool

        :param plot_hist: 绘制所有样本时间长度的频数分布直方图
        :type plot_hist: bool
        """

        assert T is None or isinstance(T, int)
        assert cl in [5, 18], 'Illegal categories number'
        assert isinstance(delta, float)
        assert isinstance(ds, float) and 1. >= ds > 0.
        assert isinstance(window, str) and window in ['floating', 'fixed']
        assert isinstance(mlii, bool)
        assert isinstance(norm, bool)
        assert isinstance(remove_paced, bool)
        assert isinstance(align, str) and align in ['left', 'center']
        assert fold is None or (isinstance(fold, int) and fold > 0 and T is not None)
        assert isinstance(test_ratio, float) and 1. > test_ratio > 0.
        assert seed is None or isinstance(seed, int)
        assert isinstance(print_info, bool)
        assert isinstance(plot_hist, bool)

        file_name = f'class{cl}_delta{delta}_ds{ds}_{window}'
        if mlii:
            file_name += '_mlii'
        if norm:
            file_name += '_norm'
        if remove_paced:
            file_name += '_nopaced'
        if not os.path.exists(os.path.join(root, 'MIT-BIH', 'processed', file_name + '.npz')):
            if not os.path.exists(os.path.join(root, 'MIT-BIH', 'mit-bih-arrhythmia-database-1.0.0')):
                if not os.path.exists(os.path.join(root, 'MIT-BIH', 'mit-bih-arrhythmia-database-1.0.0.zip')):
                    from torch.hub import download_url_to_file
                    try:
                        print(f'Download "mit-bih-arrhythmia-database-1.0.0"')
                        download_url_to_file('https://physionet.org/static/published-projects/mitdb/mit-bih-arrhythmia-database-1.0.0.zip',
                                             os.path.join(root, 'MIT-BIH', 'mit-bih-arrhythmia-database-1.0.0.zip'))
                    except:
                        raise NotImplementedError(
                            'Cannot download MIT-BIH Arrhythmia Database, '
                            'please download "mit-bih-arrhythmia-database-1.0.0.zip" '
                            'from "https://physionet.org/content/mitdb/1.0.0/" manually '
                            'and put files at "{}".'.format(os.path.join(root, 'MIT-BIH')))
                import zipfile
                with zipfile.ZipFile(os.path.join(root, 'MIT-BIH', 'mit-bih-arrhythmia-database-1.0.0.zip')) as zf:
                    zf.extractall(path=os.path.join(root, 'MIT-BIH'))

            print('LC sampling ...')
            create_np_file(root, cl, delta, ds, window, mlii, norm, remove_paced)

        file = np.load(os.path.join(root, 'MIT-BIH', 'processed', file_name + '.npz'),
                       allow_pickle=True)
        data = file['data']
        sample = file['sample']
        self.label = file['label']
        self.symbol = file['symbol']
        self.mse = file['mse']
        self.T = T

        if print_info:
            print('Number of class:')
            for n in range(len(self.symbol)):
                print(self.symbol[n], (self.label == n).sum())
            print('{} with clamped LC has MSE error {}'.format(file_name + '.npz', file['mse']))
        if plot_hist:
            import matplotlib.pyplot as plt
            length = sample[2, :] - sample[0, :]
            fig, ax = plt.subplots(1, 1)
            ax.hist(length, density=True)
            plt.xlabel('Length of Sample')
            plt.ylabel('Frequency')
            plt.title('Histogram of sample length')
            plt.show()

        print('Organizing data...')
        data = torch.from_numpy(data)
        sample = torch.from_numpy(sample)
        L, C, R = sample[0, :], sample[1, :], sample[2, :]
        if T is not None:
            self.signal = torch.zeros(T, len(self.label), data.shape[1], device=data.device, dtype=data.dtype)
            for i in range(len(self.label)):
                if align == 'center':
                    if C[i] - L[i] >= T // 2:
                        signal_l = 0
                        data_l = C[i] - T // 2
                    else:
                        signal_l = T // 2 - (C[i] - L[i])
                        data_l = L[i]
                    if R[i] - C[i] >= T - T // 2:
                        signal_r = T
                        data_r = C[i] + T - T // 2
                    else:
                        signal_r = T // 2 + (R[i] - C[i])
                        data_r = R[i]
                else:
                    signal_l, data_l = 0, L[i]
                    if R[i] - L[i] > T:
                        signal_r = T
                        data_r = L[i] + T
                    else:
                        signal_r = R[i] - L[i]
                        data_r = R[i]
                self.signal[signal_l:signal_r, i, :] = data[data_l:data_r, :]
            self.signal = self.signal.repeat_interleave(2, dim=2)
            self.signal[..., 1::2] = -self.signal[..., 1::2]
            self.signal.clamp_(min=0.)
        else:
            self.signal = []
            for i in range(len(self.label)):
                signal_ = data[L[i]:R[i], :].repeat_interleave(2, dim=1)
                signal_[:, 1::2] = -signal_[:, 1::2]
                self.signal.append(signal_.clamp(min=0.))

        if fold is not None:
            self.signal = self.signal.permute(2, 0, 1).flatten(end_dim=1)
            fold_length = floor(self.signal.shape[0] / (fold + 1) * 2)
            signal_ = torch.zeros((fold, fold_length) + self.signal.shape[1:], device=self.signal.device, dtype=self.signal.dtype)
            for i in range(fold):
                signal_[i, ...] = self.signal[int(i * (fold_length / 2)):int(i * (fold_length / 2)) + fold_length, ...]
            self.signal = signal_.transpose(1, 2)

        if seed is not None:
            local_seed = np.random.RandomState(seed)
        self.test_idx = []
        label_unique, counts = np.unique(self.label, return_counts=True)
        for label_, num_ in zip(label_unique, counts):
            self.test_idx.extend(
                np.where(self.label == label_)[0][
                    np.random.choice(num_, int(round(num_ * test_ratio)), replace=False)]
                if seed is None else
                np.where(self.label == label_)[0][
                    local_seed.choice(num_, int(round(num_ * test_ratio)), replace=False)]
            )
        self.train_idx = list(set([i for i in range(len(self.label))]).difference(set(self.test_idx)))
        self.test_idx = sorted(self.test_idx)
        self.label = torch.from_numpy(self.label)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        return self.signal[item] if self.T is None else self.signal[:, item, :], \
               self.label[item]


def create_np_file(root: str, cl: int, delta: float, ds: float, window: str, mlii: bool, norm: bool, remove_paced: bool):
    import wfdb
    # label symbol in MIT-BIH
    # '!', '"', '+', '/', 'A', 'E', 'F', 'J', 'L', 'N', 'Q', 'R', 'S', 'V', '[', ']', 'a', 'e', 'f', 'j', 'x', '|', '~'
    nonbeat_symbol = ['"', '+', "~"]
    paced_record = ['102', '104', '107', '217']
    if cl == 18:
        class_symbol = ['N', 'L', 'R', 'A', 'a', 'J', 'S', 'V', 'F', '!', 'e', 'j', 'E', '/', 'f', 'x', 'Q', '|']
    elif cl == 5:
        class_symbol = [
            ['N', 'L', 'R', 'e', 'j'],
            ['A', 'a', 'x', 'J'],
            ['V', 'E', '!'],
            ['F'],
            ['f', 'Q'],
        ]

    lc_sampler = LCSampler(delta=delta, window=window)
    data, label, sample, error = np.zeros(shape=(0, 1 if mlii else 2)), [], [], []
    with open(os.path.join(root, 'MIT-BIH', 'mit-bih-arrhythmia-database-1.0.0', 'RECORDS'), 'r') as f:
        for record in f.read().rstrip('\n').split('\n'):
            if remove_paced and record in paced_record:
                continue
            signal, fields = wfdb.rdsamp(os.path.join(root, 'MIT-BIH', 'mit-bih-arrhythmia-database-1.0.0', record))
            annotation = wfdb.rdann(os.path.join(root, 'MIT-BIH', 'mit-bih-arrhythmia-database-1.0.0', record), 'atr')
            if ds < 1:
                x = np.arange(0, signal.shape[0], 1)
                x_scale = np.arange(0, signal.shape[0], 1 / ds)
                signal = np.vstack((np.interp(x_scale, x, signal[:, 0]), np.interp(x_scale, x, signal[:, 1]))).T
            if mlii:
                if 'MLII' in fields['sig_name']:
                    signal = signal[:, [n == 'MLII' for n in fields['sig_name']]]
                else:
                    break
            else:
                if fields['sig_name'][1] == 'MLII':
                    signal = signal[:, -1::-1]
            if norm:
                for n in range(signal.shape[1]):
                    signal[:, n] = signal[:, n] / np.abs(signal[:, n]).max()
            signal_lc = lc_sampler.encode(signal)
            signal_lc_clamp = np.clip(signal_lc, a_min=-1., a_max=1.)
            error.append(np.sqrt(((lc_sampler.decode(signal_lc_clamp) - signal) ** 2).mean()))

            label_new = []
            mask_ = np.zeros_like(annotation.symbol, dtype=np.bool_)
            for n, symbol_ in enumerate(annotation.symbol):
                if isinstance(class_symbol[0], str) and symbol_ in class_symbol:
                    mask_[n] = True
                    label_new.append(class_symbol.index(symbol_))
                elif isinstance(class_symbol[0], list) and symbol_ in list(itertools.chain(*class_symbol)):
                    mask_[n] = True
                    label_new.append([symbol_ in tmp for tmp in class_symbol].index(True))
            label.extend(label_new[1:-1])
            sample_new = annotation.sample[mask_]
            sample_point = (sample_new[:-1] + sample_new[1:]) / 2
            sample_point = np.vstack((
                np.floor(sample_point[:-1] * ds),
                np.round(sample_new[1:-1] * ds),
                np.ceil(sample_point[1:] * ds)))
            sample_point = np.clip(sample_point, a_min=0., a_max=len(signal)) + len(data)
            sample.append(sample_point)
            data = np.vstack((data, signal_lc))

    data = data.astype(np.float32)
    sample = np.hstack(sample).astype(np.int64)
    label = np.hstack(label).astype(np.int64)

    if not os.path.isdir(os.path.join(root, 'MIT-BIH', 'processed')):
        os.mkdir(os.path.join(root, 'MIT-BIH', 'processed'))
    file_name = f'class{cl}_delta{delta}_ds{ds}_{window}'
    if mlii:
        file_name += '_mlii'
    if norm:
        file_name += '_norm'
    if remove_paced:
        file_name += '_nopaced'
    class_symbol = np.array(class_symbol, dtype=object)
    np.savez(os.path.join(root, 'MIT-BIH', 'processed', file_name + '.npz'),
             data=data, sample=sample, label=label, symbol=class_symbol, mse=sum(error) / len(error))


if __name__ == '__main__':
    from torch.utils.data import DataLoader, Subset
    from torch.nn.utils.rnn import pad_sequence
    from tqdm import tqdm
    import argparse


    def collate_func(batch_data):
        batch_data.sort(key=lambda _: len(_[0]), reverse=True)
        data, label = [], []
        for data_, label_ in batch_data:
            data.append(data_)
            label.append(label_)
        return pad_sequence(data), label


    parser = argparse.ArgumentParser()
    parser.add_argument('-profile', default=False, action='store_true', help='Profile runtime')
    args = parser.parse_args()

    if args.profile:
        from line_profiler import LineProfiler

        lp1 = LineProfiler()
        lp_wrapper1 = lp1(create_np_file)
        lp_wrapper1('.', 18, 0.1, 0.1, 'floating', False, True)
        lp1.print_stats()

        lp2 = LineProfiler()
        lc = LCSampler(delta=0.1)
        lp_wrapper2 = lp2(lc.encode)
        lp_wrapper2(np.random.randn(35000, 2))
        lp2.print_stats()
    else:
        for T_, cl_, window_, mlii_, align_, fold_, seed_ in tqdm(itertools.product(
                [None, 15, 30],
                [5, 18],
                ['floating', 'fixed'],
                [True, False],
                ['center', 'left'],
                [None, 3],
                [42, None])):
            if isinstance(fold_, int) and T_ is None:
                continue
            dataset = MIT_BIH(root='.', T=T_, cl=cl_, window=window_, mlii=mlii_, align=align_, fold=fold_, seed=seed_)
            collate_fn = None if T_ is not None else collate_func
            trainloader = DataLoader(Subset(dataset, dataset.train_idx), batch_size=32, collate_fn=collate_fn, shuffle=True)
            testloader = DataLoader(Subset(dataset, dataset.test_idx), batch_size=32, collate_fn=collate_fn)
            for X, Y in trainloader:
                pass
            for X, Y in testloader:
                pass
        print('test pass')
