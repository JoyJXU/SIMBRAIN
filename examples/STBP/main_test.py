import os
import sys
import platform
import argparse
import warnings
import numpy as np
import torch
from torch.backends import cudnn
from torch import nn
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam, AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from copy import deepcopy
from spikingjelly.clock_driven import encoding, functional

root_dir = ""
while not os.path.isdir(os.path.join(root_dir, 'bpsr')):
    root_dir = os.path.join(root_dir, '..')
sys.path.append(root_dir)
from bpsr.functional import bpsr_init
from dataset.mit_bih import MIT_BIH
from bpsr.monitor import Monitor
from bpsr import prune
from bpsr.loss import LossScaleEstimator
from bpsr.quantization import quantizate, Quantizer
from model import MITBIH_MLP

parser = argparse.ArgumentParser()
parser.add_argument('-device', default='cuda:0', type=str, help='Device, e.g., "cpu" or "cuda:0"')
parser.add_argument('-b', '-batch_size', default=32, type=int, help='Batch size, e.g., "32"', dest='batch_size')
parser.add_argument('-T', '-timesteps', default=40, type=int, help='Timesteps, e.g., "10"', dest='T')
parser.add_argument('-N', '-epoch', default=100, type=int, help='Training epoch, e.g., "100"', dest='epoch')
parser.add_argument('-lr', default=1e-3, type=float, help='Learning rate, e.g., "1e-3"')
parser.add_argument('-min_lr', default=1e-7, type=float, help='Convergent learning rate, e.g., "1e-6"')
parser.add_argument('-deterministic', default=False, action='store_true', help='Fix all random seed')
parser.add_argument('-clamp', default=64, type=int, help='Multi-spike clamp, e.g., "16"')
parser.add_argument('-ls', default=0., type=float, help='Synaptic operation regularization, e.g., "1."')
parser.add_argument('-cl', default=5, type=int, help='Number of classes, e.g., "5" or "18"')
args = parser.parse_args()

device = args.device
if device == 'cpu' or not torch.cuda.is_available():
    device = 'cpu'
    if not torch.cuda.is_available():
        warnings.warn('CUDA is not available')
if not device == 'cpu' and not args.deterministic:
    cudnn.benchmark = True
device = torch.device(device)
num_workers = 0 if platform.system() == 'Windows' else 8
if args.deterministic:
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

dataset = MIT_BIH(os.path.join(root_dir, 'dataset'), T=args.T, cl=args.cl, ds=0.1, remove_paced=True, seed=42)
trainloader = DataLoader(Subset(dataset, dataset.train_idx),
                         batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers = num_workers)
testloader = DataLoader(Subset(dataset, dataset.test_idx),
                        batch_size=args.batch_size, shuffle=False, drop_last=False, pin_memory=True, num_workers=num_workers)

model_size = (256, 96, 18) if args.cl == 18 else (192, 64, 5)
model = MITBIH_MLP(model_size).to(device)
bpsr_init(model, fr=0.4)
if args.ls > 0:
    sparse_scheduler = LossScaleEstimator(lambd=args.ls)
    prune_module = [model.fc1.synapse, model.fc2.synapse, model.fc3.synapse]
    for module_ in prune_module:
        prune.bpsr_prune(module_, 'weight')
crossEntropy = nn.CrossEntropyLoss()

quant_kw = dict(param_quant=Quantizer(1, 6, 4), v_quant=Quantizer(1, 8, 4),
                tau_quant=Quantizer(0, 3, 2))
quantizate(model, **quant_kw)

optimizer = AdamW(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.1, patience=10, threshold=0.05, threshold_mode='abs', verbose=True)

dataname = 'mitbih' + str(args.cl)
comment = '_l' + str(args.ls) + 'variation'
monitor = Monitor(model)
monitor.enable()

out_root = 'Hu_SNN_variation_results.txt'


# tensorboard --logdir "logs"
with SummaryWriter(log_dir=os.path.join('logs', dataname, model.__class__.__name__ + comment)) as writer:
    best_metric_dict = dict(best_acc=0)
    for epoch in range(args.epoch):
        if optimizer.param_groups[0]['lr'] < args.min_lr:
            break
        print('Epoch:', epoch)

        correct, total, fr, sop, dop, loss_sum = 0, 0, 0, 0, 0, 0
        model.train()
        for X, Y in tqdm(trainloader, ncols=80):
            X, Y = X.to(device, non_blocking=True), Y.to(device, non_blocking=True)
            X += torch.randn_like(X) * 0.05
            X = torch.clamp(X, max=args.clamp).transpose(0, 1)
            pred = model(X)
            correct += float((pred.argmax(1) == Y).float().sum())
            total += float(len(Y))
            fr += float(monitor.get_spikes() / monitor.get_neuron_num() / args.T)
            sop += float(monitor.get_synaptic_ops())
            dop += float(monitor.get_neuron_num() * args.T * args.batch_size - monitor.get_spikes())
            loss = sparse_scheduler(crossEntropy(pred * 10, Y), monitor.get_synaptic_ops()) \
                if args.ls > 0 else crossEntropy(pred * 10, Y)
            loss_sum += float(loss)
            if epoch > 0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            functional.reset_net(model)
            monitor.reset()
        train_acc = correct / total * 100
        train_fr = fr / total * 100
        train_sop = sop / total
        train_dop = dop / total
        loss_sum = loss_sum / total
        print(f'Training Acc: {train_acc:.3f}%, Train FR: {train_fr:.3f}%, Train SOP: {train_sop}, Train DOP: {train_dop}')

        correct, total, fr, sop, dop = 0, 0, 0, 0, 0
        model.eval()
        with torch.no_grad():
            for X, Y in testloader:
                X, Y = X.to(device, non_blocking=True), Y.to(device, non_blocking=True)
                X = torch.clamp(X, max=args.clamp).transpose(0, 1)
                pred = model(X)
                correct += float((pred.argmax(1) == Y).float().sum())
                total += float(len(Y))
                fr += float(monitor.get_spikes() / monitor.get_neuron_num() / args.T)
                sop += float(monitor.get_synaptic_ops())
                dop += float(monitor.get_neuron_num() * args.T * args.batch_size - monitor.get_spikes())
                functional.reset_net(model)
                monitor.reset()
        test_acc = correct / total * 100
        test_fr = fr / total * 100
        test_sop = sop / total
        test_dop = dop / total
        print(f'Test Acc: {test_acc:.3f}%, Test FR: {test_fr:.3f}%, Test SOP: {test_sop}, Test DOP: {test_dop}')
        print(f'Synapses: {float(monitor.get_synapses())}')
        
        out = open(out_root, 'a')
        out_txt = 'Epoch:' + str(epoch) + '\tAccuracy:' + str(test_acc) + '\n'
        out.write(out_txt)
        out.close()

        writer.add_scalar('Loss', loss_sum, global_step=epoch)
        writer.add_scalar('Lr', optimizer.param_groups[0]['lr'], global_step=epoch)
        writer.add_scalars('Acc', {'train': train_acc, 'test': test_acc}, global_step=epoch)
        writer.add_scalars('Fr', {'train': train_fr, 'test': test_fr}, global_step=epoch)
        writer.add_scalars('SOP', {'train': train_sop, 'test': test_sop}, global_step=epoch)
        writer.add_scalars('DOP', {'train': train_dop, 'test': test_dop}, global_step=epoch)

        if train_acc > 80:
            scheduler.step(train_acc)
        if args.ls > 0:
            sparse_scheduler.step(train_acc > 80)

        if epoch > 50 and train_acc < 50:
            break

        if test_acc > best_metric_dict['best_acc']:
            best_metric_dict = dict(best_acc=test_acc, best_fr=test_fr,
                                    best_sop=test_sop, best_dop=test_dop,
                                    best_syn=float(monitor.get_synapses()))
            best_model = deepcopy(model.state_dict())

    print('Training finished, ', best_metric_dict)
    writer.add_hparams(vars(args), best_metric_dict)
    if not os.path.isdir(os.path.join('saved_models', dataname)):
        os.makedirs(os.path.join('saved_models', dataname))
    torch.save(best_model, os.path.join('saved_models', dataname, model.__class__.__name__ + comment + '.pth'))
