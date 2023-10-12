import os
import sys
import platform
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import math

from torchvision import transforms
from tqdm import tqdm

from time import time as t

sys.path.append('/home/HDDx2/zhangxin/mnist_training/')

from bindsnet.datasets import MNIST
from bindsnet.encoding import PoissonEncoder, poisson
from bindsnet.models import IncreasingInhibitionNetwork
from bindsnet.network.monitors import Monitor
from bindsnet.utils import get_square_weights, get_square_assignments
from bindsnet.evaluation import all_activity, proportion_weighting, assign_labels
from bindsnet.analysis.plotting import (
    plot_input,
    plot_spikes,
    plot_weights,
    plot_assignments,
    plot_performance,
    plot_voltages,
)

parser = argparse.ArgumentParser()#messy code
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_neurons", type=int, default=625)#625  10
parser.add_argument("--batch_size", type=int, default=1) #128
parser.add_argument("--n_epochs", type=int, default=1)
parser.add_argument("--n_test", type=int, default=10000)#10000  1
parser.add_argument("--n_train", type=int, default=1000)#60000  10
parser.add_argument("--n_workers", type=int, default=4)
parser.add_argument("--theta_plus", type=float, default=0.05)
parser.add_argument("--time", type=int, default=250)#250  10
parser.add_argument("--dt", type=int, default=1.0)#1.0
parser.add_argument("--intensity", type=float, default=64)
parser.add_argument("--progress_interval", type=int, default=10)
parser.add_argument("--update_interval", type=int, default=20)
parser.add_argument("--update_inhibation_weights", type=int, default=500)
parser.add_argument("--plot_interval", type=int, default=250)
parser.add_argument("--plot", dest="plot", action="store_true")
parser.add_argument("--gpu", dest="gpu", action="store_true", default='gpu')#messy code
parser.set_defaults(plot=False, gpu=True)

args = parser.parse_args()

seed = args.seed
n_neurons = args.n_neurons
batch_size = args.batch_size
n_epochs = args.n_epochs
n_test = math.ceil(args.n_test // batch_size)
n_train = math.ceil(args.n_train / batch_size)
n_workers = args.n_workers
theta_plus = args.theta_plus
time = args.time
dt = args.dt
intensity = args.intensity
progress_interval = args.progress_interval
plot_interval = args.plot_interval
update_interval = args.update_interval
plot = args.plot
gpu = args.gpu
update_inhibation_weights = args.update_inhibation_weights

# Sets up Gpu use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if gpu and torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
else:
    torch.manual_seed(seed)
    device = "cpu"
    if gpu:
        gpu = False

torch.set_num_threads(min(os.cpu_count() - 1, 8))
print("Running on Device = ", device)

# Determines number of workers to use
if n_workers == -1 or platform.system() == 'Windows':
    n_workers = 0  # torch.cuda.is_available() * 4 * torch.cuda.device_count()

n_sqrt = int(np.ceil(np.sqrt(n_neurons)))
start_intensity = intensity

# Build network.
network = IncreasingInhibitionNetwork(
    n_input=784,
    n_neurons=n_neurons,
    start_inhib=10,
    max_inhib=-40.0,
    theta_plus=0.05,
    tc_theta_decay=1e7, 
    nu=(0.9e-4, 0.9e-2),
    inpt_shape=(1, 28, 28),
    batch_size=batch_size,
)

network.to(device)

# Load MNIST training data.
dataset = MNIST(
    PoissonEncoder(time=time, dt=dt),
    None,
    root=os.path.join("..", "..", "data", "MNIST"),
    # root=os.path.join("/home/jiahao/project/bindsnet-master/data/MNIST"),
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
    ),
)

# Load MNIST test data.
test_dataset = MNIST(
    PoissonEncoder(time=time, dt=dt),
    None,
    root=os.path.join("..", "..", "data", "MNIST"),
    # root=os.path.join("/home/jiahao/project/bindsnet-master/data/MNIST"),
    download=True,
    train=False,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
    ),
)

# Record spikes during the simulation.
spike_record = torch.zeros((update_interval, int(time / dt), n_neurons), device=device)

# Neuron assignments and spike proportions.
n_classes = 10
assignments = -torch.ones(n_neurons, device=device)
proportions = torch.zeros((n_neurons, n_classes), device=device)
rates = torch.zeros((n_neurons, n_classes), device=device)

# Sequence of accuracy estimates.
accuracy = {"all": [], "proportion": []}

# Voltage recording for excitatory and inhibitory layers.
som_voltage_monitor = Monitor(
    network.layers["Y"], ["v"], time=int(time / dt), device=device
)
network.add_monitor(som_voltage_monitor, name="som_voltage")

# Set up monitors for spikes and voltages
spikes = {}
voltages = {}
for layer in set(network.layers):
    spikes[layer] = Monitor(
        network.layers[layer], state_vars=["s"], time=int(time / dt), device=device
    )
    network.add_monitor(spikes[layer], name="%s_spikes" % layer)
    #messy code
for layer in set(network.layers) - {"X"}:
    voltages[layer] = Monitor(
        network.layers[layer], state_vars=["v"], time=int(time / dt), device=device
    )
    network.add_monitor(voltages[layer], name="%s_voltages" % layer)

inpt_ims, inpt_axes = None, None
spike_ims, spike_axes = None, None
weights_im = None
assigns_im = None
perf_ax = None
voltage_axes, voltage_ims = None, None
save_weights_fn = "plots/weights/weights.png"
save_performance_fn = "plots/performance/performance.png"
save_assaiments_fn = "plots/assaiments/assaiments.png"

directorys = ["plots", "plots/weights", "plots/performance", "plots/assaiments"]
for directory in directorys:
    if not os.path.exists(directory):
        os.makedirs(directory)

# diagonal weights for increassing the inhibitiosn
weights_mask = (1 - torch.diag(torch.ones(n_neurons))).to(device)

# Train the network.
print("\nBegin training.\n")
start = t()

for epoch in range(n_epochs):
    labels = []

    if epoch % progress_interval == 0:
        print("Progress: %d / %d (%.4f seconds)" % (epoch, n_epochs, t() - start))
        start = t()

    # Create a dataloader to iterate and batch data
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers, pin_memory=gpu
    )

    pbar = tqdm(total=n_train)
    for step, batch in enumerate(dataloader):#messy code
        if step == n_train:
            break

        # Get next input sample.
        # inputs = {
        #     "X": batch["encoded_image"].view(int(time / dt), batch["encoded_image"].shape[0], 1, 28, 28).to(device)
        # }
        inputs = {
            "X": batch["encoded_image"].view(int(time / dt), 1, 1, 28, 28).to(device)
        }
        # print('inputs[X]')
        # print(inputs["X"])
        # pos = torch.ones_like(inputs["X"])
        # neg = torch.ones_like(inputs["X"])
        # pos = pos * 0.1
        # neg = neg * -0.153
        # inputs["X"] = inputs["X"].float()
        # inputs["X"] = torch.where(inputs["X"]==0, neg, inputs["X"])
        # inputs["X"] = torch.where(inputs["X"]==1, pos, inputs["X"])

        if step > 0:
            if step % update_inhibation_weights == 0:#update_inhibation_weights=500a
                if step % (update_inhibation_weights * 10) == 0:
                    network.Y_to_Y.w -= weights_mask * 50
                else:
                    # Inhibit the connection even more
                    # network.Y_to_Y.w -= weights_mask * network.Y_to_Y.w.abs()*0.2
                    network.Y_to_Y.w -= weights_mask * 0.5

            # TODO: this block should be modified
            # if step % update_interval == 0:  # origin
            # if step % update_interval == 0:
                # Convert the array of labels into a tensor
                # label_tensor = torch.tensor(labels, device=device)
                first_label = []
                for label_item in labels:
                    first_label.append(label_item.numpy()[0])
                
                label_tensor = torch.tensor(first_label, device=device)

                # Get network predictions.
                all_activity_pred = all_activity(
                    spikes=spike_record, assignments=assignments, n_labels=n_classes
                )
                proportion_pred = proportion_weighting(
                    spikes=spike_record,
                    assignments=assignments,
                    proportions=proportions,
                    n_labels=n_classes,
                )

                # Compute network accuracy according to available classification strategies.
                accuracy["all"].append(
                    100
                    * torch.sum(label_tensor.long() == all_activity_pred).item()
                    / len(label_tensor)
                )
                accuracy["proportion"].append(
                    100
                    * torch.sum(label_tensor.long() == proportion_pred).item()
                    / len(label_tensor)
                )

                tqdm.write(
                    "\nAll activity accuracy: %.2f (last), %.2f (average), %.2f (best)"
                    % (
                        accuracy["all"][-1],
                        np.mean(accuracy["all"]),
                        np.max(accuracy["all"]),
                    )
                )
                tqdm.write(
                    "Proportion weighting accuracy: %.2f (last), %.2f (average), %.2f"
                    " (best)\n"
                    % (
                        accuracy["proportion"][-1],
                        np.mean(accuracy["proportion"]),
                        np.max(accuracy["proportion"]),
                    )
                )

                # Assign labels to excitatory layer neurons.
                assignments, proportions, rates = assign_labels(
                    spikes=spike_record,
                    labels=label_tensor,
                    n_labels=n_classes,
                    rates=rates,
                )

                labels = []

        labels.append(batch["label"])

        temp_spikes = 0
        factor = 1.2
        for retry in range(5):
            # Run the network on the input.
            network.run(inputs=inputs, time=time, input_time_dim=1)

            # Get spikes from the network
            temp_spikes = spikes["Y"].get("s").squeeze() #???????????

            if temp_spikes.sum().sum() < 2:
                inputs["X"] *= (
                    poisson(
                        datum=factor * batch["image"].clamp(min=0),
                        dt=dt,
                        time=int(time / dt),
                    )
                    .to(device)
                    .view(int(time / dt), batch["image"].shape[0], 1, 28, 28)
                )
                factor *= factor
            else:
                break

        # Get voltage recording.
        exc_voltages = som_voltage_monitor.get("v")

        # Add to spikes recording.
        # spike_record[step % update_interval] = temp_spikes.detach().clone().cpu()
        # TODO: the following line should be modified
        if batch_size == 1:
            spike_record[step % update_interval].copy_(temp_spikes, non_blocking=True)  # origin
        else:
            spike_record[step % update_interval].copy_(temp_spikes[:, 0, :], non_blocking=True)

        # Optionally plot various simulation information.
        # if plot and step % plot_interval == 0:
            image = batch["image"].view(28, 28)
            inpt = inputs["X"].view(time, 784).sum(0).view(28, 28)
            input_exc_weights = network.connections[("X", "Y")].w
            square_weights = get_square_weights(
                input_exc_weights.view(784, n_neurons), n_sqrt, 28
            )
            square_assignments = get_square_assignments(assignments, n_sqrt)
            spikes_ = {layer: spikes[layer].get("s") for layer in spikes}
            voltages = {"Y": exc_voltages}
            inpt_axes, inpt_ims = plot_input(
                image, inpt, label=batch["label"], axes=inpt_axes, ims=inpt_ims
            )
            spike_ims, spike_axes = plot_spikes(spikes_, ims=spike_ims, axes=spike_axes)
            [weights_im, save_weights_fn] = plot_weights(
                square_weights, im=weights_im, save=save_weights_fn
            )
            assigns_im = plot_assignments(
                square_assignments, im=assigns_im, save=save_assaiments_fn
            )
            perf_ax = plot_performance(accuracy, ax=perf_ax, save=save_performance_fn)
            voltage_ims, voltage_axes = plot_voltages(
                voltages, ims=voltage_ims, axes=voltage_axes, plot_type="line"
            )
            #
            plt.pause(1e-8)

        network.reset_state_variables()  # Reset state variables.
        pbar.set_description_str("Train progress: ")
        pbar.update()

    print("Progress: %d / %d (%.4f seconds)" % (epoch + 1, n_epochs, t() - start))
print("Training complete.\n")




# Sequence of accuracy estimates.
accuracy = {"all": 0, "proportion": 0}

# # Record spikes during the simulation.
# spike_record = torch.zeros(1, int(time / dt), n_neurons, device=device)

# Test the network.
print("\nBegin testing\n")
network.train(mode=False)
start = t()

# testttttttttttttt
batch_size = 128
n_test = math.ceil(args.n_test // batch_size)

# Create a dataloader to iterate and batch data
test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers, pin_memory=gpu
)

pbar = tqdm(total=n_test)
for step, batch in enumerate(test_dataloader):
    if step >= n_test:
        break
    # Get next input sample.
    inputs = {"X": batch["encoded_image"].view(int(time / dt), batch["encoded_image"].shape[0], 1, 28, 28)}
    # inputs = {"X": batch["encoded_image"].view(int(time / dt), 1, 1, 28, 28)}
    if gpu:
        inputs = {k: v.cuda() for k, v in inputs.items()}

    # Run the network on the input.
    network.run(inputs=inputs, time=time, input_time_dim=1)
    
    # Record spikes during the simulation.
    spike_record = torch.zeros(batch["encoded_image"].shape[0], int(time / dt), n_neurons, device=device)

    # Add to spikes recording.
    if batch_size == 1:
        spike_record[0] = spikes["Y"].get("s").squeeze()
    else:
        spike_record = spikes["Y"].get("s").squeeze()
        spike_record = spike_record.permute(1,0,2)

    # Convert the array of labels into a tensor
    label_tensor = torch.tensor(batch["label"], device=device)

    # Get network predictions.
    all_activity_pred = all_activity(
        spikes=spike_record, assignments=assignments, n_labels=n_classes
    )
    proportion_pred = proportion_weighting(
        spikes=spike_record,
        assignments=assignments,
        proportions=proportions,
        n_labels=n_classes,
    )

    # Compute network accuracy according to available classification strategies.
    accuracy["all"] += float(torch.sum(label_tensor.long() == all_activity_pred).item())
    accuracy["proportion"] += float(
        torch.sum(label_tensor.long() == proportion_pred).item()
    )

    network.reset_state_variables()  # Reset state variables.
    pbar.set_description_str("Test progress: ")
    pbar.update()


print("\nAll activity accuracy: %.2f" % (accuracy["all"] / (n_test*batch_size)))
print("Proportion weighting accuracy: %.2f \n" % (accuracy["proportion"] / (n_test*batch_size)))


print("Progress: %d / %d (%.4f seconds)" % (epoch + 1, n_epochs, t() - start))
print("Testing complete.\n")
