#!/usr/bin/env python
# coding: utf-8

# Load Libraries
from os import cpu_count, path
from time import strftime
from functools import reduce as func_reduce
from operator import mul as op_mul

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as utils

from ray import tune
from ray.tune import CLIReporter
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch

# This class contains DatasetClass and several helper functions
import ClassModule as cm


################################################################################
########################### Run Configuratinos #################################
# Set the number CPUS that should be used per trial. The number
# of concurrent trials is the minimum of 6 or the number of avlaible cores
# divided by cpus_per_trial. For the search algorithm to function properly this
# upper limit is necessary.
# There should be enough memory available to load the dataset into Memory
# for each concurrent trial
# In case of training with GPU this will be limited to number of models training
# simultaneously on GPU. Fractional values are possible, i.e. 0.5 will train 2
# networks on a GPU simultaneously. GPU needs enough memory to hold all models,
# check memory consumption of model on the GPU in advance
# num_workers controls how many subprocesses for loading a dataloader will spawn
cpus_per_trial = 2
gpus_per_trial = 0.33
num_workers = 4

# -From the given searchspace num_trials configurations will be sampled.
# -num_epochs gives the maximum number of training epochs
# -grace_period controls after how many epochs trials will be terminated
# -reduction_factor controls fraction of stopped trials stopped after grace_period
# -num_random_trials is the number of random probes of the loss function
# -combine several batches into one backprop to circumvent GPU memory limit
# -set_to_none puts gradients to None instead of 0, can result in speed-up
# -pin_memory and non_blocking can increase performance when loading data from
#  cpu to gpu, set to False when training without gpu
# -use_amp sets automatic mixed precision mode, reduces memory usage and can
#  improve training speed (especially on RTX cards). But can also lead to some
#  weird behaviour in pytorch, monitor output for nan/inf loss
# -use cudnn.benchmark when you rely on convolutions and have constant input
#  shape, increases gpu memory usage on first forward pass
# -Instance noise can improve training with images
num_trials = 48
num_epochs = 50
grace_period = 5
reduction_factor = 4
num_random_trials = 16
accumulation_steps = 2
Use_Shared_Memory = True
set_to_none = True
pin_memory = True
non_blocking = True
use_amp = False
use_benchmark = True
INSTANCE_NOISE = False
###############################################################################


def get_dataloader(train_ds, val_ds, bs):
    dl_train = utils.DataLoader(train_ds, batch_size=bs, shuffle=True,
                                num_workers=num_workers, pin_memory=pin_memory)
    dl_val = utils.DataLoader(val_ds, batch_size=bs * 2, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    return dl_train, dl_val


cudnn_av = torch.backends.cudnn.is_available()

# Failsave if there is no gpu and cuda-setting are still turned on
if not torch.cuda.is_available():
    pin_memory = False
    non_blocking = False
    use_amp = False
    use_benchmark = False
    print("No CUDA-device found, all CUDA-related features turned off")

###############################################################################
############################## Network ########################################
# Define the network
# The number of neurons per layer here has been made variable, so ray can
# search for the optimal number. The number of channels in the feature
# extraction layer could also be made variable e.g.

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3,
                 padding=1, bias=False):
        super(ResidualBlock, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(True)
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding,
                      bias=False),
            nn.BatchNorm2d(out_channels)
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        residual = x
        x = self.cnn1(x)
        x = self.cnn2(x)
        x += self.shortcut(residual)
        x = nn.SiLU(True)(x)
        return x

class CNN(nn.Module):
    def __init__(self, l1, l2, l3, input_dim=(1,20,20), num_in_features=5):
        super(CNN, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=4, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(True)
        )

        self.block2 = nn.Sequential(
            nn.MaxPool2d(1, 1),
            ResidualBlock(64, 64),
            ResidualBlock(64, 64, 2)
        )

        self.block3 = nn.Sequential(
            ResidualBlock(64, 128),
            ResidualBlock(128, 128)
        )

        self.block4 = nn.Sequential(
            ResidualBlock(128, 256),
            ResidualBlock(256, 256, 2)
        )
        self.block5 = nn.Sequential(
            ResidualBlock(256, 512),
            ResidualBlock(512, 512)
        )

        self.avgpool = nn.AvgPool2d(2)

        # Gives the number of features after the conv layer
        num_features_after_conv = self.__calc_features(input_dim)

        self.flatten = nn.Flatten()

        self.dense_nn = nn.Sequential(
            nn.Linear(num_features_after_conv + num_in_features, l1),
            nn.SiLU(True),
            nn.Linear(l1, l2),
            nn.SiLU(True),
            nn.Linear(l2, l3),
            nn.SiLU(True),
            nn.Linear(l3, 3),
            nn.SiLU(True)
        )

    def __calc_features(self, input_dim):
        x = self.block1(torch.rand(1, *input_dim))
        # print(f"After block1 {x.shape}")
        x = self.block2(x)
        # print(f"After block2 {x.shape}")
        x = self.block3(x)
        # print(f"After block3 {x.shape}")
        x = self.block4(x)
        # print(f"After block4 {x.shape}")
        # x = self.block5(x)
        # print(f"After block5 {x.shape}")
        x = self.avgpool(x)
        # print(f"After pool {x.shape}")
        feat = func_reduce(op_mul, list(x.shape))
        # print(f" Features {feat}")
        return feat

    def forward(self, cluster, clusNumXYEPt):
        if torch.cuda.is_available():
            with torch.cuda.amp.autocast(enabled=use_amp):
                x = self.block1(cluster)
                x = self.block2(x)
                x = self.block3(x)
                x = self.block4(x)
                #x = self.block5(x)
                x = self.avgpool(x)
                x = self.flatten(x)
                x = torch.cat([x, clusNumXYEPt], dim=1)
                logits = self.dense_nn(x)
        else:
            x = self.block1(cluster)
            x = self.block2(x)
            x = self.block3(x)
            x = self.block4(x)
            #x = self.block5(x)
            x = self.avgpool(x)
            x = self.flatten(x)
            x = torch.cat([x, clusNumXYEPt], dim=1)
            logits = self.dense_nn(x)
        return logits

###############################################################################


###############################################################################
###################### Training and Validation loop ###########################
# Implement train and validation loop
# Data[0] contains an image of of the cell energies and timings.
# Data[1] contains all features in a dict.
# Data[2] contains all labels
def train_loop(epoch, dataloader, model, loss_fn, optimizer, device="cpu"):

    size = len(dataloader)
    max_batch_number = size - (size % accumulation_steps)
    output_frequency = int(0.1 * size)
    running_loss = 0.0
    epoch_steps = 0
    model.train()
    model.zero_grad(set_to_none=set_to_none)
    if torch.cuda.is_available():
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Loop through the dataset
    for batch, (Clusters, Features, Label) in enumerate(dataloader):
        if batch == max_batch_number:
            break
        Features = Features.to(device, non_blocking=non_blocking)
        Label = Label.to(device, non_blocking=non_blocking)
        if INSTANCE_NOISE:
            Clusters = cm.add_instance_noise(Clusters)
            Clusters = Clusters.to(device, non_blocking=non_blocking)
        else:
            Clusters = Clusters.to(device, non_blocking=non_blocking)

        # prediction and loss
        # If GPU memory is to small one can run over several batches to mimic a
        # larger batch size, per-batch loss has to combined, usually averaging
        # is sufficient

        if torch.cuda.is_available():
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model(Clusters, Features)
                loss = loss_fn(pred, Label[:,6].long())
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()
            # Backpropagation
            if (batch+1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                # zero parameter gradients
                optimizer.zero_grad(set_to_none=set_to_none)
        else:
            pred = model(Clusters, Features)
            loss = loss_fn(pred, Label[:,6].long())
            loss = loss / accumulation_steps
            # Backpropagation
            loss.backward()

        running_loss += loss.item()
        epoch_steps += 1

        if batch % output_frequency == 0 and batch > 0:
            print(f"[Epoch {epoch+1:d}/{num_epochs:d},"
                  f"Batch {batch+1:5d}/{size}]"
                  f" loss: {running_loss/epoch_steps:.3f}")
            running_loss = 0.0

        if use_benchmark and batch == 0:
            torch.cuda.empty_cache()


def val_loop(epoch, dataloader, model, loss_fn, optimizer, device="cpu"):

    val_loss = 0.0
    val_steps = 0
    total = 0
    correct = 0
    model.eval()

    with torch.no_grad():
        for batch, (Clusters, Features, Label) in enumerate(dataloader):
            Clusters = Clusters.to(device, non_blocking=non_blocking)
            Features = Features.to(device, non_blocking=non_blocking)
            Label = Label.to(device, non_blocking=non_blocking)

            if torch.cuda.is_available():
                with torch.cuda.amp.autocast(enabled=use_amp):
                    pred = model(Clusters, Features)
                    loss = loss_fn(pred, Label[:,6].long())
            else:
                pred = model(Clusters, Features)
                loss = loss_fn(pred, Label[:,6].long())#.item()

            correct += (pred.argmax(1) == Label[:,6]).sum().item()
            total += Label.size(0)
            val_loss += loss.cpu().numpy()
            val_steps += 1

    # Save a checkpoint. It is automatically registered with Ray Tune and will
    # potentially be passed as the `checkpoint_dir`parameter in future
    # iterations.
    if epoch % grace_period == 0:
        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            _path = path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), _path)

    tune.report(loss=(val_loss / val_steps), accuracy=(correct / total))

###############################################################################


###############################################################################
############################## Test Loop ######################################
# Implement method for accuracy testing on test set
def test_accuracy(model, device="cpu"):

    # load the test dataset
    dataset_test = cm.load_data_test()

    #get dataloader
    dataloader_test = utils.DataLoader(
        dataset_test, batch_size=256, shuffle=False, num_workers=cpu_count()-1, pin_memory=pin_memory)

    correct = 0
    total = 0
    model.eval()

    with torch.no_grad():
        for batch, (Clusters, Features, Label) in enumerate(dataloader_test):
            Clusters = Clusters.to(device, non_blocking=non_blocking)
            Features = Features.to(device, non_blocking=non_blocking)
            Label = Label.to(device, non_blocking=non_blocking)

            if torch.cuda.is_available():
                with torch.cuda.amp.autocast(enabled=use_amp):
                    pred = model(Clusters, Features)
            else:
                pred = model(Clusters, Features)

            total += Label.size(0)
            correct += (pred.argmax(1) == Label[:,6]).sum().item()

    return correct / total

###############################################################################


###############################################################################
############################# Training Routine ################################
# Implement training routine
def train_model(config, data=None, checkpoint_dir=None):

    import torch
    if cudnn_av and use_benchmark:
        torch.backends.cudnn.enabled = use_benchmark
        torch.backends.cudnn.benchmark = use_benchmark
        print("Cudnn backend and benchmarking enabled")

    # load model
    model = CNN(config["l1"], config["l2"], config["l3"])

    # check for avlaible resource and initialize device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

    print(f"Training started on device {device} with virtual batch_size"
          f" {accumulation_steps * int(config['batch_size'])}"
          f" (real batch_size {int(config['batch_size'])})")

    # send model to device
    model.to(device)

    # initialise loss function and optimizer
    loss_fn = F.cross_entropy
    optimizer = optim.Adam(model.parameters(),
                           lr=config["lr"], weight_decay=config["wd"])

    # The `checkpoint_dir` parameter gets passed by Ray Tune when a checkpoint
    # should be restored.
    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    # load dataset
    if Use_Shared_Memory:
        dataset_train = cm.ClusterDataset(data=data)
    else:
        dataset_train = cm.load_data_train()

    # split trainset in train and validation subsets
    test_abs = int(len(dataset_train) * 0.8)
    data_train, data_val = utils.random_split(
        dataset_train, [test_abs, len(dataset_train) - test_abs])

    # get dataloaders
    dataloader_train, dataloader_val = get_dataloader(
        data_train, data_val, int(config["batch_size"]))

    # Start training loop
    for epoch in range(100):
        train_loop(epoch, dataloader_train, model, loss_fn,
                   optimizer, device=device)
        val_loop(epoch, dataloader_val, model, loss_fn,
                 optimizer, device=device)

    print("Finished Training")

###############################################################################


###############################################################################
############################ Main Function ####################################
# Setup all Ray Tune functionality and start training
def main(num_samples=10, max_num_epochs=10, gpus_per_trial=0):

    # Setup hyperparameter-space to search
    config = {
        "l1": tune.qlograndint(750, 1000, 2), #(500, 1000, 2),
        "l2": tune.qlograndint(125, 500, 2), #(250, 500, 2),
        "l3": tune.qlograndint(9, 65, 2), #(50, 250, 2),
        "lr": tune.loguniform(1e-4, 1e0),
        "wd": tune.loguniform(1e-5, 1e-2),
        "batch_size": tune.choice([32, 64, 128, 256])
    }

    # Init the scheduler
    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=grace_period,
        reduction_factor=reduction_factor)

    # Init the search algorithm
    searchalgorithm = HyperOptSearch(n_initial_points=num_random_trials)
    # Have to limit max number of concurrent trials for searchalgorithm
    Concurrent = int(min(6., np.floor(cpu_count()/cpus_per_trial)))
    searchalgorithm = ConcurrencyLimiter(searchalgorithm,
                                         max_concurrent=Concurrent)

    # Init the Reporter, used for printing the relevant informations
    reporter = CLIReporter(
        parameter_columns = ["l1", "l2", "l3", "lr", "wd", "batch_size"],
        metric_columns = ["loss", "accuracy", "training_iteration"],
        max_report_frequency = 300)

    # Get Current date and time for checkpoint folder
    timestr = strftime("%Y_%m_%d-%H:%M:%S")
    name = "ASHA-" + timestr

    # Load the dataset, with tune.with_parameters the data will be loaded to
    # the shared memory and every trials will have access to it
    if Use_Shared_Memory:
        dataset = cm.load_data()
    else:
        dataset = None

    # Init the run method
    result = tune.run(
        tune.with_parameters(train_model, data=dataset),
        metric="loss",
        mode="min",
        name=name,
        resources_per_trial={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        local_dir="./Ray_Results",
        search_alg=searchalgorithm,
        scheduler=scheduler,
        progress_reporter=reporter,
        checkpoint_score_attr="accuracy",
        keep_checkpoints_num=2,
        max_failures=4,
        raise_on_failed_trial=False)

    # Find best trial and use it on the testset
    best_trial = result.get_best_trial("loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: "
          f"{best_trial.last_result['loss']}")
    print(f"Best trial final validation accuracy: "
          f"{best_trial.last_result['accuracy']}")
    # Adjust the input for your model here
    best_trained_model = CNN(best_trial.config["l1"],
                             best_trial.config["l2"],
                             best_trial.config["l3"])
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)

    test_acc = test_accuracy(best_trained_model, device)
    print(f"Best trial test set accuracy: {test_acc}")

###############################################################################


###############################################################################
######################### Starting the training ###############################
if __name__ == "__main__":
    main(num_samples=num_trials, max_num_epochs=num_epochs,
         gpus_per_trial=gpus_per_trial)
