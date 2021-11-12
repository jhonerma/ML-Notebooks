#!/usr/bin/env python
# coding: utf-8

# ## Load Libraries

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as utils

from functools import reduce as func_reduce
from operator import mul as op_mul
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers.pb2 import PB2
from os import cpu_count, path
from time import strftime
import random

#This class contains DatasetClass and several helper functions
import ClassModule as cm


################################################################################
########################### Run Configuratinos #################################
# Set the number CPUS that should be used per trial
# If set to 1 number of cucurrent training networks is equal to number of CPU
# cores. There should enough memory available to load the dataset into Memory
# for each concurrent trial
# In case of training with GPU this will be limited to number of models training
# simultaneously on GPU. Fractional values are possible, i.e. 0.5 will train 2
# networks on a GPU simultaneously. GPU needs enough memory to hold all models,
# check memory consumption of model on the GPU in advance
# num_workers controls how many subprocesses for loading a dataloader will spawn
cpus_per_trial = 3
gpus_per_trial = 0
num_workers = 4

# num_trials gives the size of the population, i.e. number of different trials
# num_epochs gives the maximum number of training epochs
# perturbation_interval controls after how many epochs bad performers change HP
# combine several batches into one backprop to circumvent GPU memory limitations
# set_to_none puts gradients to None instead of 0, can result in speed-up
# pin_memory and non_blocking can increase performance when loading data from cpu
# to gpu, set to False when training without gpu
# -use_amp sets automatic mixed precision mode, reduces memory usage and can
#  improve training speed (especially on RTX cards). But can also lead to some weird
#  behaviour in pytorch, monitor output for nan/inf loss
# -use cudnn.benchmark when you rely on convolutions and have constant input shape
#  increases gpu memory usage on first forward pass
# -Instance noise can improve training with images
num_trials = 4
num_epochs = 6
perturbation_interval = 2
Use_Shared_Memory = True
accumulation_steps = 2
set_to_none = True
pin_memory = False
non_blocking = False
use_amp = True
use_benchmark = True
INSTANCE_NOISE = True
################################################################################

def get_dataloader(train_ds, val_ds, bs):
    dl_train = utils.DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    dl_val = utils.DataLoader(val_ds, batch_size=bs * 2, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    return  dl_train, dl_val

cuda_av = torch.cuda.is_available()
cuda_devcount = torch.cuda.device_count()
cudnn_av = torch.backends.cudnn.is_available()

# Failsave if there is no gpu and cuda-setting are still turned on
if not cuda_av:
        pin_memory = False
        non_blocking = False
        use_amp = False
        use_benchmark = False
        print("No CUDA-device found, all CUDA-related features turned off")

################################################################################
############################## Network #########################################
### Define the network
# Get a network-candidate from the ASHA-scheduler first and use this notebook
# for hyperparameter tuning

class CNN(nn.Module):
    def __init__(self, l1=100, l2=50, l3=25, input_dim=(2,20,20), num_in_features=5):
        super(CNN, self).__init__()
        self.feature_ext = nn.Sequential(
            nn.Conv2d(2,10, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(10,10, kernel_size=5, padding=0),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(10,10, kernel_size=3, padding=0),
            nn.SiLU(),
            nn.Conv2d(10,6, kernel_size=1),
            nn.SiLU(),
            nn.MaxPool2d(2)
        )

        self.flatten = nn.Flatten()

        # Gives the number of features after the conv layer
        num_features_after_conv = func_reduce(op_mul, list(self.feature_ext(torch.rand(1, *input_dim)).shape))

        self.dense_nn = nn.Sequential(
            nn.Linear(num_features_after_conv + num_in_features, 16),
            nn.SiLU(),
            nn.Linear(16, 128),
            nn.SiLU(),
            nn.Linear(128, 4),
            nn.SiLU(),
            nn.Linear(4,3),
            nn.SiLU()
        )

    def forward(self, cluster, clusNumXYEPt):
        if cuda_av:
            with torch.cuda.amp.autocast(enabled=use_amp):
                cluster = self.feature_ext(cluster)
                x = self.flatten(cluster)
                x = torch.cat([x, clusNumXYEPt], dim=1)
                logits = self.dense_nn(x)
        else:
            cluster = self.feature_ext(cluster)
            x = self.flatten(cluster)
            x = torch.cat([x, clusNumXYEPt], dim=1)
            logits = self.dense_nn(x)
        return logits

################################################################################


################################################################################
###################### Training and Validation loop ############################
### Implement train and validation loop
# Data[0] contains an image of of the cell energies and timings.
# Data[1] contains all features in a dict. Their shapes have to be changed from
# Data[2] contains all labels
def train_loop(epoch, dataloader, model, loss_fn, optimizer, device="cpu"):

    size = len(dataloader)
    max_batch_number = size - (size % accumulation_steps)
    output_frequency = int(0.1 * size)
    running_loss = 0.0
    epoch_steps = 0
    model.train()
    model.zero_grad(set_to_none=set_to_none)
    if cuda_av:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Loop through the dataset
    for batch, Data in enumerate(dataloader):
        if batch == max_batch_number:
            break
        Features = Data[1].to(device, non_blocking=non_blocking)
        Label = Data[2]["PartPID"].to(device, non_blocking=non_blocking)
        if INSTANCE_NOISE:
            Clusters = cm.add_instance_noise(Data[0])
            Clusters = Clusters.to(device, non_blocking=non_blocking)
        else:
            Clusters = Data[0].to(device, non_blocking=non_blocking)

        # prediction and loss
        # If GPU memory is to small one can run over several batches to mimic a
        # larger batch size, per-batch loss has to combined, usually averaging
        # is sufficient
        # Example for how amp is implemented with batch accumulation
        if cuda_av:
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model(Clusters, Features)
                loss = loss_fn(pred, Label.long())
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()
                #Backpropagation
            if (batch+1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                # zero parameter gradients
                optimizer.zero_grad(set_to_none=set_to_none)
        else:
            pred = model(Clusters, Features)
            loss = loss_fn(pred, Label.long())
            loss = loss / accumulation_steps
            # Backpropagation
            loss.backward()

            if (batch+1) % accumulation_steps == 0:
                optimizer.step()
                # zero parameter gradients
                optimizer.zero_grad(set_to_none=set_to_none)

        # Print out running loss every 10% of batches
        running_loss += loss.item()
        epoch_steps += 1

        if batch % output_frequency == 0 and batch > 0:
            print(f"[Epoch {epoch+1:d}/{num_epochs:d},"\
                  f"Batch {batch+1:5d}/{size}]" \
                  f" loss: {running_loss/epoch_steps:.3f}")
            running_loss = 0.0

        # Free up gpu memory used by cudnn for benchmarking
        if use_benchmark and batch == 0:
            torch.cuda.empty_cache()



def val_loop(epoch, dataloader, model, loss_fn, optimizer, device="cpu"):
    # Note that for inference no GradScaler() is necessary during inference
    val_loss = 0.0
    val_steps = 0
    total = 0
    correct = 0
    model.eval()

    with torch.no_grad():
        for batch, Data in enumerate(dataloader):
            Clusters = Data[0].to(device, non_blocking=non_blocking)
            Features = Data[1].to(device, non_blocking=non_blocking)
            Label = Data[2]["PartPID"].to(device, non_blocking=non_blocking)

            if cuda_av:
                with torch.cuda.amp.autocast(enabled=use_amp):
                    pred = model(Clusters, Features)
                    loss = loss_fn(pred, Label.long())#.item()
            else:
                pred = model(Clusters, Features)
                loss = loss_fn(pred, Label.long())#.item()

            correct += (pred.argmax(1) == Label).sum().item()
            total += Label.size(0)
            val_loss += loss.cpu().numpy()
            val_steps += 1

    # Save a checkpoint. It is automatically registered with Ray Tune and will
    # potentially be passed as the `checkpoint_dir`parameter in future
    # iterations. Also report the metrics back to ray with tune.report
    mean_accuracy= correct / total
    with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
        _path = path.join(checkpoint_dir, "checkpoint")
        torch.save({"epoch" : epoch,
         "model_state_dict" : model.state_dict(),
         "optimizer_state_dict" : optimizer.state_dict(),
         "mean_accuracy" : mean_accuracy}, _path)

    # Report metrics back to ray
    tune.report(loss=(val_loss / val_steps), mean_accuracy= mean_accuracy)

################################################################################


################################################################################
############################## Test Loop #######################################
### Implement method for accuracy testing on test set
def test_accuracy(model, device="cpu"):

    dataset_test = cm.load_data_test()

    dataloader_test = utils.DataLoader(
        dataset_test, batch_size=32, shuffle=False, num_workers=cpu_count()-1, pin_memory=pin_memory)

    correct = 0
    total = 0
    model.eval()

    with torch.no_grad():
        for batch, Data in enumerate(dataloader_test):
            Clusters = Data[0].to(device, non_blocking=non_blocking)
            Features = Data[1].to(device, non_blocking=non_blocking)
            Label = Data[2]["PartPID"].to(device, non_blocking=non_blocking)

            if cuda_av:
                with torch.cuda.amp.autocast(enabled=use_amp):
                    pred = model(Clusters, Features)
            else:
                pred = model(Clusters, Features)

            total += Label.size(0)
            correct += (pred.argmax(1) == Label).sum().item()

    return correct / total
################################################################################


################################################################################
############################# Training Routine #################################
### Implement training routine
def train_model(config, data=None, checkpoint_dir=None):
    epoch = 0

    # Importing torch again here is necessary to run the cudnn benchmarks for
    # every trial, since ray can't distribute these
    import torch
    if cudnn_av and use_benchmark:
        torch.backends.cudnn.enabled = use_benchmark
        torch.backends.cudnn.benchmark = use_benchmark
        print("Cudnn backend and benchmarking enabled")

    # load model
    model = CNN()

    # check for avlaible resource and initialize device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

    print(f"Training started on device {device} with virtual batch_size {accumulation_steps * int(config['batch_size'])} (real batch_size {int(config['batch_size'])})")

    # send model to device
    model.to(device)

    # initialise loss function and opptimizer
    loss_fn = F.cross_entropy
    optimizer = torch.optim.Adam(model.parameters(),lr=config["lr"], weight_decay=config["wd"])

    # check whether checkpoint is available
    if checkpoint_dir:
        checkpoint = torch.load(
            path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]


    # load dataset
    if Use_Shared_Memory:
        dataset_train = cm.ClusterDataset(data=data)
    else:
        dataset_train = cm.load_data_train()

    # split trainset in train and validation subsets
    test_abs = int(len(dataset_train) * 0.8)
    subset_train, subset_val = utils.random_split(
        dataset_train, [test_abs, len(dataset_train) - test_abs])

    # get dataloaders
    dataloader_train, dataloader_val = get_dataloader(subset_train, subset_val, int(config["batch_size"]))

    while True:
        train_loop(epoch, dataloader_train, model, loss_fn, optimizer, device=device)
        val_loop(epoch, dataloader_train, model, loss_fn, optimizer, device=device)
        epoch += 1


################################################################################


################################################################################
############################ Main Function #####################################
### Setup all Ray Tune functionality and start training
def main(num_samples=10, max_num_epochs=10, gpus_per_trial=1):


    # Setup hyperparameter-space for initial parameters. Has to be done this way,
    # current limitation of the pb2 implementation
    config = {
        "lr": tune.sample_from(lambda spec: random.uniform(1e-4, 1e0)),
        "wd" : tune.sample_from(lambda spec: random.uniform(0, 1e-1)),
        "batch_size": tune.sample_from(lambda spec: random.randint(32,1024))
    }

    # Init the scheduler
    pb2_scheduler = PB2(time_attr="training_iteration",
    metric="mean_accuracy",
    mode="max",
    perturbation_interval=perturbation_interval,
    hyperparam_bounds={
        "lr": [1e-4, 1e1],
        "wd" : [0, 1e-1],
        "batch_size": [32, 1024]
        })

    # Init the Reporter, used for printing the relevant informations
    reporter = CLIReporter(
        parameter_columns=["lr", "wd", "batch_size"],
        metric_columns=["loss", "mean_accuracy", "training_iteration"])

    #Get Current date and time for checkpoint folder
    timestr = strftime("%Y_%m_%d-%H:%M:%S")
    name = "PBT-" + timestr

    #Stops training when desired accuracy or maximum training epochs is reached
    #Implementation of a custom stopper
    class CustomStopper(tune.Stopper):
        def __init__(self):
            self.should_stop = False

        def __call__(self, trial_id, result):
            if not self.should_stop and result["mean_accuracy"] > 0.95:
                self.should_stop = True
            return self.should_stop or result["training_iteration"] >= max_num_epochs

        def stop_all(self):
            return self.should_stop

    stopper = CustomStopper()

    # Load the dataset, with tune.with_parameters the data will be loaded to the
    # shared memory and every trials will have access to it
    if Use_Shared_Memory:
        dataset = cm.load_data()
    else:
        dataset = None

    # Init the run method
    result = tune.run(
        tune.with_parameters(train_model, data=dataset),
        name = name,
        resources_per_trial={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        local_dir = "./Ray_Results",
        scheduler=pb2_scheduler,
        stop=stopper,
        progress_reporter=reporter,
        checkpoint_score_attr="mean_accuracy",
        keep_checkpoints_num=2)

    # Find best trial and use it on the testset
    best_trial = result.get_best_trial("mean_accuracy", "max", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
    print(f"Best trial final validation accuracy: {best_trial.last_result['mean_accuracy']}")

    best_trained_model = CNN()
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    best_checkpoint_dir = best_trial.checkpoint.value
    statedict = torch.load(path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(statedict["model_state_dict"])

    test_acc = test_accuracy(best_trained_model, device)
    print(f"Best trial test set accuracy: {test_acc}")

################################################################################




################################################################################
######################### Starting the training ################################
if __name__ == "__main__":
	main(num_samples=num_trials, max_num_epochs=num_epochs
        , gpus_per_trial=gpus_per_trial)
