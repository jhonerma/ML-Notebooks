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
# Set the number CPUS that should be used per trial and dataloader
# If set to 1 number of cucurrent training networks is equal to number of CPU
# cores. There should enough memory available to load the dataset into Memory
# for each concurrent trial
# In case of training with GPU this will be limited to number of models training
# simultaneously on GPU. Fractional values are possible, i.e. 0.5 will train 2
# networks on a GPU simultaneously. GPU needs enough memory to hold all models,
# check memory consumption of model on the GPU in advance
cpus_per_trial = 4
gpus_per_trial = 0

# num_trials gives the size of the population, i.e. number of different trials
# num_epochs gives the maximum number of training epochs
# perturbation_interval controls after how many epochs bad performers change HP
# pin_memory and non_blocking can increase performance when loading data from cpu
# to gpu, set to False when training without gpu
# Instance noise can improve training with images
num_trials = 3
num_epochs = 4
perturbation_interval = 2
Use_Shared_Memory = True
pin_memory = False
non_blocking = False
INSTANCE_NOISE = True
################################################################################

def get_dataloader(train_ds, val_ds, bs):
    dl_train = utils.DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=cpus_per_trial-1, pin_memory=pin_memory)
    dl_val = utils.DataLoader(val_ds, batch_size=bs * 2, shuffle=True, num_workers=cpus_per_trial-1, pin_memory=pin_memory)
    return  dl_train, dl_val


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
        cluster = self.feature_ext(cluster)
        x = self.flatten(cluster)
        x = torch.cat([x, clusNumXYEPt], dim=1)
        logits = self.dense_nn(x)
        return logits

    def forward(self, cluster, clusNumXYEPt):
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
# [batch_size] to [batch_size,1] for input into linear layers, implemented via
# function unsqueeze features here. Data[2] contains all labels
def train_loop(epoch, dataloader, model, loss_fn, optimizer, device="cpu"):

    size = len(dataloader.dataset)
    running_loss = 0.0
    epoch_steps = 0

    for batch, Data in enumerate(dataloader):
        Features = cm.unsqueeze_features(Data[1])
        if INSTANCE_NOISE:
            Clusters = cm.add_instance_noise(Data[0]).to(device, non_blocking=non_blocking)
        else:
            Clusters = Data[0].to(device, non_blocking=non_blocking)

        #Labels = torch.cat([Labels["PartPID"], dim=1]).to(device)
        Label = Data[2]["PartPID"].to(device, non_blocking=non_blocking)

        # Add all additional features into a single tensor
        ClusterProperties = torch.cat([Features["ClusterE"]
            , Features["ClusterPt"], Features["ClusterM02"]
            , Features["ClusterM20"], Features["ClusterDist"]], dim=1).to(device, non_blocking=non_blocking)

        # zero parameter gradients
        optimizer.zero_grad()

        # prediction and loss
        pred = model(Clusters, ClusterProperties)
        loss = loss_fn(pred, Label.long())

        # Backpropagation
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        epoch_steps += 1

        if batch % 10000 == 9999:
            print(f"[Epoch {epoch+1:d}, Batch {batch+1:5d}]" \
                  f" loss: {running_loss/epoch_steps:.3f}")
            running_loss = 0.0


def val_loop(epoch, dataloader, model, loss_fn, optimizer, device="cpu"):

    val_loss = 0.0
    val_steps = 0
    total = 0
    correct = 0
    size = len(dataloader.dataset)

    for batch, Data in enumerate(dataloader):
        with torch.no_grad():
            Features = cm.unsqueeze_features(Data[1])
            Clusters = Data[0].to(device, non_blocking=non_blocking)
            Label = Data[2]["PartPID"].to(device, non_blocking=non_blocking)
            ClusterProperties = torch.cat([Features["ClusterE"], Features["ClusterPt"], Features["ClusterM02"]
                                      , Features["ClusterM20"], Features["ClusterDist"]], dim=1).to(device, non_blocking=non_blocking)

            pred = model(Clusters, ClusterProperties)
            correct += (pred.argmax(1) == Label).type(torch.float).sum().item()

            loss = loss_fn(pred, Label.long())#.item()
            val_loss += loss.cpu().numpy()
            val_steps += 1

    # Save a checkpoint. It is automatically registered with Ray Tune and will
    # potentially be passed as the `checkpoint_dir`parameter in future
    # iterations. Also report the metrics back to ray with tune.report
    mean_accuracy= correct / size
    with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
        _path = path.join(checkpoint_dir, "checkpoint")
        torch.save({"epoch" : epoch,
         "model_state_dict" : model.state_dict(),
         "optimizer_state_dict" : optimizer.state_dict(),
         "mean_accuracy" : mean_accuracy}, _path)


    tune.report(loss=(val_loss / val_steps), mean_accuracy= mean_accuracy)
################################################################################


################################################################################
############################## Test Loop #######################################
### Implement method for accuracy testing on test set
def test_accuracy(model, device="cpu"):

    #load the test dataset
    dataset_test = cm.load_data_test()

    #get dataloader
    dataloader_test = utils.DataLoader(
        dataset_test, batch_size=32, shuffle=False, num_workers=cpu_count()-1, pin_memory=pin_memory)

    correct = 0
    total = len(dataloader_test.dataset)

    with torch.no_grad():
        for batch, Data in enumerate(dataloader_test):
            Features = cm.unsqueeze_features(Data[1])
            Clusters = Data[0].to(device, non_blocking=non_blocking)
            Label = Data[2]["PartPID"].to(device, non_blocking=non_blocking)
            ClusterProperties = torch.cat([Features["ClusterE"], Features["ClusterPt"], Features["ClusterM02"]
                                      , Features["ClusterM20"], Features["ClusterDist"]], dim=1).to(device, non_blocking=non_blocking)

            pred = model(Clusters, ClusterProperties)
            correct += (pred.argmax(1) == Label).type(torch.float).sum().item()

    return correct / total
################################################################################


################################################################################
############################# Training Routine #################################
### Implement training routine
def train_model(config, data=None, checkpoint_dir=None):
    epoch = 0
    # load model
    model = CNN()

    # check for avlaible resource and initialize device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

    print(f"Training started on device {device}")

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
