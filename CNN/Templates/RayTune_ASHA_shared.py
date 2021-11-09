#!/usr/bin/env python
# coding: utf-8

### Load Libraries
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
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
from os import cpu_count, path
from time import strftime

################################################################################
############################### Usage ##########################################
# When using this template these are the necessary modifications to be made in
# this script:
# 1. Figure out your available resources. How many CPU cores do I have? how much
#    Memory does my Data need. Do I have a GPU? From this configure the variables
#     cpus_per_trial and gpus_per_trial.
# 2. Change the dataset to your own data and update the training, validation and
#    test routine accordingly.
# 3. Change the model to your architecture and adjust the parameter searchspace
#    in the main function. The endpart of the main routine, where the best model
#    is used on the test, has to be adjusted for the new model, as does the
#    reporter.
# 4. Change the goal of training, if desired. At the moment the training goal is
#    to minimize loss, but one could also choose e.g. to maximaze accuracy
# 5. Choose number of epochs training should last and the number of trials by
#    adjusting the global variables below
# One can ,of course, also change the optimizer to ones favourite in the main
# routine. The loss function can also be changed there. There are also other
# search algorithms available in Ray, check the documentation for all of them
################################################################################
################################################################################

################################################################################
########################### Run Configuratinos #################################
# Set the number CPUS that should be used per trial and dataloader. The number
# of concurrent trials is the minimum of 6 or the number of avlaible cores
# divided by cpus_per_trial. For the search algorithm to function properly this
# upper limit is necessary.
# There should be enough memory available to load the dataset into Memory
# for each concurrent trial
# In case of training with GPU this will be limited to number of models training
# simultaneously on GPU. Fractional values are possible, i.e. 0.5 will train 2
# networks on a GPU simultaneously. GPU needs enough memory to hold all models,
# check memory consumption of model on the GPU in advance
cpus_per_trial = 2
gpus_per_trial = 0

# From the given searchspace num_trials configurations will be sampled.
# num_epochs gives the maximum number of training epochs
# grace_period controls after how many epochs trials will be terminated
# num_random_trials is the number of random searches to probe the loss function
# set_to_none puts gradients to None instead of 0, can result in speed-up
# pin_memory and non_blocking can increase performance when loading data from cpu
# to gpu, set to False when training without gpu
# use cudnn.benchmark when you rely on convolutions and have constant input shape
# Instance noise can improve training with images
num_trials = 6
num_epochs = 3
grace_period = 1
num_random_trials = 6
set_to_none = False
pin_memory = False
non_blocking = False
torch.backends.cudnn.benchmark = False
INSTANCE_NOISE = True
################################################################################


################################################################################
############### Dataset and various helper functions ###########################

# Function for loading data for normalization from file
def load_Normalization_Data(path=path.abspath('normalization.npz')):
    data = np.load(path, allow_pickle=True)

    maxData = { 'maxCellEnergy' : data['maxCellEnergy'], 'maxCellTiming' : data['maxCellTiming']
               ,'maxClusterE' : data['maxClusterE'], 'maxClusterPt' : data['maxClusterPt']
               ,'maxClusterM20' : data['maxClusterM20'], 'maxClusterM02' : data['maxClusterM02']
               ,'maxClusterDistFromVert' : data['maxClusterDistFromVert'], 'maxPartE' : data['maxPartE']
               ,'maxPartPt' : data['maxPartPt'], 'maxPartEta' : data['maxPartEta'], 'maxPartPhi' : data['maxPartPhi'] }

    minData = { 'minCellEnergy' : data['minCellEnergy'], 'minCellTiming' : data['minCellTiming']
               ,'minClusterE' : data['minClusterE'], 'minClusterPt' : data['minClusterPt']
               ,'minClusterM20' : data['minClusterM20'], 'minClusterM02' : data['minClusterM02']
               ,'minClusterDistFromVert' : data['minClusterDistFromVert'], 'minPartE' : data['minPartE']
               ,'minPartPt' : data['minPartPt'], 'minPartEta' : data['minPartEta'], 'minPartPhi' : data['minPartPhi'] }

    return minData, maxData

# Implementation of pytorch dataset class, gets the requested items from shared
# memory. Can be used for datapreprocessing and augmentation.
# Check the pytorch documentation for detailed instructions on setting up a
# dataset class
class ClusterDataset(utils.Dataset):
    """Cluster dataset."""
    # Initialize the class
    def __init__(self, data=None, Normalize=True, arrsize=20):

        self.data = data
        self.arrsize = arrsize
        self.Normalize = Normalize
        if self.Normalize:
            self.minData, self.maxData = load_Normalization_Data()

    # Return size of dataset
    def __len__(self):
        return self.data["Size"]

    # Routine for reconstructing clusters from given cell informations
    def __ReconstructCluster(self, ncell, modnum, row, col, cdata):
        _row = row.copy()
        _col = col.copy()
        if not np.all( modnum[0] == modnum[:ncell]):
            ModNumDif = modnum - np.min(modnum[:ncell])
            mask = np.where(ModNumDif == 1)
            _col[mask] += 48
            mask = np.where(ModNumDif == 2)
            _row[mask] += 24
            mask = np.where(ModNumDif == 3)
            _row[mask] += 24
            _col[mask] += 48

        arr = np.zeros(( self.arrsize, self.arrsize ), dtype=np.float32)

        col_min = np.min(_col[:ncell])
        row_min = np.min(_row[:ncell])
        width = np.max(_col[:ncell]) - col_min
        height = np.max(_row[:ncell]) - row_min
        offset_h = int((self.arrsize-height)/2)
        offset_w = int((self.arrsize-width)/2)

        for i in range(ncell):
            arr[ _row[i] - row_min + offset_h, _col[i] - col_min + offset_w ] = cdata[i]
        return arr

    # Function for merging the timing and energy information into one 'picture'
    def __GetCluster(self, ncell, modnum, row, col, energy, timing):
        cluster_e = self.__ReconstructCluster(ncell, modnum, row, col, energy)
        cluster_t = self.__ReconstructCluster(ncell, modnum, row, col, timing)
        return np.stack([cluster_e, cluster_t], axis=0)

    # One-hot encoding for the particle code
    def __ChangePID(self, PID):
        if (PID != 111) & (PID != 221):
            PID = np.int16(0)
        if PID == 111:
            PID = np.int16(1)
        if PID == 221:
            PID = np.int16(2)
        return PID

    # If normalize is true return normalized feature otherwise return feature
    def __Normalize(self, feature, min, max):
        feature = np.atleast_1d(feature)
        if self.Normalize:
            return self.__Norm01(feature, min, max)
        else:
            return feature

    # Function for feature normaliztion to the range 0-1
    def __Norm01(self, data, min, max):
        return (data - min) / (max - min)

    # Get a single entry from the data, do processing and format output
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        _ClusterN = self.data['ClusterN'][idx]
        _Cluster = self.__Normalize(self.data['Cluster'][idx], 0, self.maxData['maxCellEnergy'])
        _ClusterTiming = self.__Normalize(self.data['ClusterTiming'][idx], 0, self.maxData['maxCellTiming'])
        _ClusterType = self.data['ClusterType'][idx]
        _ClusterE = self.__Normalize(self.data['ClusterE'][idx], self.minData['minClusterE'], self.maxData['maxClusterE'])
        _ClusterPt = self.__Normalize(self.data['ClusterPt'][idx], self.minData['minClusterPt'], self.maxData['maxClusterPt'])
        _ClusterModuleNumber = self.data['ClusterModuleNumber'][idx]
        _ClusterCol = self.data['ClusterCol'][idx]
        _ClusterRow = self.data['ClusterRow'][idx]
        _ClusterM02 = self.__Normalize(self.data['ClusterM02'][idx], self.minData['minClusterM02'], self.maxData['maxClusterM02'])
        _ClusterM20 = self.__Normalize(self.data['ClusterM20'][idx], self.minData['minClusterM20'], self.maxData['maxClusterM20'])
        _ClusterDistFromVert = self.__Normalize(self.data['ClusterDistFromVert'][idx], self.minData['minClusterDistFromVert'], self.maxData['maxClusterDistFromVert'])
        _PartE = self.data['PartE'][idx]
        _PartPt = self.data['PartPt'][idx]
        _PartEta = self.data['PartEta'][idx]
        _PartPhi = self.data['PartPhi'][idx]
        _PartIsPrimary = self.data['PartIsPrimary'][idx]
        _PartPID = self.data['PartPID'][idx]

        _PartPID = self.__ChangePID(_PartPID)

        img = self.__GetCluster(_ClusterN, _ClusterModuleNumber, _ClusterRow, _ClusterCol, _Cluster, _ClusterTiming)

        # Stack the features in a single array
        features = np.concatenate((_ClusterE, _ClusterPt, _ClusterM02, _ClusterM20, _ClusterDistFromVert))

        labels = { "ClusterType" : _ClusterType, "PartE" : _PartE, "PartPt" : _PartPt, "PartEta" : _PartEta, "PartPhi" : _PartPhi
                  , "PartIsPrimary" : _PartIsPrimary, "PartPID" : _PartPID }

        return (img, features, labels)

# Helperfunction for loading the dataset. It is good to use functions for this,
# because pythons garbage collection will trigger at the end of a function call
# and clean up everything that is possible, i.e. free up memory and close files
def load_data(path=path.abspath('data_train.npz')):
    data = np.load(path, allow_pickle=True)
    data_dict = { 'Size' : data['Size'], 'ClusterN' : data['ClusterN'], 'Cluster' : data['Cluster']
    , 'ClusterTiming' : data['ClusterTiming'], 'ClusterType' : data['ClusterType']
    , 'ClusterE' : data['ClusterE'], 'ClusterPt' : data['ClusterPt']
    , 'ClusterModuleNumber' : data['ClusterModuleNumber'], 'ClusterCol' : data['ClusterCol'], 'ClusterCol' : data['ClusterCol']
    , 'ClusterRow' : data['ClusterRow'], 'ClusterM02' : data['ClusterM02']
    , 'ClusterM20' : data['ClusterM20'], 'ClusterDistFromVert' : data['ClusterDistFromVert']
    , 'PartE' : data['PartE'], 'PartPt' : data['PartPt']
    , 'PartEta' : data['PartEta'], 'PartPhi' : data['PartPhi']
    , 'PartIsPrimary' : data['PartIsPrimary'], 'PartPID' : data['PartPID']}
    return data_dict

# Helperfunction for obtaining dataloaders
def get_dataloader(train_ds, val_ds, bs):
    dl_train = utils.DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=cpus_per_trial-1, pin_memory=pin_memory)
    dl_val = utils.DataLoader(val_ds, batch_size=bs * 2, shuffle=True, num_workers=cpus_per_trial-1, pin_memory=pin_memory)
    return  dl_train, dl_val

# ## Instance Noise
# https://arxiv.org/abs/1610.04490
def add_instance_noise(data, std=0.1):
    return data + 0.001 * torch.distributions.Normal(0, std).sample(data.shape)

################################################################################


################################################################################
############################## Network #########################################
### Define the network
# The number of neurons per layer here has been made variable, so ray can search
# for the optimal number. The number of channels in the feature extraction
# layer could also be made variable e.g.
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
            nn.Linear(num_features_after_conv + num_in_features, l1),
            nn.SiLU(),
            nn.Linear(l1, l2),
            nn.SiLU(),
            nn.Linear(l2, l3),
            nn.SiLU(),
            nn.Linear(l3,3),
            nn.SiLU()
        )

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

    size = len(dataloader)
    output_frequency = int(0.1 * size)
    running_loss = 0.0
    epoch_steps = 0

    # Loop through the dataset
    for batch, Data in enumerate(dataloader):
        if INSTANCE_NOISE:
            Clusters = add_instance_noise(Data[0]).to(device, non_blocking=non_blocking)
        else:
            Clusters = Data[0].to(device, non_blocking=non_blocking)

        Features = Data[1].to(device, non_blocking=non_blocking)
        Label = Data[2]["PartPID"].to(device, non_blocking=non_blocking)

        # zero parameter gradients
        optimizer.zero_grad(set_to_none=set_to_none)

        # prediction and loss
        pred = model(Clusters, Features)
        loss = loss_fn(pred, Label.long())

        # Backpropagation
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        epoch_steps += 1

        if batch % output_frequency == 0 and batch > 0:
            print(f"[Epoch {epoch+1:d}/{num_epochs:d},"\
                  f"Batch {batch+1:5d}/{size}]" \
                  f" loss: {running_loss/epoch_steps:.3f}")
            running_loss = 0.0

def val_loop(epoch, dataloader, model, loss_fn, optimizer, device="cpu"):

    val_loss = 0.0
    val_steps = 0
    total = 0
    correct = 0
    size = len(dataloader.dataset)

    with torch.no_grad():
        for batch, Data in enumerate(dataloader):
            Clusters = Data[0].to(device, non_blocking=non_blocking)
            Features = Data[1].to(device, non_blocking=non_blocking)
            Label = Data[2]["PartPID"].to(device, non_blocking=non_blocking)

            pred = model(Clusters, Features)
            correct += (pred.argmax(1) == Label).type(torch.float).sum().item()

            loss = loss_fn(pred, Label.long())#.item()
            val_loss += loss.cpu().numpy()
            val_steps += 1

    # Save a checkpoint. It is automatically registered with Ray Tune and will
    # potentially be passed as the `checkpoint_dir`parameter in future
    # iterations.
    if epoch % grace_period == 0:
        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            _path = path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), _path)

    tune.report(loss=(val_loss / val_steps), accuracy= correct / size)

################################################################################



################################################################################
############################## Test Loop #######################################
### Implement method for accuracy testing on test set
def test_accuracy(model, device="cpu"):

    #load the test dataset
    data = load_data(path=path.abspath('data_test.npz'))
    dataset_test = ClusterDataset(data=data)

    #get dataloader
    dataloader_test = utils.DataLoader(
        dataset_test, batch_size=32, shuffle=False, num_workers=cpu_count()-1, pin_memory=pin_memory)

    correct = 0
    total = len(dataloader_test.dataset)

    with torch.no_grad():
        for batch, Data in enumerate(dataloader_test):
            Clusters = Data[0].to(device, non_blocking=non_blocking)
            Features = Data[1].to(device, non_blocking=non_blocking)
            Label = Data[2]["PartPID"].to(device, non_blocking=non_blocking)

            pred = model(Clusters, Features)
            correct += (pred.argmax(1) == Label).type(torch.float).sum().item()

    return correct / total

################################################################################


################################################################################
############################# Training Routine #################################
### Implement training routine
def train_model(config, data=None, checkpoint_dir=None):

    # load model
    model = CNN(config["l1"],config["l2"],config["l3"])

    # check for avlaible resource and initialize device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

    print(f"Training started on device {device}")

    # send model to device
    model.to(device)

    # initialise loss function and optimizer
    loss_fn = F.cross_entropy
    optimizer = torch.optim.Adam(model.parameters(),lr=config["lr"], weight_decay=config["wd"])

    # The `checkpoint_dir` parameter gets passed by Ray Tune when a checkpoint
    # should be restored.
    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    # load dataset
    dataset_train = ClusterDataset(data=data)

    # split trainset in train and validation subsets
    test_abs = int(len(dataset_train) * 0.8)
    data_train, data_val = utils.random_split(
        dataset_train, [test_abs, len(dataset_train) - test_abs])

    # get dataloaders
    dataloader_train, dataloader_val = get_dataloader(data_train, data_val, int(config["batch_size"]))

    #Start training loop
    for epoch in range(100):
        train_loop(epoch, dataloader_train, model, loss_fn, optimizer, device=device)
        val_loop(epoch, dataloader_val, model, loss_fn, optimizer, device=device)

    print("Finished Training")

################################################################################


################################################################################
############################ Main Function #####################################
### Setup all Ray Tune functionality and start training
def main(num_samples=10, max_num_epochs=10, gpus_per_trial=0):

    # Setup hyperparameter-space to search
    config = {
        "l1": tune.qlograndint(64, 3000, 2),
        "l2": tune.qlograndint(32, 2000, 2),
        "l3": tune.qlograndint(3, 1000, 2),
        "lr": tune.loguniform(1e-4, 1e-1),
        "wd": tune.loguniform(1e-5, 1e-3),
        "batch_size": tune.choice([32, 64, 128, 256])
    }

    # Init the scheduler
    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=grace_period,
        reduction_factor=2)

    # Init the search algorithm
    searchalgorithm = HyperOptSearch(n_initial_points=num_random_trials)
    # Have to limit max number of concurrent trials for searchalgorithm
    searchalgorithm = ConcurrencyLimiter(searchalgorithm,
                max_concurrent=int(min(6., np.floor(cpu_count()/cpus_per_trial))))

    # Init the Reporter, used for printing the relevant informations
    reporter = CLIReporter(
        parameter_columns=["l1", "l2", "l3", "lr","wd", "batch_size"],
        metric_columns=["loss", "accuracy", "training_iteration"])

    #Get Current date and time for checkpoint folder
    timestr = strftime("%Y_%m_%d-%H:%M:%S")
    name = "ASHA-" + timestr

    # Load the dataset, with tune.with_parameters the data will be loaded to the
    # shared memory and every trials will have access to it
    dataset = load_data()

    # Init the run method
    result = tune.run(
        tune.with_parameters(train_model, data=dataset),
        metric="loss",
        mode="min",
        name = name,
        resources_per_trial={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        local_dir = "./Ray_Results",
        search_alg = searchalgorithm,
        scheduler=scheduler,
        progress_reporter=reporter,
        checkpoint_score_attr="accuracy",
        keep_checkpoints_num=2)

    # Find best trial and use it on the testset
    best_trial = result.get_best_trial("loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
    print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")
    # Adjust the input for your model here
    best_trained_model = CNN(best_trial.config["l1"], best_trial.config["l2"], best_trial.config["l3"])
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(path.join(best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)

    test_acc = test_accuracy(best_trained_model, device)
    print(f"Best trial test set accuracy: {test_acc}")

################################################################################


################################################################################
######################### Starting the training ################################
if __name__ == "__main__":
    main(num_samples=num_trials, max_num_epochs=num_epochs
        , gpus_per_trial=gpus_per_trial)
