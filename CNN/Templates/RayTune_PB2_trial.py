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

################################################################################
############################### Usage ##########################################
# When using this template these are the necessary modifications to be made in
# this script:
# 1. Figure out your available resources. How many CPU cores do I have? how much
#    Memory does my Data need. From this configure the variables cpus_per_trial
#    And gpus_per_trial
# 2. Change the dataset to your own data and update the training, validation and
#    test routine accordingly.
# 3. Change the model to your architecture and adjust the parameter searchspace
#    in the main function. The endpart of the main routine, where the best model
#    is used on the test, has to be adjusted for the new model, as does the
#    reporter.
# 4. Change the goal of training, if desired. At the moment the training goal is
#    to minimize loss, but one could also choose e.g. to maximaze accuracy
# 5. Choose number of epochs training should last and the number of trials by
#    adjusting
# One can ,of course, also change the optimizer to ones favourite in the main
# routine. The loss function can also be changed there. There are also other
# search algorithms available in Ray, check the documentation for all of them
################################################################################
################################################################################


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
cpus_per_trial = 2
gpus_per_trial = 0

# Set the numbers of trials to be run. From the given searchspace num_trials
# configurations will be sampled. num_epochs gives the maximum number of training
# epochs for the best perfoming trials
num_trials = 5
num_epochs = 3
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

# Implementation of pytorch dataset class for my dataset, loads the full dataset
# into ram. Can be used for datapreprocessing and augmentation.
# Check the pytorch documentation for detailed instructions on setting up a
# dataset class
class ClusterDataset_Full(utils.Dataset):
    """Cluster dataset."""
    #Load data
    def __init__(self, npz_file, Normalize=True, arrsize=20):
        """
        Args:
            npz_file (string): Path to the npz file.
        """
        self.data = np.load(npz_file, allow_pickle=True)
        self.arrsize = arrsize
        self.ClusterN = self.data['ClusterN']
        self.Cluster = self.data['Cluster']
        self.ClusterTiming = self.data['ClusterTiming']
        self.ClusterType = self.data['ClusterType']
        self.ClusterE = self.data['ClusterE']
        self.ClusterPt = self.data['ClusterPt']
        self.ClusterModuleNumber = self.data['ClusterModuleNumber']
        self.ClusterCol = self.data['ClusterCol']
        self.ClusterRow = self.data['ClusterRow']
        self.ClusterM02 = self.data['ClusterM02']
        self.ClusterM20 = self.data['ClusterM20']
        self.ClusterDistFromVert = self.data['ClusterDistFromVert']
        self.PartE = self.data['PartE']
        self.PartPt = self.data['PartPt']
        self.PartEta = self.data['PartEta']
        self.PartPhi = self.data['PartPhi']
        self.PartIsPrimary = self.data['PartIsPrimary']
        self.PartPID = self.data['PartPID']
        self.Normalize = Normalize
        if Normalize:
            self.minData, self.maxData = load_Normalization_Data()

    #return size of dataset
    def __len__(self):
        return self.data["Size"]

    #Routine for reconstructing clusters from given cell informations
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

    # function for merging the timing and energy information into one 'picture'
    def __GetCluster(self, ncell, modnum, row, col, energy, timing):
        cluster_e = self.__ReconstructCluster(ncell, modnum, row, col, energy)
        cluster_t = self.__ReconstructCluster(ncell, modnum, row, col, timing)
        return np.stack([cluster_e, cluster_t], axis=0)

    # one-hot encoding for the particle code
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

        _ClusterN = self.ClusterN[idx]
        _Cluster = self.__Normalize(self.Cluster[idx], 0, self.maxData['maxCellEnergy'])
        _ClusterTiming = self.__Normalize(self.ClusterTiming[idx], 0, self.maxData['maxCellTiming'])
        _ClusterType = self.ClusterType[idx]
        _ClusterE = self.__Normalize(self.ClusterE[idx], self.minData['minClusterE'], self.maxData['maxClusterE'])
        _ClusterPt = self.__Normalize(self.ClusterPt[idx], self.minData['minClusterPt'], self.maxData['maxClusterPt'])
        _ClusterModuleNumber = self.ClusterModuleNumber[idx]
        _ClusterCol = self.ClusterCol[idx]
        _ClusterRow = self.ClusterRow[idx]
        _ClusterM02 = self.__Normalize(self.ClusterM02[idx], self.minData['minClusterM02'], self.maxData['maxClusterM02'])
        _ClusterM20 = self.__Normalize(self.ClusterM20[idx], self.minData['minClusterM20'], self.maxData['maxClusterM20'])
        _ClusterDistFromVert = self.__Normalize(self.ClusterDistFromVert[idx], self.minData['minClusterDistFromVert'], self.maxData['maxClusterDistFromVert'])
        _PartE = self.PartE[idx]
        _PartPt = self.PartPt[idx]
        _PartEta = self.PartEta[idx]
        _PartPhi = self.PartPhi[idx]
        _PartIsPrimary = self.PartIsPrimary[idx]
        _PartPID = self.PartPID[idx]

        _PartPID = self.__ChangePID(_PartPID)

        img = self.__GetCluster(_ClusterN, _ClusterModuleNumber, _ClusterRow, _ClusterCol, _Cluster, _ClusterTiming)

        features = { "ClusterType" : _ClusterType, "ClusterE" : _ClusterE, "ClusterPt" : _ClusterPt
                    , "ClusterM02" : _ClusterM02, "ClusterM20" : _ClusterM20 , "ClusterDist" : _ClusterDistFromVert}
        labels = { "PartE" : _PartE, "PartPt" : _PartPt, "PartEta" : _PartEta, "PartPhi" : _PartPhi
                  , "PartIsPrimary" : _PartIsPrimary, "PartPID" : _PartPID }

        return (img, features, labels)

# Helperfunction for getting the dataset. It is good to use functions for this,
# because pythons garbage collection will trigger at the end of a function call
# and clean up everything that is possible, i.e. free up memory and close files
def load_data_train(path=path.abspath('data_train.npz')):
    ds_train = ClusterDataset_Full(path)
    return ds_train

def load_data_test(path=path.abspath('data_test.npz')):
    ds_test = ClusterDataset_Full(path)
    return ds_test

# Helperfunction for obtaining dataloaders
def get_dataloader(train_ds, val_ds, bs):
    dl_train = utils.DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=cpus_per_trial-1)
    dl_val = utils.DataLoader(val_ds, batch_size=bs * 2, shuffle=True, num_workers=cpus_per_trial-1)
    return  dl_train, dl_val

# Helper function for getting the right dimension for input features
# [batch_size, 1] from dimension [batch_size] of data saved in a dict
def unsqueeze_features(features):
    for key in features.keys():
        features[key] = features[key].view(-1,1)
    return features

### Add Instance Noise to training image, can improve training
# https://arxiv.org/abs/1610.04490
INSTANCE_NOISE = True
def add_instance_noise(data, device, std=0.01):
    return data + 0.001 * torch.distributions.Normal(0, std).sample(data.shape).to(device)

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
        Clusters = Data[0].to(device)
        Features = unsqueeze_features(Data[1])
        Labels = Data[2]

        # add instance noise to cluster
        if INSTANCE_NOISE:
            Clusters = add_instance_noise(Clusters, device)

        # Add all additional features into a single tensor
        ClusterProperties = torch.cat([Features["ClusterE"]
            , Features["ClusterPt"], Features["ClusterM02"]
            , Features["ClusterM20"], Features["ClusterDist"]], dim=1).to(device)
        #Labels = torch.cat([Labels["PartPID"], dim=1]).to(device)
        Label = Labels["PartPID"].to(device)

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

        if batch % 100000 == 99999:
            print("[%d, %5d] loss: %.3f" % (epoch + 1, batch + 1,
                                            running_loss / epoch_steps))
            running_loss = 0.0


def val_loop(epoch, dataloader, model, loss_fn, optimizer, device="cpu"):

    val_loss = 0.0
    val_steps = 0
    total = 0
    correct = 0
    size = len(dataloader.dataset)

    for batch, Data in enumerate(dataloader):
        with torch.no_grad():
            Clusters = Data[0].to(device)
            Features = unsqueeze_features(Data[1])
            Labels = Data[2]
            ClusterProperties = torch.cat([Features["ClusterE"], Features["ClusterPt"], Features["ClusterM02"]
                                      , Features["ClusterM20"], Features["ClusterDist"]], dim=1).to(device)
            #Labels = torch.cat([Labels["PartPID"], dim=1]).to(device)
            Label = Labels["PartPID"].to(device)

            pred = model(Clusters, ClusterProperties)
            correct += (pred.argmax(1) == Label).type(torch.float).sum().item()

            loss = loss_fn(pred, Label.long())#.item()
            val_loss += loss.cpu().numpy()
            val_steps += 1

    # Save a checkpoint. It is automatically registered with Ray Tune and will
    # potentially be passed as the `checkpoint_dir`parameter in future
    # iterations. Also report the metrics back to ray with tune.report
    mean_accuracy= correct / size
    if epoch % 5 == 0:
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

    dataset_test = load_data_test()

    dataloader_test = utils.DataLoader(
        dataset_test, batch_size=32, shuffle=False, num_workers=cpu_count()-1)

    correct = 0
    total = len(dataloader_test.dataset)

    with torch.no_grad():
        for batch, Data in enumerate(dataloader_test):
            Clusters = Data[0].to(device)
            Features = unsqueeze_features(Data[1])
            Labels = Data[2]
            ClusterProperties = torch.cat([Features["ClusterE"], Features["ClusterPt"], Features["ClusterM02"]
                                      , Features["ClusterM20"], Features["ClusterDist"]], dim=1).to(device)
            #Labels = torch.cat([Labels["PartPID"], dim=1]).to(device)
            Label = Labels["PartPID"].to(device)


            pred = model(Clusters, ClusterProperties)
            correct += (pred.argmax(1) == Label).type(torch.float).sum().item()

    return correct / total
################################################################################


################################################################################
############################# Training Routine #################################
### Implement training routine
def train_model(config, checkpoint_dir=None):
    epoch = 0
    # load model
    model = CNN()

    # check for avlaible resource and initialize device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
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
    dataset_train = load_data_train()

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

    # Setup hyperparameter-space to search
    config = {
        "lr": tune.loguniform(1e-4, 1e0),
        "wd" : tune.uniform(0, 1e-1),
        "batch_size": tune.choice([16, 32, 64, 128, 256])
    }

    # Init the scheduler
    pb2_scheduler = PB2(time_attr="training_iteration",
    metric="mean_accuracy",
    mode="max",
    perturbation_interval=5,
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

    # Init the run method
    result = tune.run(
        train_model,
        name = name,
        resources_per_trial={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        local_dir = "./Ray_Results",
        scheduler=pb2_scheduler,
        stop=stopper,
        progress_reporter=reporter,
        checkpoint_score_attr="mean_accuracy",
        keep_checkpoints_num=4)

    # Find best trial and use it on the testset
    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(best_trial.last_result["mean_accuracy"]))

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
    print("Best trial test set accuracy: {}".format(test_acc))

################################################################################


################################################################################
######################### Starting the training ################################
if __name__ == "__main__":
	main(num_samples=num_trials, max_num_epochs=num_epochs
        , gpus_per_trial=gpus_per_trial)