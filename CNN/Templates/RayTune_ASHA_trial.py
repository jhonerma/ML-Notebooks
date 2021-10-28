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
from ray.tune.schedulers import ASHAScheduler
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
#    in the main function
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
num_epochs = 30
################################################################################


################################################################################
############### Dataset and various helper functions ###########################

# Implementation of pytorch dataset class for my dataset, loads the full dataset
# into ram. Can be used for datapreprocessing and augmentation.
# Check the pytorch documentation for detailed instructions on setting up a
# dataset class
class ClusterDataset_Full(utils.Dataset):
    """Cluster dataset."""
    #Load data
    def __init__(self, npz_file, arrsize=20):
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

    # Get a single entry from the data, do processing and format output
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        ClusterN = self.ClusterN[idx]
        Cluster = self.Cluster[idx]
        ClusterTiming = self.ClusterTiming[idx]
        ClusterType = self.ClusterType[idx]
        ClusterE = self.ClusterE[idx]
        ClusterPt = self.ClusterPt[idx]
        ClusterModuleNumber = self.ClusterModuleNumber[idx]
        ClusterCol = self.ClusterCol[idx]
        ClusterRow = self.ClusterRow[idx]
        ClusterM02 = self.ClusterM02[idx]
        ClusterM20 = self.ClusterM20[idx]
        ClusterDistFromVert = self.ClusterDistFromVert[idx]
        PartE = self.PartE[idx]
        PartPt = self.PartPt[idx]
        PartEta = self.PartEta[idx]
        PartPhi = self.PartPhi[idx]
        PartIsPrimary = self.PartIsPrimary[idx]
        PartPID = self.PartPID[idx]

        PartPID = self.__ChangePID(PartPID)

        img = self.__GetCluster(ClusterN, ClusterModuleNumber, ClusterRow, ClusterCol, Cluster, ClusterTiming)

        features = { "ClusterType" : ClusterType, "ClusterE" : ClusterE, "ClusterPt" : ClusterPt
                    , "ClusterM02" : ClusterM02, "ClusterM20" : ClusterM20 , "ClusterDist" : ClusterDistFromVert}
        labels = { "PartE" : PartE, "PartPt" : PartPt, "PartEta" : PartEta, "PartPhi" : PartPhi
                  , "PartIsPrimary" : PartIsPrimary, "PartPID" : PartPID }

        return (img, features, labels)

# Helperfunction for getting the dataset. It is good to use functions for this,
# because pythons garbage collection will trigger at the end of a function call
# and clean up everything that is possible, i.e. free up memory and close files
# Ray needs an absolute Path to the data
def load_data_train(path='/home/jhonerma/ML-Notebooks/CNN/Data/data_train.npz'):
    ds_train = ClusterDataset_Full(path)
    return ds_train

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
def add_instance_noise(data, device, std=0.1):
    return data + 0.01 * torch.distributions.Normal(0, std).sample(data.shape).to(device)

################################################################################
############################## Network #########################################
### Define the network
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

        ClusterProperties = torch.cat([Features["ClusterE"]
            , Features["ClusterPt"], Features["ClusterM02"]
            , Features["ClusterM20"], Features["ClusterDist"]], dim=1).to(device)
        #Labels = torch.cat([Labels["PartPID"], dim=1]).to(device)
        Label = Labels["PartPID"].to(device)

        if INSTANCE_NOISE:
            Clusters = add_instance_noise(Clusters, device)

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

        if batch % 50000 == 49999:
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
    dataset_test = cm.load_data_test('data_test.npz')
    #get dataloader
    dataloader_test = utils.DataLoader(
        dataset_test, batch_size=4, shuffle=False, num_workers=cpu_av-1)

    correct = 0
    total = len(dataloader_test.dataset)

    with torch.no_grad():
        for batch, Data in enumerate(dataloader_test):
            Clusters = Data[0].to(device)
            Features = unsqueeze_features(Data[1])
            Labels = Data[2]
            ClusterProperties = torch.cat([Features["ClusterE"]
            , Features["ClusterPt"], Features["ClusterM02"]
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

    # load model
    model = CNN(config["l1"],config["l2"],config["l3"])

    # check for avlaible resource and initialize device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

    # send model to device
    model.to(device)

    # initialise loss function and optimizer
    loss_fn = F.cross_entropy
    optimizer = torch.optim.Adam(model.parameters(),lr=config["lr"], weight_decay=config["wd"])

    # check whether checkpoint is available
    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    # load dataset
    dataset_train = load_data_train()

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
        "l1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 8)),
        "l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 8)),
        "l3": tune.sample_from(lambda _: 2 ** np.random.randint(2, 8)),
        "lr": tune.loguniform(1e-4, 1e-1),
        "wd": tune.loguniform(1e-5, 1e-3),
        "batch_size": tune.choice([32, 64, 128, 256, 512])
    }

    # Init the scheduler
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)

    # Init the Reporter, used for printing the relevant informations
    reporter = CLIReporter(
        parameter_columns=["l1", "l2", "l3", "lr","wd", "batch_size"],
        metric_columns=["loss", "accuracy", "training_iteration"])

    #Get Current date and time for checkpoint folder
    timestr = strftime("%Y_%m_%d-%H:%M:%S")
    name = "ASHA-" + timestr

    # Init the run method
    result = tune.run(
        train_model,
        name = name,
        resources_per_trial={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        local_dir = "./Ray_Results",
        scheduler=scheduler,
        progress_reporter=reporter)

    # Find best trial and use it on the testset
    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(best_trial.last_result["accuracy"]))

    best_trained_model = CNN(best_trial.config["l1"], best_trial.config["l2"], best_trial.config["l3"])
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
    print("Best trial test set accuracy: {}".format(test_acc))

################################################################################


################################################################################
######################### Starting the training ################################
if __name__ == "__main__":
    main(num_samples=num_trials, max_num_epochs=num_epochs
        , gpus_per_trial=gpus_per_trial)
