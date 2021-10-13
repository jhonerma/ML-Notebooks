#!/usr/bin/env python
# coding: utf-8

# ## Load Libraries

# In[1]:


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as utils

from functools import partial as func_partial
from functools import reduce as func_reduce
from operator import mul as op_mul
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import PopulationBasedTraining
from os import cpu_count, path
from time import strftime

#This class contains DatasetClass and several helper functions
import ClassModule as cm


# In[2]:


# Show number of avlaible CPU threads
# With mulithreading this number is twice the number of physical cores
cpu_av = cpu_count()
print("Number of available CPU's: {}".format(cpu_av))


# In[3]:


# Set the number CPUS that should be used per trial and dataloader
# If set to 1 number of cucurrent training networking is equal to this number
# In case of training with GPU this will be limited to number of models training simultaneously on GPU
# So number of CPU threads for each trial can be increased 
cpus_per_trial = 1
gpus_per_trial = 0


# In[4]:


def get_dataloader(train_ds, val_ds, bs):
    dl_train = utils.DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=cpus_per_trial-1)
    dl_val = utils.DataLoader(val_ds, batch_size=bs * 2, shuffle=True, num_workers=cpus_per_trial-1)
    return  dl_train, dl_val


# ## Instance Noise

# In[5]:


# https://arxiv.org/abs/1610.04490
INSTANCE_NOISE = False

def add_instance_noise(data, std=0.01):
    return data + torch.distributions.Normal(0, std).sample(data.shape).to(device)


# ## Define the network
# Get a network-candidate from the ASHA-scheduler first and use this notebook for hyperparameter tuning

# In[6]:


class CNN(nn.Module):
    def __init__(self, input_dim=(2,20,20), num_in_features=5):
        super(CNN, self).__init__()
        self.feature_ext = nn.Sequential(
            nn.Conv2d(2,10, kernel_size=3, padding=2),
            nn.ReLU(),
            nn.Conv2d(10,10, kernel_size=3,  padding=2),
            nn.ReLU(),
            nn.Conv2d(10,10, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(10,5, kernel_size=1, padding=0),
            nn.ReLU()
        )
        
        self.flatten = nn.Flatten()
        
        # Gives the number of features after the conv layer
        num_features_after_conv = func_reduce(op_mul, list(self.feature_ext(torch.rand(1, *input_dim)).shape))
        
        self.dense_nn = nn.Sequential(
            nn.Linear(num_features_after_conv + num_in_features, 4),
            nn.ReLU(),
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
            nn.ReLU()
        )
        
    def forward(self, cluster, clusNumXYEPt):
        cluster = self.feature_ext(cluster)
        x = self.flatten(cluster)
        x = torch.cat([x, clusNumXYEPt], dim=1)
        logits = self.dense_nn(x)
        return logits


# ## Implement train and validation loop
# [0: 'ClusterN', 1:'Cluster', 2:'ClusterTiming', 3:'ClusterType', 4:'ClusterE', 5:'ClusterPt', 6:'ClusterModuleNumber', 7:'ClusterRow', 8:'ClusterCol', 9:'ClusterM02', 10:'ClusterM20', 11:'ClusterDistFromVert', 12:'PartE', 13:'PartPt', 14:'PartEta', 15:'PartPhi', 16:'PartIsPrimary', 17:'PartPID']

# In[7]:


def train_loop(epoch, dataloader, model, loss_fn, optimizer, device="cpu"):

    size = len(dataloader.dataset)
    running_loss = 0.0
    epoch_steps = 0

    for batch, Data in enumerate(dataloader):
        Clusters = Data[0].to(device)
        Features = cm.unsqueeze_features(Data[1])
        Labels = Data[2]
        
        ClusterProperties = torch.cat([Features["ClusterE"], Features["ClusterPt"], Features["ClusterM02"]
                                      , Features["ClusterM20"], Features["ClusterDist"]], dim=1)
        ClusterProperties.to(device)
         
        if INSTANCE_NOISE:
            Clusters = add_instance_noise(Clusters)
            
        # zero parameter gradients
        optimizer.zero_grad()
        
        # prediction and loss
        pred = model(Clusters, ClusterProperties)
        loss = loss_fn(pred, Labels["PartPID"].long())
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        epoch_steps += 1
        
        if batch % 2000 == 1999:
            print("[%d, %5d] loss: %.3f" % (epoch + 1, batch + 1,
                                            running_loss / epoch_steps))
            running_loss = 0.0        


# In[8]:


def val_loop(epoch, dataloader, model, loss_fn, optimizer, device="cpu"):
    
    val_loss = 0.0
    val_steps = 0
    total = 0
    correct = 0
    size = len(dataloader.dataset)
    
    for batch, Data in enumerate(dataloader):
        with torch.no_grad():
            Clusters = Data[0].to(device)
            Features = cm.unsqueeze_features(Data[1])
            Labels = Data[2]
            ClusterProperties = torch.cat([Features["ClusterE"], Features["ClusterPt"], Features["ClusterM02"]
                                      , Features["ClusterM20"], Features["ClusterDist"]], dim=1)           
            ClusterProperties.to(device)
            
            pred = model(Clusters, ClusterProperties)
            correct += (pred.argmax(1) == Labels["PartPID"]).type(torch.float).sum().item()

            loss = loss_fn(pred, Labels["PartPID"].long())#.item()
            val_loss += loss.cpu().numpy()
            val_steps += 1
    
    with tune.checkpoint_dir(epoch) as checkpoint_dir:
        _path = path.join(checkpoint_dir, "checkpoint")
        torch.save((model.state_dict(), optimizer.state_dict()), _path)
        
    tune.report(loss=(val_loss / val_steps), accuracy= correct / size)


# ## Implement method for accuracy testing on test set

# In[9]:


def test_accuracy(model, device="cpu"):
    
    dataset_test = cm.load_data_test()
    
    dataloader_test = utils.DataLoader(
        dataset_test, batch_size=4, shuffle=False, num_workers=2)
    
    correct = 0
    total = len(dataloader_test.dataset)
    
    with torch.no_grad():
        for batch, Data in enumerate(dataloader_test):
            Clusters = Data[0].to(device)
            Features = cm.unsqueeze_features(Data[1])
            Labels = Data[2]
            ClusterProperties = torch.cat([Features["ClusterE"], Features["ClusterPt"], Features["ClusterM02"]
                                      , Features["ClusterM20"], Features["ClusterDist"]], dim=1)            
            ClusterProperties.to(device)
            
            
            pred = model(Clusters, ClusterProperties)
            correct += (pred.argmax(1) == Labels["PartPID"]).type(torch.float).sum().item()

    return correct / total


# ## Implement training routine

# In[13]:


def train_model(config, checkpoint_dir=None):
    
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
        model_state, optimizer_state = torch.load(
            path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
        
    # load dataset
    dataset_train = cm.load_data_train()
    
    # split trainset in train and validation subsets
    test_abs = int(len(dataset_train) * 0.8)
    subset_train, subset_val = utils.random_split(
        dataset_train, [test_abs, len(dataset_train) - test_abs])

    # get dataloaders 
    dataloader_train, dataloader_val = get_dataloader(subset_train, subset_val, int(config["batch_size"]))
                                                      
    for epoch in range(100):
        train_loop(epoch, dataloader_train, model, loss_fn, optimizer, device=device)
        val_loop(epoch, dataloader_train, model, loss_fn, optimizer, device=device)                                              
    
    print("Finished Training")


# ## Setup all Ray Tune functionality and start training

# In[17]:


def main(num_samples=10, max_num_epochs=10, gpus_per_trial=1):
    
    # Setup hyperparameter-space to search
    config = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "wd" : tune.uniform(0, 1e-1),
        "batch_size": tune.choice([2, 4, 8, 16, 32, 64])
    }

    # Init the scheduler
    scheduler = PopulationBasedTraining(
    time_attr="training_iteration",
    metric="accuracy",
    mode="max",
    perturbation_interval=10)
        
    
    # Init the Reporter
    reporter = CLIReporter(
        parameter_columns=["lr", "wd", "batch_size"],
        metric_columns=["loss", "accuracy", "training_iteration"])
    
    #Get Current date and time
    timestr = strftime("%Y_%m_%d-%H:%M:%S")
    name = "ASHA-" + timestr
    
    # Init the run method
    result = tune.run(
        func_partial(train_model),
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
    
    best_trained_model = CNN()
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


# In[18]:


main(num_samples=10, max_num_epochs=100, gpus_per_trial=gpus_per_trial)


# In[ ]:




