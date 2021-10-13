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
from ray.tune.schedulers import ASHAScheduler
from os import cpu_count, path
from time import strftime

from filelock import FileLock

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
cpus_per_trial = 2
gpus_per_trial = 0.33


# In[4]:


def get_dataloader(train_ds, val_ds, bs):
    
    # We add FileLock here because multiple workers will want to
    # download data, and this may cause overwrites since
    # DataLoader is not threadsafe.
    #with FileLock(path.expanduser("/media/DATA/ML-Notebooks/CNN/Data/.data.lock")):
    dl_train = utils.DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=cpus_per_trial-1)
    dl_val = utils.DataLoader(val_ds, batch_size=bs * 2, shuffle=True, num_workers=cpus_per_trial-1)
    
    return  dl_train, dl_val


# ## Instance Noise

# In[5]:


# https://arxiv.org/abs/1610.04490
INSTANCE_NOISE = True

def add_instance_noise(data, device, std=0.01):
    return data + torch.distributions.Normal(0, std).sample(data.shape).to(device)


# ## Define the network

# In[6]:


class CNN(nn.Module):
    def __init__(self, l1=100, l2=50, l3=25, input_dim=(2,20,20), num_in_features=5):
        super(CNN, self).__init__()
        self.feature_ext = nn.Sequential(
            nn.Conv2d(2,10, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(10,10, kernel_size=5, padding=0),
            nn.ReLU(),  
            nn.MaxPool2d(2),
            nn.Conv2d(10,10, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(10,6, kernel_size=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.flatten = nn.Flatten()
        
        # Gives the number of features after the conv layer
        num_features_after_conv = func_reduce(op_mul, list(self.feature_ext(torch.rand(1, *input_dim)).shape))
        
        self.dense_nn = nn.Sequential(
            nn.Linear(num_features_after_conv + num_in_features, l1),
            nn.ReLU(),
            nn.Linear(l1, l2),
            nn.ReLU(),
            nn.Linear(l2, l3),
            nn.ReLU(),
            nn.Linear(l3,3),
            nn.ReLU()
        )
        
    def forward(self, cluster, clusNumXYEPt):
        cluster = self.feature_ext(cluster)
        x = self.flatten(cluster)
        x = torch.cat([x, clusNumXYEPt], dim=1)
        logits = self.dense_nn(x)
        return logits


# ## Implement train and validation loop
# Data[0] contains an image of of the cell energies and timings. <br>
# Data[1] contains all features in a dict. Their shapes have to be changed from [batch_size] to [batch_size,1] for input into linear layers, implemented via function here <br>
# Data[2] contains all labels

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
        
        if batch % 100000 == 99999:
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


# ## Implement method for accuracy testing on test set

# In[9]:


def test_accuracy(model, device="cpu"):
    
    dataset_test = cm.load_data_test('/media/DATA/ML-Notebooks/CNN/Data/data_test.npz')
    
    dataloader_test = utils.DataLoader(
        dataset_test, batch_size=4, shuffle=False, num_workers=cpu_av-1)
    
    correct = 0
    total = len(dataloader_test.dataset)
    
    with torch.no_grad():
        for batch, Data in enumerate(dataloader_test):
            Clusters = Data[0].to(device)
            Features = cm.unsqueeze_features(Data[1])
            Labels = Data[2]
            ClusterProperties = torch.cat([Features["ClusterE"], Features["ClusterPt"], Features["ClusterM02"]
                                      , Features["ClusterM20"], Features["ClusterDist"]], dim=1).to(device)            
            #Labels = torch.cat([Labels["PartPID"], dim=1]).to(device)
            Label = Labels["PartPID"].to(device)
            
            
            pred = model(Clusters, ClusterProperties)
            correct += (pred.argmax(1) == Label).type(torch.float).sum().item()

    return correct / total


# ## Implement training routine

# In[10]:


def train_model(config, dataloader_train=None, dataloader_test=None, checkpoint_dir=None):
    
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
    dataset_train = cm.load_data_train('/media/DATA/ML-Notebooks/CNN/Data/data_train.npz')
    
    # split trainset in train and validation subsets
    test_abs = int(len(dataset_train) * 0.8)
    subset_train, subset_val = utils.random_split(
        dataset_train, [test_abs, len(dataset_train) - test_abs])

    # get dataloaders 
    dataloader_train, dataloader_val = get_dataloader(subset_train, subset_val, int(config["batch_size"]))
                                                      
    for epoch in range(100):
        train_loop(epoch, dataloader_train, model, loss_fn, optimizer, device=device)
        val_loop(epoch, dataloader_val, model, loss_fn, optimizer, device=device)                                              
    
    print("Finished Training")


# ## Setup all Ray Tune functionality and start training

# In[11]:


def main(num_samples=10, max_num_epochs=10, gpus_per_trial=1):
    
    # Setup hyperparameter-space to search
    config = {
        "l1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 8)),
        "l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 8)),
        "l3": tune.sample_from(lambda _: 2 ** np.random.randint(2, 8)),
        "lr": tune.loguniform(1e-4, 1e-1),
        "wd": tune.loguniform(1e-5, 1e-3),
        "batch_size": tune.choice([16, 32, 64, 128, 256])
    }

    # Init the scheduler
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    
    # Init the Reporter
    reporter = CLIReporter(
        parameter_columns=["l1", "l2", "l3", "lr","wd", "batch_size"],
        metric_columns=["loss", "accuracy", "training_iteration"])
    
    #Get Current date and time
    timestr = strftime("%Y_%m_%d-%H:%M:%S")
    name = "ASHA-" + timestr
    
    #Load the dataset for use in shared memory
    #data = cm.load_data_train('/media/DATA/ML-Notebooks/CNN/Data/data_train.npz')
    #test_abs = int(len(data) * 0.8)
    #subset_train, subset_val = utils.random_split(
    #    data, [test_abs, len(data) - test_abs])

    # get dataloaders 
    #dataloader_train, dataloader_val = get_dataloader(subset_train, subset_val, 64)
    
    # Init the run method
    result = tune.run(
        func_partial(train_model),
        #tune.with_parameters(train_model, dataset_train=data),
        #tune.with_parameters(train_model, dataloader_train=dataloader_train, dataloader_val=dataloader_val),
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


# In[12]:


main(num_samples=10, max_num_epochs=30, gpus_per_trial=gpus_per_trial)


# In[ ]:




