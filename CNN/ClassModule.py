import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as utils
from os import cpu_count, path

# Function for loading data for normalization from file
def load_Normalization_Data(path=path.abspath('Data/normalization.npz')):
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


# Implementation of pytorch dataset class, loads the full dataset
# into ram. Can be used for datapreprocessing and augmentation.
# Check the pytorch documentation for detailed instructions on setting up a
# dataset class
class ClusterDataset_Full(utils.Dataset):
    """Cluster dataset."""
    # Initialize class and load data
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

        # Stack the features in a single array
        features = np.concatenate((_ClusterE, _ClusterPt, _ClusterM02, _ClusterM20, _ClusterDistFromVert))

        labels = { "ClusterType" : _ClusterType, "PartE" : _PartE, "PartPt" : _PartPt, "PartEta" : _PartEta, "PartPhi" : _PartPhi
                  , "PartIsPrimary" : _PartIsPrimary, "PartPID" : _PartPID }

        return (img, features, labels)

### Add Instance Noise to training image, can improve training
# https://arxiv.org/abs/1610.04490
def add_instance_noise(data, std=0.1):
    return data + 0.001 * torch.distributions.Normal(0, std).sample(data.shape)


#Helper functions for loading the dataset
def load_data_train(path=path.abspath('Data/data_train.npz')):
    ds_train = ClusterDataset_Full(path)
    return ds_train

def load_data_test(path=path.abspath('Data/data_test.npz')):
    ds_test = ClusterDataset_Full(path)
    return ds_test

# Helperfunction for loading the dataset. It is good to use functions for this,
# because pythons garbage collection will trigger at the end of a function call
# and clean up everything that is possible, i.e. free up memory and close files
def load_data(path=path.abspath('Data/data_train.npz')):
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


#Helper function used for getting the right dimension for input features [batch_size, 1] for linear layers saved in a dict
def unsqueeze_features(features):
    for key in features.keys():
        features[key] = features[key].view(-1,1)
    return features
