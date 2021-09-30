import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as utils

#DatasetClass for partialy importing dataset while loading
# Needs reworking opening and closing the file too many times with multithreading
# which leads to crashing
# if clusters were saved as seperate images it would make sense to load them
class ClusterDataset_Partial(utils.Dataset):
    """Cluster dataset."""

    def __init__(self, npz_file, arrsize=20):
        """
        Args:
            npz_file (string): Path to the npz file.
        """
        self.data = np.load(npz_file, allow_pickle=True)
        self.arrsize = arrsize
        

    def __len__(self):
        return self.data["Size"]
    
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
    
    def __GetClusters(self, ncell, modnum, row, col, energy, timing):
        
        cluster_e = self.__ReconstructCluster(ncell, modnum, row, col, energy)
        cluster_t = self.__ReconstructCluster(ncell, modnum, row, col, timing)

        return np.stack([cluster_e, cluster_t], axis=1)
    
    def __ChangePID(self, PID):
        if (PID != 111) & (PID != 221):
            PID = np.int16(0)
        if PID == 111:
            PID = np.int16(1)
        if PID == 221:
            PID = np.int16(2)
        return PID

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        ClusterN = self.data['ClusterN'][idx]
        Cluster = self.data['Cluster'][idx]
        ClusterTiming = self.data['ClusterTiming'][idx]
        ClusterType = self.data['ClusterType'][idx]
        ClusterE = self.data['ClusterE'][idx]
        ClusterPt = self.data['ClusterPt'][idx]
        ClusterModuleNumber = self.data['ClusterModuleNumber'][idx]
        ClusterCol = self.data['ClusterCol'][idx]
        ClusterRow = self.data['ClusterRow'][idx]
        ClusterM02 = self.data['ClusterM02'][idx]
        ClusterM20 = self.data['ClusterM20'][idx]
        ClusterDistFromVert = self.data['ClusterDistFromVert'][idx]
        PartE = self.data['PartE'][idx]
        PartPt = self.data['PartPt'][idx]
        PartEta = self.data['PartEta'][idx]
        PartPhi = self.data['PartPhi'][idx]
        PartIsPrimary = self.data['PartIsPrimary'][idx]
        PartPID = self.data['PartPID'][idx]
       
        PartPID = self.__ChangePID(PartPID)
        
        img = self.__GetClusters(ClusterN, ClusterModuleNumber, ClusterRow, ClusterCol, Cluster, ClusterTiming)
        img = torch.from_numpy(img)
        
        features = { "ClusterType" : ClusterType, "ClusterE" : ClusterE, "ClusterPt" : ClusterPt
                    , "ClusterM02" : ClusterM02, "ClusterM20" : ClusterM20 , "ClusterDistFromVert" : ClusterDistFromVert}
        labels = { "PartE" : PartE, "PartPt" : PartPt, "PartEta" : PartEta, "PartPhi" : PartPhi
                  , "PartIsPrimary" : PartIsPrimary, "PartPID" : PartPID }
        
        return (img, features, labels)
        
        
# Load the full dataset into ram       
class ClusterDataset_Full(utils.Dataset):
    """Cluster dataset."""

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

    def __len__(self):
        return self.data["Size"]
    
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
    
    def __GetClusters(self, ncell, modnum, row, col, energy, timing):       
        cluster_e = self.__ReconstructCluster(ncell, modnum, row, col, energy)
        cluster_t = self.__ReconstructCluster(ncell, modnum, row, col, timing)
        return np.stack([cluster_e, cluster_t], axis=0)
    
    def __ChangePID(self, PID):
        if (PID != 111) & (PID != 221):
            PID = np.int16(0)
        if PID == 111:
            PID = np.int16(1)
        if PID == 221:
            PID = np.int16(2)
        return PID

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
        
        img = self.__GetClusters(ClusterN, ClusterModuleNumber, ClusterRow, ClusterCol, Cluster, ClusterTiming)
        
        features = { "ClusterType" : ClusterType, "ClusterE" : ClusterE, "ClusterPt" : ClusterPt
                    , "ClusterM02" : ClusterM02, "ClusterM20" : ClusterM20 , "ClusterDist" : ClusterDistFromVert}
        labels = { "PartE" : PartE, "PartPt" : PartPt, "PartEta" : PartEta, "PartPhi" : PartPhi
                  , "PartIsPrimary" : PartIsPrimary, "PartPID" : PartPID }
        
        return (img, features, labels)
        

# Load data for normalization from file        
def loadNormalizationData():
    data = np.load('Data/normalization.npz', allow_pickle=True)
    
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
               
    return maxData, minData


#Helper functions for loading the dataset    
def load_data_train(path='/home/jhonerma/ML-Notebooks/CNN/Data/data_train.npz'):
    ds_train = ClusterDataset_Full(path)
    return ds_train

def load_data_test(path='/home/jhonerma/ML-Notebooks/CNN/Data/data_test.npz'):
    ds_test = ClusterDataset_Full(path)
    return ds_test

def load_data_train_partial(path='/home/jhonerma/ML-Notebooks/CNN/Data/data_train.npz'):
    ds_train = ClusterDataset_Partial(path)
    return ds_train

def load_data_test_partial(path='/home/jhonerma/ML-Notebooks/CNN/Data/data_test.npz'):
    ds_test = ClusterDataset_Partial(path)
    return ds_test
    
    
#Helper function used for getting the right dimension for input features [batch_size, 1] for linear layers saved in a dict    
def unsqueeze_features(features):
    for key in features.keys():
        features[key] = features[key].view(-1,1)        
    return features
