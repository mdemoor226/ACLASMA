import torch
import numpy as np
import soundfile as sf
import librosa
from .smote import SMOTEFAST
import os
import pickle
from .data_stats import get_stats

class Data(torch.utils.data.Dataset):
    def __init__(self, data, Train=False, SMOTE=False, Target_sr=16000, smotecfg=None, year=2023):
        assert len(data['Data']) == len(data['Labels'])
        self.Train = Train
        self.SMOTE = SMOTE
        self.target_sr = Target_sr
        self.year = year
        
        #Data Normalization/Standardization Stats
        """
        if os.path.isfile("./saved_data/data_mean.npy"):
            mean = np.load("./saved_data/data_mean.npy")
            std = np.load("./saved_data/data_std.npy")
            if mean.shape[0] / Target_sr != 10:
                print("Error, saved dataset statistics are not compatible. Updating now... (Make sure you have 64+ gbs of RAM")
                mean, std = get_stats(Target_sr)
        else:
            mean, std = get_stats(Target_sr)
        
        self.data_mean = mean
        self.data_std = std
        #"""
        #Load Data...
        if self.SMOTE:
            assert Train, "SMOTE not available for testing...."
            assert smotecfg is not None
            if os.path.isfile("./saved_data/SmoteMeta.pkl"):
                with open("./saved_data/SmoteMeta.pkl", "rb") as SmoteFile:
                    SmoteMeta = pickle.load(SmoteFile)

                DataExtend = SmoteMeta['Data']
                LabelsExtend = SmoteMeta['Labels']
                N_Indices = SmoteMeta['Neighbors']
                Steps = SmoteMeta['Steps']
                Cutoff = SmoteMeta['Cutoff']
                self.oversample = SMOTEFAST(target_sr=self.target_sr, knn_count=smotecfg['k_neighbors'], ignore_topk=smotecfg['ignore_topk'], k_neighbors=smotecfg['k_neighbors'])
            else:
                Data = data['Data']#.tolist()
                Labels = data['Labels']
                #This shouldn't require the same repeated argument. Fix it? 
                self.oversample = SMOTEFAST(target_sr=self.target_sr, knn_count=smotecfg['k_neighbors'], ignore_topk=smotecfg['ignore_topk'], k_neighbors=smotecfg['k_neighbors'])
                DataExtend, LabelsExtend, N_Indices, Steps, Cutoff = self.oversample.fit_resample(Data, Labels)
                with open("./saved_data/SmoteMeta.pkl", "wb") as SmoteFile:
                    SmoteMeta = {'Data': DataExtend, 'Labels': LabelsExtend, 'Neighbors': N_Indices, 'Steps': Steps, 'Cutoff': Cutoff}
                    pickle.dump(SmoteMeta, SmoteFile)
            
            self.Data = DataExtend
            self.Labels = LabelsExtend
            self.N_Indices = N_Indices
            self.Steps = Steps
            self.Cutoff = Cutoff
        else:
            self.Data = data['Data']
            self.Labels = data['Labels']
            self.Cutoff = len(data['Data'])
    
    def __len__(self):
        return len(self.Data)
    
    def __getitem__(self, idx):
        if self.Train:
            #print("File Path:",self.Data[idx])
            #input()
            wav, fs = sf.read(self.Data[idx])
            Data = librosa.core.to_mono(wav.transpose()).transpose()
            if True:#self.year == 2023:
                new_size = 288000
                reps = int(np.ceil(new_size/Data.shape[0]))
                offset = np.random.randint(low=0, high=int(reps*Data.shape[0]-new_size+1))
                Data = np.tile(Data, reps=reps)[offset:offset+new_size].astype(np.float32)
            else:
                Data = Data[:10 * self.target_sr].astype(np.float32)#.reshape(1,-1)
            if self.SMOTE and idx >= self.Cutoff:
                #if x in minority class: # Target Domain <<<<------- AND TARGET DOMAIN??????!!!!!?????
                #X = Minority Class Samples
                #rows = which minority class sample to use
                #cols = which nearest neighbor of/to the minority class sample to use
                #nn_num = catalog of nearest neighbor indices
                #nn_data = nearest neighbors themselvs
                #steps = ...
                NN_wav, NN_fs = sf.read(self.Data[:self.Cutoff][self.Labels[:self.Cutoff] == self.Labels[idx]][self.N_Indices[idx]])
                NN_Data = librosa.core.to_mono(NN_wav.transpose()).transpose()[:10 * self.target_sr].astype(np.float32)#.reshape(1,-1)
                Diff = Data - NN_Data
                Data = Data + self.Steps[idx].astype(np.float32)*Diff
                Data = Data.astype(Data.dtype)
        
        else:
            Data = self.Data[idx,:]
        
        #Data = (Data - self.data_mean) / self.data_std
        
        return {'Data': torch.from_numpy(Data), 'Labels': torch.Tensor([self.Labels[idx]]).int()}
    
    def refresh(self, Model):
        DataExtend, LabelsExtend, N_Indices, Steps, Cutoff = self.oversample.fit_resample(self.Data[:self.Cutoff], self.Labels[:self.Cutoff], Model)
        with open("./saved_data/SmoteMeta.pkl", "wb") as SmoteFile:
            SmoteMeta = {'Data': DataExtend, 'Labels': LabelsExtend, 'Neighbors': N_Indices, 'Steps': Steps, 'Cutoff': Cutoff}
            pickle.dump(SmoteMeta, SmoteFile)
        
        self.Data = DataExtend
        self.Labels = LabelsExtend
        self.N_Indices = N_Indices
        self.Steps = Steps
        self.Cutoff = Cutoff
