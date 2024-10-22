import torch
import numpy as np
import soundfile as sf
import librosa
import os
import pickle

class Data(torch.utils.data.Dataset):
    def __init__(self, data, Train=False):
        assert len(data['Data']) == len(data['Labels'])
        self.Train = Train
        
        #Load Data...
        self.Data = data['Data']
        self.Labels = data['Labels']
    
    def __len__(self):
        return len(self.Data)
    
    def __getitem__(self, idx):
        if self.Train:
            wav, fs = sf.read(self.Data[idx])
            Data = librosa.core.to_mono(wav.transpose()).transpose()
            new_size = 288000
            reps = int(np.ceil(new_size/Data.shape[0]))
            offset = np.random.randint(low=0, high=int(reps*Data.shape[0]-new_size+1))
            Data = np.tile(Data, reps=reps)[offset:offset+new_size].astype(np.float32)
        
        else:
            Data = self.Data[idx,:]
        
        return {'Data': torch.from_numpy(Data), 'Labels': torch.Tensor([self.Labels[idx]]).int()}
    
