import torch
import numpy as np
import os
import json
import soundfile as sf
import librosa
from collections import defaultdict
from tqdm import tqdm

def get_stats(target_sr=16000, data_year='./2022-data/', source_dir="./"):
    print('Loading train data')
    Evcategories = os.listdir(source_dir + data_year + "dev_data")
    Tscategories = os.listdir(source_dir + data_year + "eval_data")
    if data_year=='./2023-data/':
        Trcategories = Evcategories + Tscategories 
        RawShape = 288000
    else:
        Trcategories = Evcategories.copy()
        RawShape = 288000
        #RawShape = 10*target_sr
    
    train_raw = np.empty((0, RawShape), dtype=np.float32)
    Dicts = [data_year + 'dev_data/', data_year + 'eval_data/']
    eps = 1e-12
    print("Categories:",Trcategories)
    for label, category in enumerate(Trcategories):
        print(category)
        if data_year=='./2023-data/':
            dicts = [Dicts[0]] if category in set(Evcategories) else [Dicts[1]] 
        else:
            dicts = Dicts.copy()
        raw_arrays = defaultdict(list)
        for dict in dicts:
            train_raw_list = []
            for count, file in tqdm(enumerate(os.listdir(source_dir + dict + category + "/train")), total=len(os.listdir(source_dir + dict + category + "/train"))):
                if file.endswith(".wav"):
                    file_path = source_dir + dict + category + "/train/" + file
                    wav, fs = sf.read(file_path)
                    raw = librosa.core.to_mono(wav.transpose()).transpose()
                    if True:#data_year == './2023-data/':
                        new_size = 288000
                        reps = int(np.ceil(new_size/raw.shape[0]))
                        offset = np.random.randint(low=0, high=int(reps*raw.shape[0]-new_size+1))
                        raw = np.tile(raw, reps=reps)[offset:offset+new_size].astype(np.float32)
                    else:
                        raw = raw[:10 * target_sr]#.reshape(1,-1)
                    train_raw_list.append(raw)
            raw_arrays[dict] = np.array(train_raw_list, dtype=np.float32)
        train_raw = np.concatenate((train_raw, np.concatenate(list(raw_arrays.values()), axis=0)), axis=0)
    
    raw_arrays = None
    print("Calculating Stats...")
    Mean = np.mean(train_raw, axis=0)
    STD = np.std(train_raw, axis=0)
    
    print("Mean:",Mean)
    print("Standard Deviation:",STD)
    np.save(source_dir + data_year + "saved_data/data_mean.npy",Mean)
    np.save(source_dir + data_year + "saved_data/data_std.npy",STD)
    return Mean, STD



if __name__ == '__main__':
    get_stats(source_dir="../")
