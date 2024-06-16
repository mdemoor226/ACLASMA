import torch
import numpy as np
import os
import json
import soundfile as sf
import librosa
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import sklearn.preprocessing as sklp
from scipy.stats import hmean
from collections import defaultdict

#From the ProxyNCA++ Repository
def load_config(config_name="config.json"):
    config = json.load(open(config_name))
    def eval_json(config):
        for k in config:
            if type(config[k]) == list:
                config[k] = [eval(value) for value in config[k]]
            elif type(config[k]) != dict:
                config[k] = eval(config[k])
            else:
                eval_json(config[k])

    eval_json(config)
    return config

# Borrowed from Wilkinghoff
def adjust_size(wav, new_size):
    reps = int(np.ceil(new_size/wav.shape[0]))
    offset = np.random.randint(low=0, high=int(reps*wav.shape[0]-new_size+1))
    return np.tile(wav, reps=reps)[offset:offset+new_size]

def prepare_data(year='./2022-data/', target_sr=16000):
    # load train data
    print('Loading train data')
    Evcategories = os.listdir(year + "dev_data")
    Tscategories = os.listdir(year + "eval_data")
    Trcategories = Evcategories + Tscategories if year=='./2023-data/' else Evcategories.copy()

    if os.path.isfile(year + './saved_data/train_ids.npy'):#' + str(target_sr) + '_train_raw.npy'):
        #train_raw = np.load('./saved_data/' + str(target_sr) + '_train_raw.npy')
        train_ids = np.load(year + 'saved_data/train_ids.npy')
        train_files = np.load(year + 'saved_data/train_files.npy')
        train_atts = np.load(year + 'saved_data/train_atts.npy')
        train_domains = np.load(year + 'saved_data/train_domains.npy')
    else:
        if not os.path.exists(year + 'saved_data'):
            os.mkdir(year + 'saved_data')

        #train_raw = np.empty((0, 10*target_sr), dtype=np.float32)
        train_ids = []
        train_files = []
        train_atts = []
        train_domains = []
        Dicts = [year + 'dev_data/', year + 'eval_data/']
        eps = 1e-12
        print("Categories:",Trcategories)
        for label, category in enumerate(Trcategories):
            print(category)
            if year=='./2023-data/':
                dicts = [Dicts[0]] if category in set(Evcategories) else [Dicts[1]] 
            else:
                dicts = Dicts.copy()
            
            #raw_arrays = defaultdict(list)
            for dict in dicts:
                #print(dict + category + "/train")
                #import code
                #code.interact(local=locals())
                train_raw_list = []
                for count, file in tqdm(enumerate(os.listdir(dict + category + "/train")), total=len(os.listdir(dict + category + "/train"))):
                    if file.endswith(".wav"):
                        file_path = dict + category + "/train/" + file
                        #wav, fs = sf.read(file_path)
                        #raw = librosa.core.to_mono(wav.transpose()).transpose()[:10 * target_sr]#.reshape(1,-1)
                        #train_raw_list.append(raw)
                        train_ids.append(category + '_' + file.split('_')[1])
                        train_files.append(file_path)
                        train_domains.append(file.split('_')[2])
                        train_atts.append('_'.join(file.split('.wav')[0].split('_')[6:]))
                #raw_arrays[dict] = np.array(train_raw_list, dtype=np.float32)
            #train_raw = np.concatenate((train_raw, np.concatenate(list(raw_arrays.values()), axis=0)), axis=0)
        # reshape arrays and store
        train_ids = np.array(train_ids)
        train_files = np.array(train_files)
        train_atts = np.array(train_atts)
        train_domains = np.array(train_domains)
        print("Saving extracted data...")
        np.save(year + 'saved_data/train_ids.npy', train_ids)
        np.save(year + 'saved_data/train_files.npy', train_files)
        np.save(year + 'saved_data/train_atts.npy', train_atts)
        np.save(year + 'saved_data/train_domains.npy', train_domains)
        #np.save('./saved_data/' + str(target_sr) + '_train_raw.npy', train_raw)
    
    print("Success!")
    # load evaluation data
    print('Loading evaluation data')
    if os.path.isfile(year + 'saved_data/' + str(target_sr) + '_eval_raw.npy'):
        eval_raw = np.load(year + 'saved_data/' + str(target_sr) + '_eval_raw.npy')
        eval_ids = np.load(year + 'saved_data/eval_ids.npy')
        eval_normal = np.load(year + 'saved_data/eval_normal.npy')
        eval_files = np.load(year + 'saved_data/eval_files.npy')
        eval_atts = np.load(year + 'saved_data/eval_atts.npy')
        eval_domains = np.load(year + 'saved_data/eval_domains.npy')
    else:
        eval_raw = []
        eval_ids = []
        eval_normal = []
        eval_files = []
        eval_atts = []
        eval_domains = []
        eps = 1e-12
        for label, category in enumerate(Evcategories):
            print(category)
            for count, file in tqdm(enumerate(os.listdir(year + "dev_data/" + category + "/test")), total=len(os.listdir(year + "dev_data/" + category + "/test"))):
                if file.endswith(".wav"):
                    file_path = year + "dev_data/" + category + "/test/" + file
                    wav, fs = sf.read(file_path)
                    raw = librosa.core.to_mono(wav.transpose()).transpose()[:10 * target_sr]
                    if year=='./2023-data/':
                        raw = adjust_size(raw, 288000)
                    else:
                        raw = raw[:10 * target_sr]
                    eval_raw.append(raw)
                    eval_ids.append(category + '_' + file.split('_')[1])
                    eval_normal.append(file.split('_test_')[1].split('_')[0] == 'normal')
                    eval_files.append(file_path)
                    eval_domains.append(file.split('_')[2])
                    eval_atts.append('_'.join(file.split('.wav')[0].split('_')[6:]))
        # reshape arrays and store
        eval_ids = np.array(eval_ids)
        eval_normal = np.array(eval_normal)
        eval_files = np.array(eval_files)
        eval_atts = np.array(eval_atts)
        eval_domains = np.array(eval_domains)
        #eval_raw = np.expand_dims(np.array(eval_raw, dtype=np.float32), axis=-1)
        eval_raw = np.array(eval_raw, dtype=np.float32)
        print("Saving extracted data...")
        np.save(year + 'saved_data/eval_ids.npy', eval_ids)
        np.save(year + 'saved_data/eval_normal.npy', eval_normal)
        np.save(year + 'saved_data/eval_files.npy', eval_files)
        np.save(year + 'saved_data/eval_atts.npy', eval_atts)
        np.save(year + 'saved_data/eval_domains.npy', eval_domains)
        np.save(year + 'saved_data/' + str(target_sr) + '_eval_raw.npy', eval_raw)
 
    print("Success!")
    # load test data
    print('Loading test data')
    if os.path.isfile(year + 'saved_data/' + str(target_sr) + '_test_raw.npy'):
        test_raw = np.load(year + 'saved_data/' + str(target_sr) + '_test_raw.npy')
        test_ids = np.load(year + 'saved_data/test_ids.npy')
        test_files = np.load(year + 'saved_data/test_files.npy')
    else:
        test_raw = []
        test_ids = []
        test_files = []
        eps = 1e-12
        for label, category in enumerate(Tscategories):
            print(category)
            for count, file in tqdm(enumerate(os.listdir(year + "eval_data/" + category + "/test")),
                                    total=len(os.listdir(year + "eval_data/" + category + "/test"))):
                if file.endswith(".wav"):
                    file_path = year + "eval_data/" + category + "/test/" + file
                    wav, fs = sf.read(file_path)
                    raw = librosa.core.to_mono(wav.transpose()).transpose()
                    #import code
                    #code.interact(local=locals())
                    if year=='./2023-data/':
                        raw = adjust_size(raw, 288000)
                    else:
                        raw = raw[:10 * target_sr]
                    #if count in {0, 1}:
                    #    import code
                    #    code.interact(local=locals())
                    test_raw.append(raw)
                    test_ids.append(category + '_' + file.split('_')[1])
                    test_files.append(file_path)
        #import code
        #code.interact(local=locals())
        # reshape arrays and store
        test_ids = np.array(test_ids)
        test_files = np.array(test_files)
        ##test_raw = np.expand_dims(np.array(test_raw, dtype=np.float32), axis=-1)
        test_raw = np.array(test_raw, dtype=np.float32)
        print("Saving extracted data...")
        np.save(year + 'saved_data/test_ids.npy', test_ids)
        np.save(year + 'saved_data/test_files.npy', test_files)
        np.save(year + 'saved_data/' + str(target_sr) + '_test_raw.npy', test_raw)

    print("Success!")
    
    # encode ids as labels
    #le_4train = LabelEncoder()
    #
    #source_train = np.array([file.split('_')[3] == 'source' for file in train_files.tolist()])
    #source_eval = np.array([file.split('_')[3] == 'source' for file in eval_files.tolist()])
    #train_ids_4train = np.array(['###'.join([train_ids[k], train_atts[k], str(source_train[k])]) for k in np.arange(train_ids.shape[0])])
    #eval_ids_4train = np.array(['###'.join([eval_ids[k], eval_atts[k], str(source_eval[k])]) for k in np.arange(eval_ids.shape[0])])
    #le_4train.fit(np.concatenate([train_ids_4train, eval_ids_4train], axis=0))
    #num_classes_4train = len(np.unique(np.concatenate([train_ids_4train, eval_ids_4train], axis=0)))
    #train_labels_4train = le_4train.transform(train_ids_4train)
    #eval_labels_4train = le_4train.transform(eval_ids_4train)
    #eval_raw = eval_raw[eval_normal]
    #eval_labels_4train = eval_labels_4train[eval_normal]

    # encode ids as labels
    le_4train = LabelEncoder()

    source_train = np.array([file.split('_')[3] == 'source' for file in train_files.tolist()])
    source_eval = np.array([file.split('_')[3] == 'source' for file in eval_files.tolist()])
    train_ids_4train = np.array(['###'.join([train_ids[k], train_atts[k], str(source_train[k])]) for k in np.arange(train_ids.shape[0])])
    eval_ids_4train = np.array(['###'.join([eval_ids[k], eval_atts[k], str(source_eval[k])]) for k in np.arange(eval_ids.shape[0])])
    le_4train.fit(np.concatenate([train_ids_4train, eval_ids_4train], axis=0))
    num_classes_4train = len(np.unique(np.concatenate([train_ids_4train, eval_ids_4train], axis=0)))
    train_labels_4train = le_4train.transform(train_ids_4train)
    eval_labels_4train = le_4train.transform(eval_ids_4train)

    le = LabelEncoder()
    train_labels = le.fit_transform(train_ids)
    eval_labels = le.transform(eval_ids)
    test_labels = le.transform(test_ids)
    num_classes = len(np.unique(train_labels))

    # distinguish between normal and anomalous samples on development set
    unknown_raw = eval_raw[~eval_normal]
    unknown_labels = eval_labels[~eval_normal]
    unknown_labels_4train = eval_labels_4train[~eval_normal]
    unknown_files = eval_files[~eval_normal]
    unknown_ids = eval_ids[~eval_normal]
    unknown_domains = eval_domains[~eval_normal]
    source_unknown = source_eval[~eval_normal]
    eval_raw = eval_raw[eval_normal]
    eval_labels = eval_labels[eval_normal]
    eval_labels_4train = eval_labels_4train[eval_normal]
    eval_files = eval_files[eval_normal]
    eval_ids = eval_ids[eval_normal]
    eval_domains = eval_domains[eval_normal]
    source_eval = source_eval[eval_normal]
    
    IDs = {'Train': train_ids, 'Eval': eval_ids, 'Test': test_ids}
    Sources = {'Train': source_train, 'Eval': source_eval, 'Unknown': source_unknown}
    Labels = {'Train': train_labels, 'Eval': eval_labels, 'Unknown': unknown_labels, 'Test': test_labels}
    RawShapes = {'Train': None, 'Eval': eval_raw.shape, 'Unknown': unknown_raw.shape, 'Test': test_raw.shape}
    Categories = {'Train': Trcategories, 'Eval': Evcategories, 'Test': Tscategories}
    Meta = {'categories': Categories, 'num_trclasses': num_classes_4train, 'label_encoder': le, 'IDs': IDs, 'Sources': Sources, 'Labels': Labels, 'Shapes': RawShapes}
    #print(num_classes_4train)
    #input("See above...")
    #import code
    #trclasses = train_labels_4train
    #code.interact(local=locals())
    
    TrData = {'Data': train_files, 'Labels': train_labels_4train} 
    EvData = {'Data': eval_raw, 'Labels': eval_labels_4train}
    UnkData = {'Data': unknown_raw, 'Labels': unknown_labels_4train}
    TsData = {'Data': test_raw, 'Labels': test_labels, 'Files': test_files}
    return TrData, EvData, UnkData, TsData, Meta



class BalancedBatchSampler(torch.utils.data.sampler.Sampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """
    def __init__(self, labels, n_samples, batch_size):
        self.labels = labels
        self.labels_set = list(set(self.labels))
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = batch_size // n_samples#n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = batch_size

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][self.used_label_indices_count[class_]:self.used_label_indices_count[class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
                
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size


class BalancedFullBatchSampler(torch.utils.data.sampler.Sampler):
    """
    Like the one above but keeps sampling classes until the Full Batch Size is satisfied.
    """
    def __init__(self, labels, n_samples, batch_size):
        self.labels = labels
        self.labels_set = list(set(self.labels))
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = batch_size

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            indices = []
            while len(indices) < self.batch_size:
                num_classes = max(1,(self.batch_size - len(indices)) // self.n_samples)
                classes = np.random.choice(self.labels_set, num_classes, replace=False)
                for class_ in classes:
                    num_samples = self.n_samples if len(indices) + self.n_samples <= self.batch_size else self.batch_size - len(indices)
                    indices.extend(self.label_to_indices[class_][self.used_label_indices_count[class_]:self.used_label_indices_count[class_] + num_samples])
                    self.used_label_indices_count[class_] += num_samples
                    if self.used_label_indices_count[class_] + num_samples > len(self.label_to_indices[class_]):
                        np.random.shuffle(self.label_to_indices[class_])
                        self.used_label_indices_count[class_] = 0
                    
            yield indices
            self.count += self.batch_size

    def __len__(self):
        return self.n_dataset // self.batch_size



