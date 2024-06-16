import torch
import numpy as np
import soundfile as sf
import librosa
from imblearn.over_sampling import SMOTE
from sklearn.utils.multiclass import check_classification_targets
from imblearn.utils._validation import ArraysTransformer
from imblearn.utils import check_sampling_strategy
from sklearn.utils import (
    _get_column_indices,
    _safe_indexing,
    check_array,
    check_random_state,
)


class SMOTEFAST(SMOTE):
    def __init__(self, target_sr=16000, knn_count=5, ignore_topk=0, **kwargs):
        super(SMOTEFAST, self).__init__(**kwargs)
        self.target_sr=target_sr
        self.k_nn = knn_count + 2 #BUG: Unfortunately the + 2? seems to necessary. Fix this/Raise an Issue/Something?
        self.ignore_topk = ignore_topk
    
    def _make_samples(self, X, y_dtype, y_type, nn_num, n_samples, step_size=1.0):
        random_state = check_random_state(self.random_state)
        samples_indices = random_state.randint(low=0, high=nn_num.size, size=n_samples)

        # np.newaxis for backwards compatability with random_state
        steps = step_size * random_state.uniform(size=n_samples)[:, np.newaxis]
        rows = np.floor_divide(samples_indices, nn_num.shape[1])
        cols = np.mod(samples_indices, nn_num.shape[1])
        
        X_extended = X[rows].astype(X.dtype)
        y_extended = np.full(n_samples, fill_value=y_type, dtype=y_dtype)
        return X_extended, y_extended, nn_num[rows, cols], steps #self.Data[:self.cutoff][target_class_indices][self.neighbor_indices[idx]]
    
    def _get_data(self, X_class_files, Model=None):
        X_class = []
        for file_path in X_class_files.tolist():
            wav, fs = sf.read(file_path)
            raw = librosa.core.to_mono(wav.transpose()).transpose()[:10 * self.target_sr]#.reshape(1,-1)
            X_class.append(raw)
        
        Batch = np.array(X_class, dtype=np.float32)
        if Model is not None:
            #Forward data samples through model (thus performing dimensionality reduction) and obtain output embedding vectors
            batch = 64
            Model.eval() 
            with torch.no_grad():
                #Forward through model
                forward = []
                for i in range(np.ceil(Batch.shape[0] / batch).astype(np.int32)):
                    forward.append(torch.nn.functional.normalize(Model(torch.from_numpy(Batch[i*batch:(i+1)*batch,:]).cuda()), dim=1))
                Batch = torch.vstack(forward).cpu().numpy()
            
            Model.train()
            
        return Batch
    
    def _fit_resample(self, X, y, Model=None):
        self._validate_estimator()

        X_resampled = [X]#[X.copy()]
        y_resampled = [y]#[y.copy()]
        n_indices = [np.zeros_like(y)]
        steps = [np.zeros_like(X)]
        for num, (class_sample, n_samples) in enumerate(self.sampling_strategy_.items()):
            print("Processing oversampling for {} out of {} classes...".format(num+1,len(self.sampling_strategy_.keys())))
            if n_samples == 0: 
                continue
            target_class_indices = np.flatnonzero(y == class_sample)
            X_class_files = _safe_indexing(X, target_class_indices)
            X_class = self._get_data(X_class_files, Model)#[:,:3]
            og_samples = X_class.shape[0]
            if og_samples < self.k_nn: 
                X_class = np.repeat(X_class, [1]*(og_samples - 1) + [self.k_nn - og_samples], axis=0)
                X_class_files = np.repeat(X_class_files, [1]*(og_samples - 1) + [self.k_nn - og_samples], axis=0)
            
            self.nn_k_.fit(X_class)
            nns = self.nn_k_.kneighbors(X_class, return_distance=False)[:og_samples, 1:]
            if og_samples < self.k_nn:
                X_class = X_class[:og_samples,:] #I don't think this line is necessary....
                X_class_files = X_class_files[:og_samples]
                nns[:og_samples-1][nns[:og_samples-1] >= og_samples] = og_samples - 1
                nns[-1][nns[-1] >= og_samples-1] = np.random.randint(0,og_samples-1)
            
            X_new, y_new, neighbor_indices, interp_steps = self._make_samples(X_class_files, y.dtype, class_sample, nns, n_samples, 1.0)
            X_resampled.append(X_new)
            y_resampled.append(y_new)
            n_indices.append(neighbor_indices)
            steps.append(interp_steps[:,0])

        X_resampled = np.hstack(X_resampled)
        y_resampled = np.hstack(y_resampled)
        n_indices = np.hstack(n_indices)
        steps = np.hstack(steps)

        return X_resampled, y_resampled, n_indices, steps
    
    #My own custom sampler...
    def _custom_sampling_strategy(self, y):
        ids, counts = np.unique(y, return_counts=True)
        MaxV = np.sort(counts)[-self.ignore_topk-1].item()
        print("Max Sampling Value:",MaxV)
        for key in self.sampling_strategy_.keys():
            count = counts[ids==key].item()
            if count == 1 or count > MaxV:
                self.sampling_strategy_[key] = 0
            elif 10*count < MaxV:
                self.sampling_strategy_[key] = 10*count
            else:
                self.sampling_strategy_[key] = MaxV
    
    def fit_resample(self, X, y, Model=None):
        #Update this description sometime perhaps...
        """Resample the dataset.

        Parameters
        ----------
        X : {array-like, dataframe, sparse matrix} of shape \
                (n_samples, n_features)
            Matrix containing the data which have to be sampled.

        y : array-like of shape (n_samples,)
            Corresponding label for each sample in X.

        Returns
        -------
        X_resampled : {array-like, dataframe, sparse matrix} of shape \
                (n_samples_new, n_features)
            The array containing the resampled data.

        y_resampled : array-like of shape (n_samples_new,)
            The corresponding label of `X_resampled`.
        """
        cutoff = len(X)
        check_classification_targets(y)
        self.sampling_strategy_ = check_sampling_strategy(self.sampling_strategy, y, self._sampling_type)
        self._custom_sampling_strategy(y)
        output = self._fit_resample(X, y, Model)
        return (output[0], output[1], output[2], output[3], cutoff)

    def refresh(self, X, y, Model):
        #Data-Files needs to be updated...
        #Nearest-Neighbors needs to be updated...
        #Steps needs to be updated...
        return self.fit_resample(X, y, Model)

