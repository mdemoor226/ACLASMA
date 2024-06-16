import numpy as np
from scipy import sparse
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
import warnings
warnings.filterwarnings("ignore")


class SMOTEFAST(SMOTE):
    def __init__(self, target_sr=16000, knn_count=5, ignore_topk=0, **kwargs):
        super(SMOTEFAST, self).__init__(**kwargs)
        self.target_sr=target_sr
        self.k_nn = knn_count + 2 #BUG: Unfortunately the + 2? seems to necessary. Fix this/Raise an Issue/Something?
        self.ignore_topk = ignore_topk
    
    def _get_data(self, X_class_files):
        X_class = []
        for file_path in X_class_files.tolist():
            wav, fs = sf.read(file_path)
            raw = librosa.core.to_mono(wav.transpose()).transpose()[:10 * self.target_sr]#.reshape(1,-1)
            X_class.append(raw)
        
        return np.array(X_class, dtype=np.float32)
    
    def _fit_resample(self, X, y):
        def make_samples(X, y_dtype, y_type, nn_num, n_samples, step_size=1.0):
            random_state = check_random_state(self.random_state)
            samples_indices = random_state.randint(low=0, high=nn_num.size, size=n_samples)

            # np.newaxis for backwards compatability with random_state
            steps = step_size * random_state.uniform(size=n_samples)[:, np.newaxis]
            rows = np.floor_divide(samples_indices, nn_num.shape[1])
            cols = np.mod(samples_indices, nn_num.shape[1])
            
            """
            #X_new = self._generate_samples(X, nn_data, nn_num, rows, cols, steps, y_type, y)
            diffs = nn_data[nn_num[rows, cols]] - X[rows]
            X_new = np.astype(X[rows] + steps * diffs, X.dtype)
            y_new = np.full(n_samples, fill_value=y_type, dtype=y_dtype)
            return X_new, y_new, nn_num[rows, cols], steps
            #"""
            #import code
            #code.interact(local=locals())
            X_extended = X[rows].astype(X.dtype)
            y_extended = np.full(n_samples, fill_value=y_type, dtype=y_dtype)
            return X_extended, y_extended, nn_num[rows, cols], steps #self.Data[:self.cutoff][target_class_indices][self.neighbor_indices[idx]]


        self._validate_estimator()

        X_resampled = [X]#[X.copy()]
        y_resampled = [y]#[y.copy()]
        n_indices = [np.zeros_like(y)]
        steps = [np.zeros_like(X)]
        #print(self.random_state)
        #input("Random State...")
        #import code
        #code.interact(local=locals())
        for num, (class_sample, n_samples) in enumerate(self.sampling_strategy_.items()):
            print("Processing oversampling for {} out of {} classes...".format(num+1,len(self.sampling_strategy_.keys())))
            if n_samples == 0: #and n_samples > THRESHOLD???!!!!????  
                continue
            target_class_indices = np.flatnonzero(y == class_sample)
            X_class_files = _safe_indexing(X, target_class_indices)
            X_class = self._get_data(X_class_files)#[:,:3]
            og_samples = X_class.shape[0]
            if og_samples < self.k_nn: 
                #Warning: This may cause the algorithm to vanilla oversample some samples with replacement (no smote). Fix this?
                #import code
                #code.interact(local=locals())
                #X_class = np.repeat(X_class, self.k_nn - X_class.shape[0], axis=0)
                #X_class_files = np.repeat(X_class_files, self.k_nn - X_class_files.shape[0], axis=0)
                X_class = np.repeat(X_class, [1]*(og_samples - 1) + [self.k_nn - og_samples], axis=0)
                X_class_files = np.repeat(X_class_files, [1]*(og_samples - 1) + [self.k_nn - og_samples], axis=0)
                #code.interact(local=locals())

            
            self.nn_k_.fit(X_class)
            nns = self.nn_k_.kneighbors(X_class, return_distance=False)[:og_samples, 1:]
            #print(nns)
            #input("nns")
            if og_samples < self.k_nn:
                X_class = X_class[:og_samples,:]
                X_class_files = X_class_files[:og_samples]
                nns[:og_samples-1][nns[:og_samples-1] >= og_samples] = og_samples - 1
                nns[-1][nns[-1] >= og_samples-1] = np.random.randint(0,og_samples-1)
                #print(nns)
                #input("Updated nns")
                #import code
                #code.interact(local=locals())
            #nns = self.nn_k_.kneighbors(X_class, return_distance=False)[:, 1:]
            #print("Neighbors!")
            #import code
            #code.interact(local=locals())
            #X_new, y_new, neighbor_indices = make_samples(X_class_files, y.dtype, class_sample, nns, n_samples, 1.0)
            #print(X_new.shape)
            #input()
            X_new, y_new, neighbor_indices, interp_steps = make_samples(X_class_files, y.dtype, class_sample, nns, n_samples, 1.0)
            #import code
            #code.interact(local=locals())
            X_resampled.append(X_new)
            y_resampled.append(y_new)
            n_indices.append(neighbor_indices)
            steps.append(interp_steps[:,0])
            #break
            #if num % 50 == 0:
            #    #Periodically compress data into a numpy array to save memory.
            #    X_resampled = [np.vstack(X_resampled)]
            #    y_resampled = [np.hstack(y_resampled)]
            #    n_indices = [np.hstack(n_indices)]
            #    #steps = [np.vstack(steps)]

        assert not sparse.issparse(X)
        #import code
        #code.interact(local=locals())
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
    
    def fit_resample(self, X, y):
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
        #import code
        #print("Sampling Strategies...")
        #code.interact(local=locals())
        self._custom_sampling_strategy(y)
        output = self._fit_resample(X, y)
        return (output[0], output[1], output[2], output[3], cutoff)

    def refresh(self, X):
        #Data-Files needs to be updated...
        #Nearest-Neighbors needs to be updated...
        #Steps needs to be updated...
        pass
    
#Borrowed from ProxyNCA++ repository
def assign_by_euclidian_at_k(X, T, k):
    """ 
    X : [nb_samples x nb_features], e.g. 100 x 64 (embeddings)
    k : for each sample, assign target labels of k nearest points
    """
    distances = sklearn.metrics.pairwise.pairwise_distances(X)

    # get nearest points
    indices   = np.argsort(distances, axis = 1)[:, 1 : k + 1] 
    return np.array([[T[i] for i in ii] for ii in indices])

