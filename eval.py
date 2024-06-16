import pandas as pd
import numpy as np
import os
from sklearn.metrics import roc_auc_score, pairwise
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from scipy.stats import hmean
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")

def evaluate(embeddings, Preds, metadata, logger, evalcfg, n_subclusters=16, Test=False):#labels, Sources, IDs, Preds, le, n_subclusters=16)
    if evalcfg['knn']:
        #pred_eval, pred_unknown = pred_knn(Preds, embeddings, metadata, evalcfg)
        pred_eval, pred_unknown, pred_test = pred_knn(Preds, embeddings, metadata, evalcfg)
        Preds['Test'] = pred_test
    else:
        if Test:
            pred_eval, pred_unknown, pred_test = test_cos(Preds, embeddings, metadata, n_subclusters)
            #import code
            #code.interact(local=locals())
            Preds['Test'] = pred_test
        else:
            pred_eval, pred_unknown = pred_cos(Preds, embeddings, metadata, n_subclusters)

    Preds['Eval'] = pred_eval
    Preds['Unknown'] = pred_unknown

    logger.info('####################')
    logger.info('####################')
    logger.info('####################')
    if Test:
        final_results_dev = None
        logger.info('final results for evaluation set')
        final_results_eval = evaluate_eval(Preds, metadata, logger)
        logger.info(np.round(final_results_eval*100, 2))
    else:
        final_results_eval = None
        logger.info('final results for development set')
        final_results_dev = evaluate_dev(Preds, metadata, logger) 
        logger.info(np.round(final_results_dev*100, 2))
    logger.info('####################')
    logger.info('>>>> finished! <<<<<')
    logger.info('####################')
    return final_results_dev, final_results_eval

def pred_knn(Preds, embeddings, metadata, evalcfg):
    train_labels = metadata['Labels']['Train']
    eval_labels = metadata['Labels']['Eval']
    unknown_labels = metadata['Labels']['Unknown']
    test_labels = metadata['Labels']['Test']
    le = metadata['label_encoder']

    source_train = metadata['Sources']['Train']
    pred_eval = Preds['Eval']
    pred_unknown = Preds['Unknown']
    pred_test = Preds['Test']

    # extract embeddings
    x_train_ln = embeddings['Train']
    x_eval_ln = embeddings['Eval']
    x_unknown_ln = embeddings['Unknown']
    x_test_ln = embeddings['Test']
    
    for j, lab in tqdm(enumerate(np.unique(train_labels))):
        if np.sum(eval_labels==lab) == 0:
            continue
        
        dist_eval = pairwise_knn(x_eval_ln[eval_labels == lab], x_train_ln[(train_labels == lab)], k=evalcfg['k_value'])[0]
        score_eval1 = np.mean(dist_eval[:,:evalcfg['k_value']], axis=1)
        #score_eval2 = np.median(dist_eval[:,:evalcfg['k_value']], axis=1)
        #import code
        #code.interact(local=locals())
        score_eval = score_eval1
        dist_unknown = pairwise_knn(x_unknown_ln[unknown_labels == lab], x_train_ln[(train_labels == lab)], k=evalcfg['k_value'])[0]
        score_unknown1 = np.mean(dist_unknown[:,:evalcfg['k_value']], axis=1)
        #score_unknown2 = np.median(dist_unknown[:,:evalcfg['k_value']], axis=1)
        #import code
        #code.interact(local=locals())
        score_unknown = score_unknown1
        dist_test = pairwise_knn(x_test_ln[test_labels == lab], x_train_ln[(train_labels == lab)], k=evalcfg['k_value'])[0]
        score_test = np.mean(dist_test[:,:evalcfg['k_value']], axis=1)
        pred_eval[eval_labels == lab, j] = score_eval
        pred_unknown[unknown_labels == lab, j] = score_unknown
        pred_test[test_labels == lab, j] = score_test
    
    return pred_eval, pred_unknown, pred_test

def pred_knn_subcluster(Preds, embeddings, metadata, evalcfg, n_subclusters=16):
    train_labels = metadata['Labels']['Train']
    eval_labels = metadata['Labels']['Eval']
    unknown_labels = metadata['Labels']['Unknown']
    #test_labels = metadata['Labels']['Test']
    le = metadata['label_encoder']

    source_train = metadata['Sources']['Train']
    pred_eval = Preds['Eval']
    pred_unknown = Preds['Unknown']
    #pred_test = Preds['Test']

    # extract embeddings
    x_train_ln = embeddings['Train']
    x_eval_ln = embeddings['Eval']
    x_unknown_ln = embeddings['Unknown']
    #x_test_ln = embeddings['Test']
    
    for j, lab in tqdm(enumerate(np.unique(train_labels))):
        if np.sum(eval_labels==lab) == 0:
            continue
        
        # prepare mean values for domains
        kmeans = KMeans(n_clusters=n_subclusters, random_state=0).fit(x_train_ln[(train_labels == lab)])
        means_source_ln = kmeans.cluster_centers_
        means_target_ln = x_train_ln[~source_train * (train_labels == lab)]
        means_total = np.concatenate((means_source_ln, means_target_ln), axis=-1)

        dist_eval = pairwise_knn(x_eval_ln[eval_labels == lab], means_total, k=evalcfg['k_value'])[0]
        score_eval = np.mean(dist_eval[:,:evalcfg['k_value']], axis=1)
        dist_unknown = pairwise_knn(x_unknown_ln[unknown_labels == lab], means_total, k=evalcfg['k_value'])[0]
        score_unknown = np.mean(dist_unknown[:,:evalcfg['k_value']], axis=1)
        #dist_test = pairwise_knn(x_test_ln[test_labels == lab], x_train_ln[(train_labels == lab)], k=evalcfg['k_value'])[0]
        #score_test = np.mean(dist_test[:,:evalcfg['k_value']], axis=1)
        pred_eval[eval_labels == lab, j] = score_eval
        pred_unknown[unknown_labels == lab, j] = score_unknown
        #pred_test[test_labels == lab, j] = score_test
    
    return pred_eval, pred_unknown#, pred_test

def pred_cos(Preds, embeddings, metadata, n_subclusters=16):
    train_labels = metadata['Labels']['Train']
    eval_labels = metadata['Labels']['Eval']
    unknown_labels = metadata['Labels']['Unknown']
    #test_labels = metadata['Labels']['Test']
    le = metadata['label_encoder']

    source_train = metadata['Sources']['Train']
    pred_eval = Preds['Eval']
    pred_unknown = Preds['Unknown']
    #pred_test = Preds['Test']

    # extract embeddings
    x_train_ln = embeddings['Train']
    x_eval_ln = embeddings['Eval']
    x_unknown_ln = embeddings['Unknown']
    #x_test_ln = embeddings['Test']
    
    for j, lab in tqdm(enumerate(np.unique(train_labels))):
        # prepare mean values for domains
        kmeans = KMeans(n_clusters=n_subclusters, random_state=0).fit(x_train_ln[(train_labels == lab)])
        means_source_ln = kmeans.cluster_centers_
        means_target_ln = x_train_ln[~source_train * (train_labels == lab)]
        
        # compute cosine distances
        eval_cos = np.max(np.dot(x_eval_ln[eval_labels == lab], means_target_ln.transpose()),axis=-1)
        eval_cos = 1.0 - np.maximum(eval_cos, np.max(np.dot(x_eval_ln[eval_labels == lab], means_source_ln.transpose()), axis=-1))
        unknown_cos = np.max(np.dot(x_unknown_ln[unknown_labels == lab], means_target_ln.transpose()),axis=-1)
        unknown_cos = 1.0 - np.maximum(unknown_cos, np.max(np.dot(x_unknown_ln[unknown_labels == lab], means_source_ln.transpose()), axis=-1))
        #test_cos = np.max(np.dot(x_test_ln[test_labels==lab], means_target_ln.transpose()), axis=-1)
        #test_cos = 1.0 - np.maximum(test_cos, np.max(np.dot(x_test_ln[test_labels==lab], means_source_ln.transpose()), axis=-1))
        if np.sum(eval_labels==lab)>0:
            pred_eval[eval_labels == lab, j] = eval_cos
            pred_unknown[unknown_labels == lab, j] = unknown_cos
        #if np.sum(test_labels==lab)>0:
        #    pred_test[test_labels == lab, j] = test_cos
    return pred_eval, pred_unknown#, pred_test

def test_cos(Preds, embeddings, metadata, n_subclusters=16):
    train_labels = metadata['Labels']['Train']
    eval_labels = metadata['Labels']['Eval']
    unknown_labels = metadata['Labels']['Unknown']
    test_labels = metadata['Labels']['Test']
    le = metadata['label_encoder']

    source_train = metadata['Sources']['Train']
    pred_eval = Preds['Eval']
    pred_unknown = Preds['Unknown']
    pred_test = Preds['Test']

    # extract embeddings
    x_train_ln = embeddings['Train']
    x_eval_ln = embeddings['Eval']
    x_unknown_ln = embeddings['Unknown']
    x_test_ln = embeddings['Test']
    
    for j, lab in tqdm(enumerate(np.unique(train_labels))):
        # prepare mean values for domains
        kmeans = KMeans(n_clusters=n_subclusters, random_state=0).fit(x_train_ln[(train_labels == lab)])
        means_source_ln = kmeans.cluster_centers_
        means_target_ln = x_train_ln[~source_train * (train_labels == lab)]
        
        # compute cosine distances
        eval_cos = np.max(np.dot(x_eval_ln[eval_labels == lab], means_target_ln.transpose()),axis=-1)
        eval_cos = 1.0 - np.maximum(eval_cos, np.max(np.dot(x_eval_ln[eval_labels == lab], means_source_ln.transpose()), axis=-1))
        unknown_cos = np.max(np.dot(x_unknown_ln[unknown_labels == lab], means_target_ln.transpose()),axis=-1)
        unknown_cos = 1.0 - np.maximum(unknown_cos, np.max(np.dot(x_unknown_ln[unknown_labels == lab], means_source_ln.transpose()), axis=-1))
        test_cos = np.max(np.dot(x_test_ln[test_labels==lab], means_target_ln.transpose()), axis=-1)
        test_cos = 1.0 - np.maximum(test_cos, np.max(np.dot(x_test_ln[test_labels==lab], means_source_ln.transpose()), axis=-1))
        if np.sum(eval_labels==lab)>0:
            pred_eval[eval_labels == lab, j] = eval_cos
            pred_unknown[unknown_labels == lab, j] = unknown_cos
        if np.sum(test_labels==lab)>0:
            pred_test[test_labels == lab, j] = test_cos
    return pred_eval, pred_unknown, pred_test

#Borrowed from ProxyNCA++ repository
def pairwise_knn(X, Y, k):
    """ 
    X : [nb_samples x nb_features], e.g. 100 x 64 (embeddings)
    k : for each sample, assign target labels of k nearest points
    """
    distances = pairwise.euclidean_distances(X,Y)

    # get nearest points
    indices = np.argsort(distances, axis = 1)[:,:k]# 1 : k + 1] 
    return distances, indices

#eval_labels, unknown_labels, eval_ids, pred_eval, pred_unknown, le, source_eval, source_unknown, categories
def evaluate_dev(Preds, metadata, logger):
    # print results for development set
    print('#######################################################################################################')
    print('DEVELOPMENT SET')
    print('#######################################################################################################')
    train_labels = metadata['Labels']['Train']
    eval_labels = metadata['Labels']['Eval']
    unknown_labels = metadata['Labels']['Unknown']
    test_labels = metadata['Labels']['Test']
    eval_ids = metadata['IDs']['Eval']
    pred_eval = Preds['Eval']
    pred_unknown = Preds['Unknown']
    source_eval = metadata['Sources']['Eval']
    source_unknown = metadata['Sources']['Unknown']
    categories = metadata['categories']['Eval']
    le = metadata['label_encoder']
    
    aucs = []
    p_aucs = []
    aucs_source = []
    p_aucs_source = []
    aucs_target = []
    p_aucs_target = []

    for cat in np.unique(eval_ids):
        y_pred = np.concatenate([pred_eval[eval_labels == le.transform([cat]), le.transform([cat])], pred_unknown[unknown_labels == le.transform([cat]), le.transform([cat])]], axis=0)
        y_true = np.concatenate([np.zeros(np.sum(eval_labels == le.transform([cat]))), np.ones(np.sum(unknown_labels == le.transform([cat])))], axis=0)
        auc = roc_auc_score(y_true, y_pred)
        aucs.append(auc)
        p_auc = roc_auc_score(y_true, y_pred, max_fpr=0.1)
        p_aucs.append(p_auc)
        print('AUC for category ' + str(cat) + ': ' + str(auc * 100))
        print('pAUC for category ' + str(cat) + ': ' + str(p_auc * 100))

        source_all = np.concatenate([source_eval[eval_labels == le.transform([cat])],
                                     source_unknown[unknown_labels == le.transform([cat])]], axis=0)
        auc = roc_auc_score(y_true[source_all], y_pred[source_all])
        p_auc = roc_auc_score(y_true[source_all], y_pred[source_all], max_fpr=0.1)
        aucs_source.append(auc)
        p_aucs_source.append(p_auc)
        print('AUC for source domain of category ' + str(cat) + ': ' + str(auc * 100))
        print('pAUC for source domain of category ' + str(cat) + ': ' + str(p_auc * 100))
        auc = roc_auc_score(y_true[~source_all], y_pred[~source_all])
        p_auc = roc_auc_score(y_true[~source_all], y_pred[~source_all], max_fpr=0.1)
        aucs_target.append(auc)
        p_aucs_target.append(p_auc)
        print('AUC for target domain of category ' + str(cat) + ': ' + str(auc * 100))
        print('pAUC for target domain of category ' + str(cat) + ': ' + str(p_auc * 100))
    print('####################')
    aucs = np.array(aucs)
    p_aucs = np.array(p_aucs)
    for cat in categories:
        mean_auc = hmean(aucs[np.array([eval_id.split('_')[0] for eval_id in np.unique(eval_ids)]) == cat])
        print('mean AUC for category ' + str(cat) + ': ' + str(mean_auc * 100))
        mean_p_auc = hmean(p_aucs[np.array([eval_id.split('_')[0] for eval_id in np.unique(eval_ids)]) == cat])
        print('mean pAUC for category ' + str(cat) + ': ' + str(mean_p_auc * 100))
    #print('####################')
    logger.info('####################')
    for cat in categories:
        mean_auc = hmean(aucs[np.array([eval_id.split('_')[0] for eval_id in np.unique(eval_ids)]) == cat])
        mean_p_auc = hmean(p_aucs[np.array([eval_id.split('_')[0] for eval_id in np.unique(eval_ids)]) == cat])
        #print('mean of AUC and pAUC for category ' + str(cat) + ': ' + str((mean_p_auc + mean_auc) * 50))
        logger.info('mean of AUC and pAUC for category ' + str(cat) + ': ' + str((mean_p_auc + mean_auc) * 50))
    #print('####################')
    logger.info('####################')
    mean_auc_source = hmean(aucs_source)
    #print('mean AUC for source domain: ' + str(mean_auc_source * 100))
    logger.info('mean AUC for source domain: ' + str(mean_auc_source * 100))
    mean_p_auc_source = hmean(p_aucs_source)
    #print('mean pAUC for source domain: ' + str(mean_p_auc_source * 100))
    logger.info('mean pAUC for source domain: ' + str(mean_p_auc_source * 100))
    mean_auc_target = hmean(aucs_target)
    #print('mean AUC for target domain: ' + str(mean_auc_target * 100))
    logger.info('mean AUC for target domain: ' + str(mean_auc_target * 100))
    mean_p_auc_target = hmean(p_aucs_target)
    #print('mean pAUC for target domain: ' + str(mean_p_auc_target * 100))
    logger.info('mean pAUC for target domain: ' + str(mean_p_auc_target * 100))
    mean_auc = hmean(aucs)
    #print('mean AUC: ' + str(mean_auc * 100))
    logger.info('mean AUC: ' + str(mean_auc * 100))
    mean_p_auc = hmean(p_aucs)
    #print('mean pAUC: ' + str(mean_p_auc * 100))
    logger.info('mean pAUC: ' + str(mean_p_auc * 100))
    final_results_dev = np.array([mean_auc_source, mean_p_auc_source, mean_auc_target, mean_p_auc_target, mean_auc, mean_p_auc])
    return final_results_dev

#Preds, le, Labels, categories, IDs
#test_labels, test_ids, pred_test, le, categories
def evaluate_eval(Preds, metadata, logger):
    # print results for eval set
    print('#######################################################################################################')
    print('EVALUATION SET')
    print('#######################################################################################################')
    train_labels = metadata['Labels']['Train']
    eval_labels = metadata['Labels']['Eval']
    unknown_labels = metadata['Labels']['Unknown']
    test_labels = metadata['Labels']['Test']
    test_files = metadata['TsFiles']
    #eval_ids = metadata['IDs']['Eval']
    test_ids = metadata['IDs']['Test']
    pred_train = Preds['Train']
    pred_test = Preds['Test']
    #import code
    #code.interact(local=locals())
    categories = metadata['categories']['Test']
    le = metadata['label_encoder']
    data_year = metadata['data_year']
    year = metadata['data_year'][2:6]

    aucs = []
    p_aucs = []
    aucs_source = []
    p_aucs_source = []
    aucs_target = []
    p_aucs_target = []
    for j, cat in enumerate(np.unique(test_ids)):
        order = np.array([int(wav[-8:-4]) for wav in test_files[test_labels == le.transform([cat])].tolist()])
        y_pred = pred_test[test_labels == le.transform([cat]), le.transform([cat])][np.argsort(order)]
        y_true = np.array(pd.read_csv(data_year + 
            './dcase' + year + '_evaluator/ground_truth_data/ground_truth_' + cat.split('_')[0] + '_section_' + cat.split('_')[1] + '_test.csv', header=None).iloc[:, 1] == 1)
        auc = roc_auc_score(y_true, y_pred)
        aucs.append(auc)
        #import code
        #code.interact(local=locals())
        p_auc = roc_auc_score(y_true, y_pred, max_fpr=0.1)
        p_aucs.append(p_auc)
        print('AUC for category ' + str(cat) + ': ' + str(auc * 100))
        print('pAUC for category ' + str(cat) + ': ' + str(p_auc * 100))
        source_all = np.array(pd.read_csv(data_year + 
            './dcase' + year + '_evaluator/ground_truth_domain/ground_truth_' + cat.split('_')[0] + '_section_' + cat.split('_')[1] + '_test.csv', header=None).iloc[:, 1] == 0)
        auc = roc_auc_score(y_true[source_all], y_pred[source_all])
        p_auc = roc_auc_score(y_true[source_all], y_pred[source_all], max_fpr=0.1)
        aucs_source.append(auc)
        p_aucs_source.append(p_auc)
        print('AUC for source domain of category ' + str(cat) + ': ' + str(auc * 100))
        print('pAUC for source domain of category ' + str(cat) + ': ' + str(p_auc * 100))
        auc = roc_auc_score(y_true[~source_all], y_pred[~source_all])
        p_auc = roc_auc_score(y_true[~source_all], y_pred[~source_all], max_fpr=0.1)
        aucs_target.append(auc)
        p_aucs_target.append(p_auc)
        print('AUC for target domain of category ' + str(cat) + ': ' + str(auc * 100))
        print('pAUC for target domain of category ' + str(cat) + ': ' + str(p_auc * 100))
    print('####################')
    aucs = np.array(aucs)
    p_aucs = np.array(p_aucs)
    for cat in categories:
        mean_auc = hmean(aucs[np.array([test_id.split('_')[0] for test_id in np.unique(test_ids)]) == cat])
        print('mean AUC for category ' + str(cat) + ': ' + str(mean_auc * 100))
        mean_p_auc = hmean(p_aucs[np.array([test_id.split('_')[0] for test_id in np.unique(test_ids)]) == cat])
        print('mean pAUC for category ' + str(cat) + ': ' + str(mean_p_auc * 100))
    #print('####################')
    logger.info('####################')
    for cat in categories:
        mean_auc = hmean(aucs[np.array([test_id.split('_')[0] for test_id in np.unique(test_ids)]) == cat])
        mean_p_auc = hmean(p_aucs[np.array([test_id.split('_')[0] for test_id in np.unique(test_ids)]) == cat])
        #print('mean of AUC and pAUC for category ' + str(cat) + ': ' + str((mean_p_auc + mean_auc) * 50))
        logger.info('mean of AUC and pAUC for category ' + str(cat) + ': ' + str((mean_p_auc + mean_auc) * 50))
    #print('####################')
    logger.info('####################')
    mean_auc_source = hmean(aucs_source)
    #print('mean AUC for source domain: ' + str(mean_auc_source * 100))
    logger.info('mean AUC for source domain: ' + str(mean_auc_source * 100))
    mean_p_auc_source = hmean(p_aucs_source)
    #print('mean pAUC for source domain: ' + str(mean_p_auc_source * 100))
    logger.info('mean pAUC for source domain: ' + str(mean_p_auc_source * 100))
    mean_auc_target = hmean(aucs_target)
    #print('mean AUC for target domain: ' + str(mean_auc_target * 100))
    logger.info('mean AUC for target domain: ' + str(mean_auc_target * 100))
    mean_p_auc_target = hmean(p_aucs_target)
    #print('mean pAUC for target domain: ' + str(mean_p_auc_target * 100))
    logger.info('mean pAUC for target domain: ' + str(mean_p_auc_target * 100))
    mean_auc = hmean(aucs)
    #print('mean AUC: ' + str(mean_auc * 100))
    logger.info('mean AUC: ' + str(mean_auc * 100))
    mean_p_auc = hmean(p_aucs)
    #print('mean pAUC: ' + str(mean_p_auc * 100))
    logger.info('mean pAUC: ' + str(mean_p_auc * 100))
    final_results_eval = np.array([mean_auc_source, mean_p_auc_source, mean_auc_target, mean_p_auc_target, mean_auc, mean_p_auc])
    
    # create challenge submission files
    """
    print('creating submission files')
    sub_path = './submission-files/'
    if not os.path.exists(sub_path):
        os.makedirs(sub_path)
    for j, cat in enumerate(np.unique(test_ids)):
        # anomaly scores
        file_idx = test_labels == le.transform([cat])
        results_an = pd.DataFrame()
        results_an['output1'], results_an['output2'] = [[f.split('/')[-1] for f in test_files[file_idx]],
                                                        [str(s) for s in pred_test[file_idx, le.transform([cat])]]]
        results_an.to_csv(sub_path + '/anomaly_score_' + cat.split('_')[0] + '_section_' + cat.split('_')[-1] + '_test.csv',
                          encoding='utf-8', index=False, header=False)

        # decision results
        train_scores = pred_train[train_labels == le.transform([cat]), le.transform([cat])]
        threshold = np.percentile(train_scores, q=90)
        import code
        code.interact(local=locals())
        decisions = pred_test[file_idx, le.transform([cat])] > threshold
        results_dec = pd.DataFrame()
        results_dec['output1'], results_dec['output2'] = [[f.split('/')[-1] for f in test_files[file_idx]],
                                                          [str(int(s)) for s in decisions]]
        results_dec.to_csv(sub_path + '/decision_result_' + cat.split('_')[0] + '_section_' + cat.split('_')[-1] + '_test.csv',
                           encoding='utf-8', index=False, header=False)
    #"""
    return final_results_eval


if __name__ == '__main__':
    #Updated this file below in the future to run evaluations separately from training (if desired)...
    pass

