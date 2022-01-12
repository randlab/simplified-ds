import numpy as np
import scipy
import ot
import properscoring

def to_dict_of_lists(list_of_dicts):
    return {key: np.array([d[key] for d in list_of_dicts])
            for key in list_of_dicts[0]}

def score_categorical(ensemble1, ensemble2, function, score):
    dict1 = to_dict_of_lists(ensemble1.apply(function))
    dict2 = to_dict_of_lists(ensemble2.apply(function))
    return {key: score(dict1[key], dict2[key])
            for key in dict1}

def ot_categorical(ensemble1, ensemble2, function):
    return score_categorical(ensemble1, ensemble2, function, score=ot_point_by_point)
    
def quantiles_categorical(ensemble, quantiles, function):
    return {key: np.quantile(item, q=quantiles, axis=0)
            for key, item in to_dict_of_lists(ensemble.apply(function)).items()}

def avg_score_by_category(ensemble1, ensemble2, function, score):
    dict_score = score_categorical(ensemble1, ensemble2, function, score)
    return {key: np.mean(item) for key, item in dict_score.items()}

def avg_ot_by_category(ensemble1, ensemble2, function):
    return avg_score_by_category(ensemble1, ensemble2, function, score=ot_point_by_point)

def avg_score(ensemble1, ensemble2, function, score):
    dict_avg_score = avg_score_by_category(ensemble1, ensemble2, function, score)
    return np.mean([item for key, item in dict_avg_score.items()])

def avg_ot(ensemble1, ensemble2, function):
    return avg_score(ensemble1, ensemble2, function, score=ot_point_by_point)

def avg_wasserstein(ensemble1, ensemble2, function):
    return avg_score(ensemble1, ensemble2, function, score=wasserstein_point_by_point)

def ot_point_by_point(array1, array2):
    return [ot.emd2_1d(p1, p2, metric='minkowski') for p1, p2 in zip(array1.T, array2.T)]

def wasserstein_point_by_point(array1, array2):
    return [scipy.stats.wasserstein_distance(p1, p2) for p1, p2 in zip(array1.T, array2.T)]


def crps_pointwise(ensemble, true, function):
    return properscoring.crps_ensemble(function(true),
                                       np.stack(ensemble.apply(function), axis=-1))