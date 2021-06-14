import numpy as np
import ot
import properscoring

def to_dict_of_lists(list_of_dicts):
    return {key: np.array([d[key] for d in list_of_dicts])
            for key in list_of_dicts[0]}

def ot_categorical(ensemble1, ensemble2, function):
    dict1 = to_dict_of_lists(ensemble1.apply(function))
    dict2 = to_dict_of_lists(ensemble2.apply(function))
    return {key: ot_point_by_point(dict1[key], dict2[key])
            for key in dict1}
    
def quantiles_categorical(ensemble, quantiles, function):
    return {key: np.quantile(item, q=quantiles, axis=0)
            for key, item in to_dict_of_lists(ensemble.apply(function)).items()}
    
def avg_ot_by_category(ensemble1, ensemble2, function):
    dict_ot = ot_categorical(ensemble1, ensemble2, function)
    return {key: np.mean(item) for key, item in dict_ot.items()}

def avg_ot(ensemble1, ensemble2, function):
    dict_avg_ot = avg_ot_by_category(ensemble1, ensemble2, function)
    return np.mean([item for key, item in dict_avg_ot.items()])

def ot_point_by_point(array1, array2):
    return [ot.emd2_1d(p1, p2) for p1, p2 in zip(array1.T, array2.T)]
    