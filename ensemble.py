import copy
import os
import pickle

import numpy as np

from mpstool.stats import histogram, variogram, get_categories
from mpstool.connectivity import get_function

class Ensemble:
    def __init__(self, members_list, compute_metrics=True):
        if len(members_list) == 0:
            raise ValueError("The members_list is empty")
        self.members = members_list
        self.size = len(members_list)
        # watch out!
        self.categories = get_categories(members_list[0])
        
        if compute_metrics:
            self.compute_metrics()
        
    def __repr__(self):
        return f"Ensemble({self.members.__repr__()})"
        
    def __str__(self):
        return f"Ensemble of size {self.size}"
    
    def to_directory(self, dir_path, overwrite=False):
        os.makedirs(dir_path, exist_ok=overwrite)
        for i, member in enumerate(self.members):
            with open(self._member_path(dir_path, i), 'wb') as fh:
                pickle.dump(np.array(member, dtype='int8'), fh)
                
    def compute_metrics(self):
        self.proportions = [histogram(m) for m in self.members]
        self.variograms = [variogram(m, axis=1) for m in self.members]
        self.connectivities = [get_function(m, axis=1) for m in self.members]
        
    def proportions_ensemble(self):
        return self.to_dict_of_lists(self.proportions)
    
    def variograms_ensemble(self):
        return self.to_dict_of_lists(self.variograms)
    
    def connectivities_ensemble(self):
        return self.to_dict_of_lists(self.connectivities)
    
    def proportion_quantiles(self, quantiles=[0.05, 0.25, 0.5, 0.75, 0.95]):
        ensemble = self.proportions_ensemble()
        return self.ensemble_to_quantiles(ensemble, quantiles)
    
    def variogram_quantiles(self, quantiles=[0.05, 0.25, 0.5, 0.75, 0.95]):
        ensemble = self.variograms_ensemble()
        return self.ensemble_to_quantiles(ensemble, quantiles)
    
    def connectivity_quantiles(self, quantiles=[0.05, 0.25, 0.5, 0.75, 0.95]):
        ensemble = self.connectivities_ensemble()
        return self.ensemble_to_quantiles(ensemble, quantiles)
               
    def ensemble_to_quantiles(self, ensemble, quantiles):
        return {c: np.quantile(ensemble[c], q=quantiles, axis=0) for c in self.categories}
    
    def to_dict_of_lists(self, list_of_dicts):
        result = {c: np.array([d[c] for d in list_of_dicts]) for c in self.categories}
        return result
    
    @classmethod            
    def from_directory(cls, dir_path, overwrite=False):
        members_list = []
        for filename in os.listdir(dir_path):
            if 'member' in filename:
                with open(os.path.join(dir_path, filename), 'rb') as fh:
                    member = pickle.load(fh)
                    members_list.append(member)
        if len(members_list) == 0:
            raise EmptyEnsembleError(f"The specified directory {dir_path} does not contain any ensemble members")
        return cls(members_list)
            
    @classmethod
    def from_ti(cls, ti, x_size, y_size, size, seed):
        rng = np.random.default_rng(seed=seed)
        x_corners = rng.integers(low=x_size, high=ti.nx, size=size)
        y_corners = rng.integers(low=y_size, high=ti.ny, size=size)
        ref_ensemble = [cls._resize_ti(ti,
                                       ix0=x-x_size,
                                       ix1=x,
                                       iy0=y-y_size,
                                       iy1=y).val[0,0,:]
                    for x,y in zip(x_corners, y_corners)]
        return cls(ref_ensemble)
    
    @classmethod
    def from_deesse_output(cls, deesse_output):
        return cls([sim.val[0,0,:,:] for sim in deesse_output['sim']])

    @staticmethod
    def _resize_ti(ti, ix0, ix1, iy0, iy1):
        new_image = copy.copy(ti)
        new_image.resize(ix0=ix0, ix1=ix1, iy0=iy0, iy1=iy1)
        return new_image
    
    @staticmethod
    def _member_path(dir_path, i):
        return os.path.join(dir_path, f'member-{i}.pickle')
    
class SingleRealization(Ensemble):
    
    @classmethod
    def from_image(cls, image):
        return cls([image.val[0,0,:,:]])
    
    
class EmptyEnsembleError(Exception):
    pass