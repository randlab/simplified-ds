import copy
import os
import pickle

import numpy as np

class Ensemble:
    def __init__(self, members_list):
        if len(members_list) == 0:
            raise ValueError("Cannot create ensemble from empty list")
        self.members = members_list
            
    def __len__(self):
        return len(self.members)
    
    def __getitem__(self, index):
        return self.members[index]
        
    def __repr__(self):
        return f"Ensemble({self.members.__repr__()})"
        
    def __str__(self):
        return f"Ensemble of size {len(self)}"
    
    def to_directory(self, dir_path, overwrite=False, dtype='int8'):
        os.makedirs(dir_path, exist_ok=overwrite)
        for i, member in enumerate(self):
            with open(self._member_path(dir_path, i), 'wb') as fh:
                pickle.dump(np.array(member, dtype=dtype), fh)
                
    def apply(self, function):
        return [function(m) for m in self]
                
    
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
    def from_deesse_output(cls, deesse_output, dtype=None):
        return cls([np.array(sim.val[0,0,:,:], dtype=dtype) for sim in deesse_output['sim']])

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
    def from_geone_image(cls, image):
        return cls([image.val[0,0,:,:]])
    
    
class EmptyEnsembleError(Exception):
    pass