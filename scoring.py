import numpy as np
import ot
import properscoring

class EnsembleComparator:
    def __init__(self, ensemble1, ensemble2):
        self.ensemble1 = ensemble1
        self.ensemble2 = ensemble2
        
    def ot_connectivity(self):
        ensemble1 = self.ensemble1.connectivities_ensemble()
        ensemble2 = self.ensemble2.connectivities_ensemble()
        return np.mean(self._ot_ensemble(ensemble1, ensemble2))
    
    def ot_variogram(self):
        ensemble1 = self.ensemble1.variograms_ensemble()
        ensemble2 = self.ensemble2.variograms_ensemble()
        return np.mean(self._ot_ensemble(ensemble1, ensemble2))
    
    def _ot_ensemble(self, ensemble1, ensemble2):
        return [self._ot_point_by_point(ensemble1[c], ensemble2[c])
             for c in ensemble1]
    
    def ot_proportions(self):
        proportions1 = self.ensemble1.proportions_ensemble()
        proportions2 = self.ensemble2.proportions_ensemble()
        ot_scores = [ot.emd2_1d(proportions1[c], proportions2[c])
                     for c in proportions1]
        return np.mean(ot_scores)
    
    @staticmethod
    def _ot_point_by_point(array1, array2):
        return [ot.emd2_1d(p1, p2) for p1, p2 in zip(array1.T, array2.T)]
    
class EnsembleToReferenceComparator:
    def __init__(self, ensemble, reference):
        self.ensemble = ensemble
        self.reference = reference
        
    def crps_connectivity(self):
        ensemble = self.ensemble.connectivities_ensemble()
        ref = self.reference.connectivities_ensemble()
        crps_scores = self._crps(ensemble, ref)
        return np.mean(crps_scores)
    
    def crps_variogram(self):
        ensemble = self.ensemble.variograms_ensemble()
        ref = self.reference.variograms_ensemble()
        crps_scores = self._crps(ensemble, ref)
        return np.mean(crps_scores)
        
    @staticmethod    
    def _crps(ensemble, ref):
        return [properscoring.crps_ensemble(ref[c][0,:ensemble[c].shape[-1]].T, ensemble[c].T)
             for c in ensemble]
    
    def crps_proportions(self):
        ensemble = self.ensemble.proportions_ensemble()
        ref = self.reference.proportions_ensemble()
        crps_scores = [properscoring.crps_ensemble(ref[c][0], ensemble[c])
             for c in ensemble]
        return np.mean(crps_scores)