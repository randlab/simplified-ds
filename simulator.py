import time
import os

from geone import deesseinterface

from ensemble import Ensemble, EmptyEnsembleError

class DS:
    def __init__(self,
                 root_dir,
                 nneighboringNode,
                 distanceThreshold,
                 maxScanFraction,
                ):
   
        self.nneighboringNode = nneighboringNode
        self.distanceThreshold = distanceThreshold
        self.maxScanFraction = maxScanFraction
        
        self.results_dir = self.results_dir(root_dir)
        self.time_file = os.path.join(self.results_dir, 'time-seconds.txt')
        self.epsilon = 1e-5
        
    def create_ensemble(self, ti, ensemble_size, nthreads, seed=444, overwrite=False):
    
        pyrGenParams = deesseinterface.PyramidGeneralParameters(
                    npyramidLevel=2,
                    kx=[2, 2], ky=[2, 2], kz=[0, 0]
                    )

        pyrParams = deesseinterface.PyramidParameters(
            nlevel=2, 
            pyramidType='categorical_auto'
        )

        deesse_input = deesseinterface.DeesseInput(
            nx=200, ny=200, nz=1,
            nv=1, varname='code',
            nTI=1, TI=ti,
            distanceType='categorical',
            nneighboringNode=self.nneighboringNode,
            distanceThreshold=self.distanceThreshold+self.epsilon,
            maxScanFraction=self.maxScanFraction,
            pyramidGeneralParameters=pyrGenParams, # set pyramid general parameters
            pyramidParameters=pyrParams,           # set pyramid parameters for each variable
            npostProcessingPathMax=1,
            seed=seed,
            nrealization=ensemble_size)
        
        # run deeesse and measure time
        tic = time.perf_counter()
        deesse_output = deesseinterface.deesseRun(deesse_input, nthreads=nthreads)
        toc = time.perf_counter()
        elapsed_time = toc-tic
        
        # build ensemble
        deesse_ensemble = Ensemble.from_deesse_output(deesse_output)
        deesse_ensemble.to_directory(self.results_dir, overwrite=overwrite)
        
        # then write time (to handle overwrite correctly)
        self.write_timing(elapsed_time)
        
        return deesse_ensemble, elapsed_time
    
    def get_ensemble(self, ti, ensemble_size, nthreads, seed=444, overwrite=False):
        try:
            ensemble, timing = self.read_ensemble()
            if ensemble.size < ensemble_size:
                raise EmptyEnsembleError("Too small ensemble")
        except (EmptyEnsembleError, FileNotFoundError):
            ensemble, timing = self.create_ensemble(ti, ensemble_size, nthreads, seed=seed, overwrite=True)
        
        return ensemble, timing
        
    
    def read_ensemble(self):
        ensemble = Ensemble.from_directory(self.results_dir)
        timing = self.read_timing()
        return ensemble, timing
    
    def write_timing(self, timing):
        with open(self.time_file, 'w') as fh:
            fh.write(str(timing))
            
    def read_timing(self):
        with open(self.time_file, 'r') as fh:
            return float(fh.read())
            
    def results_dir(self, root_dir):
        return os.path.join(root_dir, f'ds-{self.nneighboringNode}-{self.distanceThreshold}-{self.maxScanFraction}')

    
class DSBC(DS):
    def __init__(self,
                 root_dir,
                 nneighboringNode,
                 maxScanFraction,
                 distanceThreshold=0,
                ):
        # threshold = 0 irrespectively of user's input
        super().__init__(root_dir=root_dir,
                        nneighboringNode=nneighboringNode,
                        distanceThreshold=0,
                        maxScanFraction=maxScanFraction,
                        )
        
    def results_dir(self, root_dir):
        return os.path.join(root_dir, f'dsbc-{self.nneighboringNode}-{self.maxScanFraction}') 
       