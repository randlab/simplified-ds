from simulator import DS, DSBC
from geone import deesseinterface
import copy
import numpy as np

class TsanfleuronMixin:
    def deesse_input(self, ti, ensemble_size, nthreads, seed):
        pyrGenParams = deesseinterface.PyramidGeneralParameters(
            npyramidLevel=2,
            kx=[2, 2], ky=[2, 2], kz=[0, 0]
            )

        pyrParams = deesseinterface.PyramidParameters(
            nlevel=2, 
            pyramidType='continuous'
        )
        
        new_ti = self.cut_ti(ti) # here was cut ti
        grid = self.sim_grid(ti)
        #grid = new_ti # remove this and uncomment upper line
        deesse_input = deesseinterface.DeesseInput(
            nx=grid.nx, ny=grid.ny, nz=grid.nz,
            sx=grid.sx, sy=grid.sy, sz=grid.sz,
            ox=grid.ox, oy=grid.oy, oz=grid.oz,
            nv=1, varname=grid.varname,
            nTI=1, TI=new_ti,
            dataImage=grid, # here was grid
            distanceType='continuous',
            relativeDistanceFlag=True, # set relative distance
            nneighboringNode=self.nneighboringNode,
            distanceThreshold=self.distanceThreshold+self.epsilon,
            maxScanFraction=self.maxScanFraction,
            #pyramidGeneralParameters=pyrGenParams, # set pyramid general parameters
            #pyramidParameters=pyrParams,           # set pyramid parameters for each variable
            npostProcessingPathMax=1,
            seed=seed,
            nrealization=ensemble_size)
        return deesse_input
    
    def small_ti(self, ti):
        small_ti = copy.deepcopy(ti)
        small_ti.resize(ix0=self.ox-2*self.nx, ix1=self.ox+3*self.nx,
            iy0=self.oy-2*self.ny, iy1=self.oy+3*self.ny)
        return small_ti
    
    def cut_ti(self, ti):
        rng = np.random.default_rng(seed=self.ox*self.oy)
        new_ti = copy.deepcopy(ti)
        new_ti.val[:,:,self.oy:self.oy+self.ny, self.ox:self.ox+self.nx] = np.nan
        mx = self.nx // 5
        my = self.ny // 5
        x = rng.integers(self.ox + mx, self.ox + self.nx - mx)
        y = rng.integers(self.oy + my, self.oy + self.ny - my)
        new_ti.val[:,:,y, :] = ti.val[:,:,y, :]
        new_ti.val[:,:,:,x] = ti.val[:,:, :, x]
        return new_ti
    
    def sim_grid(self, ti):
        grid = self.cut_ti(ti)
        grid.resize(ix0=self.ox-self.nx//2, ix1=self.ox+3*self.nx//2,
                    iy0=self.oy-self.ny//2, iy1=self.oy+3*self.ny//2)
        return grid
    
    def im_ref(self, ti):
        im = copy.deepcopy(ti)
        im.resize(ix0=self.ox, ix1=self.ox+self.nx,
                  iy0=self.oy, iy1=self.oy+self.ny)
        return im
    
    def im_ref_with_sim(self, ti):
        im = copy.deepcopy(ti)
        im.resize(ix0=self.ox-self.nx//2, ix1=self.ox+3*self.nx//2,
                    iy0=self.oy-self.ny//2, iy1=self.oy+3*self.ny//2)
        return im

    def run_deesse(self, deesse_input, nthreads):
        print("running specialized")
        deesse_output = deesseinterface.deesseRun(deesse_input, nthreads=nthreads)
        for output in deesse_output['sim']:
            output.resize(ix0=self.nx//2, ix1=3*self.nx//2, iy0=self.ny//2, iy1=3*self.ny//2)
        return deesse_output    
    
class TsanfleuronDS(DS, TsanfleuronMixin):
    def __init__(self,
                 root_dir,
                 nneighboringNode,
                 distanceThreshold,
                 maxScanFraction,
                 ox,
                 oy,
                 nx = 200,
                 ny = 200,
                 ):
        self.ox = ox
        self.oy = oy
        self.nx = nx
        self.ny = ny
        self.epsilon=0
        super().__init__(                 
                 root_dir,
                 nneighboringNode,
                 distanceThreshold,
                 maxScanFraction,
                )

class TsanfleuronDSBC(DSBC, TsanfleuronMixin):
    def __init__(self,
                 root_dir,
                 nneighboringNode,
                 distanceThreshold,
                 maxScanFraction,
                 ox,
                 oy,
                 nx = 200,
                 ny = 200,
                 ):
        self.ox = ox
        self.oy = oy
        self.nx = nx
        self.ny = ny
        self.epsilon = 1e-6
        super().__init__(                 
                 root_dir=root_dir,
                 nneighboringNode=nneighboringNode,
                 distanceThreshold=0,
                 maxScanFraction=maxScanFraction
                )