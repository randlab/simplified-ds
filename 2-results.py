#!/usr/bin/env python
# coding: utf-8

# # Apply cross validation

# In[1]:


import os

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.dummy import DummyClassifier

from geone.img import readImageGslib, readPointSetGslib
from geone.deesseinterface import DeesseClassifier
from geone.imgplot import drawImage2D
from mpstool.cv_metrics import brier_score, zero_one_score, balanced_linear_score, SkillScore


# In[2]:


DATA_DIR = 'data_roussillon/'
SAMPLES_DIR = DATA_DIR
OUTPUT_DIR = 'output/'
COLOR_SCHEME_BINARY = [ 
        [x/255 for x in [166,206,227]],
        [x/255 for x in [31,120,180]],
      ]


# In[3]:


# Stratified 5-fold cross-validation with randomly shuffled data
cv = StratifiedKFold(n_splits=5,
                     shuffle=True,
                     random_state=20191201,
                    )

scoring = {
    'brier':brier_score,
    'skill_brier':SkillScore(DummyClassifier(strategy='prior'), 0, brier_score),
}


# ### Training image selection

# ## Roussillon

# In[4]:


ti_true = readImageGslib(DATA_DIR+'trueTI.gslib')
mask = readImageGslib(DATA_DIR+'mask.gslib')
trend = readImageGslib(DATA_DIR+'trend.gslib')
im_angle = readImageGslib(DATA_DIR+'orientation.gslib')


# In[5]:


nx, ny, nz = mask.nx, mask.ny, mask.nz      # number of cells
sx, sy, sz = mask.sx, mask.sy, mask.sz      # cell unit
ox, oy, oz = mask.ox, mask.oy, mask.oz      # origin (corner of the "first" grid cell)

deesse_roussillon = DeesseClassifier(
    varnames=['X','Y','Z','Facies'],
    nx=nx, ny=ny, nz=nz,
    sx=sx, sy=sy, sz=sz,
    ox=ox, oy=oy, oz=oz,
    nv=2, varname=['Facies', 'trend'],
    nTI=1, TI=ti_true,
    mask=mask.val,
    rotationUsage=1,            # use rotation without tolerance
    rotationAzimuthLocal=True,  #    rotation according to azimuth: local
    rotationAzimuth=im_angle.val[0,:,:,:],      #    rotation azimuth: map of values
    dataImage=trend,
    outputVarFlag=[True, False],
    distanceType=[0,1],
    nneighboringNode=[50,1],
    distanceThreshold=[0.05, 0.05],
    maxScanFraction=0.5,
    npostProcessingPathMax=1,
    seed=20191201,
    nrealization=40,
    nthreads=40,
)


# In[6]:


# fill here
scan_fractions = [0.1, 0.2, 0.4, 0.8]
eps=1e-5
parameter_selector = GridSearchCV(deesse_roussillon,
                    param_grid=[{'maxScanFraction': scan_fractions,
                                'nneighboringNode': [[8, 1]],
                                'distanceThreshold': [[t+eps, 0.1] for t in [2/16, 4/16]]},
{'maxScanFraction': scan_fractions,
                                'nneighboringNode': [[16, 1]],
                                'distanceThreshold': [[t+eps, 0.1] for t in [2/16, 3/16, 4/16]]},
                                {'maxScanFraction': scan_fractions,
                                'nneighboringNode': [[32, 1]],
                                'distanceThreshold': [[t+eps, 0.1] for t in [1/16, 2/16, 3/16, 4/16]]},
                                {'maxScanFraction': scan_fractions,
                                'nneighboringNode': [[64, 1]],
                                'distanceThreshold': [[t+eps, 0.1] for t in [1/32, 1/16, 2/16, 3/16, 4/16]]},
                               
                               ],
                    scoring=scoring,
                    n_jobs=1,
                    cv=cv,
                    refit=False,
                    verbose=0,
                    error_score='raise',
                    return_train_score=False,
                   )


# In[7]:


try:
    results = pd.read_csv('df_roussillon.csv', index_col=0)
except FileNotFoundError:
    df = pd.DataFrame(readPointSetGslib(SAMPLES_DIR + 'roussillon_observations_600.gslib').to_dict())
    parameter_selector.fit(df[['X','Y','Z']], df['Facies_real00000'])
    results = pd.DataFrame(parameter_selector.cv_results_)
    results.to_csv('df_roussillon.csv')


# In[8]:


results.head()


# In[9]:


# fill here
scan_fractions = [0.001, 0.002, 0.004, 0.006, 0.008, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.4, 0.6]
eps=1e-5
parameter_selector_dsbc = GridSearchCV(deesse_roussillon,
                    param_grid={'maxScanFraction': scan_fractions,
                                'nneighboringNode': [[64, 1], [32, 1], [16,1], [8, 1]],
                                'distanceThreshold': [[eps, 0.1]]},
                    scoring=scoring,
                    n_jobs=1,
                    cv=cv,
                    refit=False,
                    verbose=0,
                    error_score='raise',
                    return_train_score=False,
                   )


# In[ ]:


try:
    results_dsbc = pd.read_csv('df_dsbc_roussillon.csv', index_col=0)
except FileNotFoundError:
    df = pd.DataFrame(readPointSetGslib(SAMPLES_DIR + 'roussillon_observations_600.gslib').to_dict())
    parameter_selector_dsbc.fit(df[['X','Y','Z']], df['Facies_real00000'])
    results_dsbc = pd.DataFrame(parameter_selector_dsbc.cv_results_)
    results_dsbc.to_csv('df_dsbc_roussillon.csv')


# In[ ]:



