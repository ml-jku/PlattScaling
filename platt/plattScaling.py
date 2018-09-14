#Copyright (C) 2018 Andreas Mayr, Guenter Klambauer
#Licensed under GNU General Public License v3.0 (see https://github.com/ml-jku/PlattScaling/blob/master/LICENSE)

import warnings
import numpy as np
import scipy
import scipy.stats
import os

plattLib = np.ctypeslib.load_library('libPlatt',os.path.dirname(__file__))

plattLib.plattScaling.restype=None
plattLib.plattScaling.argtypes=[
np.ctypeslib.ctypes.c_int32,
np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='CONTIGUOUS,ALIGNED'),
np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='CONTIGUOUS,ALIGNED'),
np.ctypeslib.ctypes.c_double,
np.ctypeslib.ctypes.c_double,
np.ctypeslib.ctypes.POINTER(np.ctypeslib.ctypes.c_double),
np.ctypeslib.ctypes.POINTER(np.ctypeslib.ctypes.c_double),
np.ctypeslib.ctypes.POINTER(np.ctypeslib.ctypes.c_double)
]



def plattScaling(predictions, labels, norm=True):
  assert(type(predictions)==np.ndarray)
  assert(np.logical_or(predictions.dtype=="float32", predictions.dtype=="float64"))
  assert(type(labels)==np.ndarray)
  assert(np.logical_or(labels.dtype=="bool", np.logical_or(labels.dtype=="int32", labels.dtype=="int64")))
  if labels.dtype=="bool":
    assert(np.sort(np.unique(labels)).tolist()[0]==0)
    assert(np.sort(np.unique(labels)).tolist()[1]==1)
  assert(len(predictions)==len(labels))
  assert(np.std(predictions)!=0.0)
  
  if np.any(np.logical_or(np.isnan(predictions), np.isnan(labels))):
    warnings.warn("Detected NAs in predictions or labels. Removing.")
    mask=np.logical_not(np.logical_or(np.isnan(predictions), np.isnan(labels)))
    predictions=predictions[mask]
    labels=labels[mask]

  mm=np.median(predictions)
  ss=np.std(predictions)
  if not norm:
    mm=0.0
    ss=1.0
  
  predictions=(predictions - mm)/ss
  predictions=np.ascontiguousarray(predictions)
  predictions=predictions.astype(np.float64)
  labels=labels>0.5
  prior0=float(np.sum(np.logical_not(labels)))
  prior1=float(np.sum(labels))
  labels=labels.astype(np.int32)
  
  A=np.ctypeslib.ctypes.c_double(0.0)
  B=np.ctypeslib.ctypes.c_double(0.0)
  suc=np.ctypeslib.ctypes.c_double(0.0)
  Ap=np.ctypeslib.ctypes.pointer(A)
  Bp=np.ctypeslib.ctypes.pointer(B)
  sucp=np.ctypeslib.ctypes.pointer(suc)
  plattLib.plattScaling(len(labels), predictions, labels, prior0, prior1, Ap, Bp, sucp)
  
  A=A.value
  B=B.value
  suc=suc.value
  
  if suc<0.5:
    warnings.warn("Platt scaling was not successful!")  
  
  success=True
  if A > (-1e-7):
    warnings.warn("Curve-fitting was not successful")
    A=-1.0
    B=0.0
    success=False
  
  newPred=1.0/(1.0+np.exp(A*predictions+B))
  
  if (np.logical_or(np.std(newPred)==0.0, np.std(predictions)==0.0)):
    warnings.warn("Standard deviation of predictions is zero.")
    A=-1.0
    B=0.0
    success=False
  else:
    scor=scipy.stats.spearmanr(newPred, predictions)
    
    if np.isnan(scor.correlation):
      A=-1.0
      B=0.0
      success=False
    if scor.correlation < 0.99:
      warnings.warn("Platt Scaling changed the ranking of the values.")
      success=False

  
  retDict={
    "type": "plattScalingResult",
    "pred": newPred,
    "A": A,
    "B": B,
    "norm": (mm,ss),
    "success": (success and suc>0.5),
    "successSoft": success
  }
  
  return retDict



def predictProb(pS, predictions):
  assert(type(pS)==dict)
  assert("type" in pS)
  assert(pS["type"]=="plattScalingResult")
  assert("A" in pS)
  A=pS["A"]
  assert("B" in pS)
  B=pS["B"]
  assert("norm" in pS)
  mm=pS["norm"][0]
  ss=pS["norm"][1]
  assert(type(A)==float)
  assert(type(B)==float)
  assert(type(predictions)==np.ndarray)
  assert(np.logical_or(predictions.dtype=="float32", predictions.dtype=="float64"))

  predictions=predictions.astype(np.float64)
  predictions=(predictions - mm)/ss
  vec=1.0/(1.0+np.exp(A*predictions+B))
  return vec


