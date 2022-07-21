
import numpy as np
from tensorflow.keras import backend
from tensorflow.keras.backend import sigmoid

def swish(x, beta = 1):
    return (x * sigmoid(beta * x))

def MSE_GP(a, b):
    c=a-b
    d=c*c
    e=sum(d)
    f=e/(len(a)+len(b))
    return f

def RMSE_GP(a, b):
    return np.sqrt(MSE_GP(a, b))

def R2_GP(yt,yo):
    return 1-(sum((yt-yo)**2)/sum((yt-np.mean(yt))**2))

def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

def r2(y_true, y_pred):
    SS_res =  backend.sum(backend.square( y_true-y_pred ))
    SS_tot = backend.sum(backend.square( y_true - backend.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + backend.epsilon()) )