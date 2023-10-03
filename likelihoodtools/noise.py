
import numpy as np


def add_noise(data, likelihood_model):
    
    n = data.size
    
    noise = likelihood_model.gen_noise(n)
    
    return data + noise.reshape(data.shape)


def thermal_noise_var(T, sr, complex_data=True):
    
    R=50
    kb = 1.380649e-23
    
    df = sr if complex_data else sr/2
    return kb*R*T*df