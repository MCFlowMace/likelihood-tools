"""

Author: F. Thomas
Date: May 11, 2023

"""

import numpy as np

from scipy.special import expit

def sin_model(A, w, phi, b, x):
    return A*np.sin(w*x + phi) + b


#this function serves as f from the presentation
#due to the parallelization it is somehow important that this function lives on the top level of a python module file
#does not work if defined in notebook
#but can be imported to notebook or can live at the top of a script like demonstrated here
def sin_model_data(A, w, phi, b):
    x = np.linspace(0, 1000, 5000)
    return sin_model(A, w, phi, b, x)


def smoothing(a=1., x0=0., end=False):

    if end:
        f = lambda x: 1 - expit(a*(x - x0))
    else:
        f = lambda x: expit(a*(x - x0))

    return f


class ChirpModel:

    def __init__(self, sr, tau, spacing_samples=300000, smoothing_factor=None):
        self.sr = sr
        self.dt = 1/sr
        self.tau = tau
        self.n_samples = int(tau/self.dt) + spacing_samples
        self.smoothing_factor = smoothing_factor

    def __call__(self, A, w, alpha, phi0, t0, return_window=False):
        t = np.arange(self.n_samples)*self.dt
        data = np.zeros_like(t, dtype=np.complex128)

        if self.smoothing_factor is None:
            signal_ind = (t>t0)&(t<t0+self.tau)
        else:
            signal_ind = np.arange(self.n_samples)

        data[signal_ind] = A*np.exp(1.0j*(w*(t[signal_ind]-t0) + alpha/2*(t[signal_ind]-t0)**2 + phi0))

        t_end = t0 + self.tau

        if self.smoothing_factor is not None:
            window = smoothing(self.smoothing_factor, t0)(t)*smoothing(self.smoothing_factor, t_end, end=True)(t)
            data = data*window
        else:
            window = None

        if return_window:
            return data, window
        else:
            return data


class SinModel:

    def __init__(self, sr, tau):
        self.sr = sr
        self.dt = 1/sr
        self.tau = tau
        self.n_samples = int(tau/self.dt) + 100

    def __call__(self, A, w, phi0, t0):
        t = np.arange(self.n_samples)*self.dt
        data = np.zeros_like(t, dtype=np.complex128)
        signal_ind = (t>t0)&(t<t0+self.tau)
        data[signal_ind] = A*np.exp(1.0j*(w*(t[signal_ind]-t0) + phi0))

        return data