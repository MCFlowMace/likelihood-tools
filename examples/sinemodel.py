"""

Author: F. Thomas
Date: May 11, 2023

"""

import numpy as np

def sin_model(A, w, phi, b, x):
    return A*np.sin(w*x + phi) + b


#this function serves as f from the presentation
#due to the parallelization it is somehow important that this function lives on the top level of a python module file
#does not work if defined in notebook
#but can be imported to notebook or can live at the top of a script like demonstrated here
def sin_model_data(A, w, phi, b):
    x = np.linspace(0, 10, 50)
    return sin_model(A, w, phi, b, x)
