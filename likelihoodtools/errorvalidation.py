
"""

Author: F. Thomas
Date: September 22, 2022

"""

import numpy as np
from scipy.linalg import eigh, inv
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt

from likelihoodtools.likelihood import llh_to_sigma, sigma_to_confidence, confidence_to_threshold


def get_correlation_matrix(cov):
    sigma = np.sqrt(np.diag(cov))
    return cov/(np.expand_dims(sigma, -1)*np.expand_dims(sigma, 0))


def get_eigen_decomposition(cov):

    #la, U_ = eig(cov)

    #corr = get_correlation_matrix(cov)
    #zero_correlation = np.abs(corr)<0.1
    #cov[zero_correlation] = 0.

    la, U = eigh(cov)
    La = np.diag(la).real
    #U = U_.real

    return U, La


def get_sub_cov(cov, i, j):
    diff = j-i
    return cov[i:j+1:diff, i:j+1:diff]


def get_2d_error_ellipse(cov, truth, n_points=20):
    t = np.linspace(0, 2*np.pi, n_points)

    U, La = get_eigen_decomposition(cov)
    vec = np.zeros((2, n_points))
    vec[0] = np.cos(t)
    vec[1] = np.sin(t)
    offset = (U@np.sqrt(La)@vec)

    return truth + offset.transpose()


def get_sub_views(truth):
    indices = []

    for i in range(len(truth)):
        for j in range(i+1, len(truth)):
            indices.append((i,j))

    return indices


def get_error_ellipses(cov, truth, n_points=20, phi_max=2*np.pi, endpoint=False):

    ellipses = []
    ellipses_2D = []
    indices = []

    t = np.linspace(0, phi_max, n_points, endpoint=endpoint)

    U, La = get_eigen_decomposition(cov)

    for i in range(len(truth)):
        for j in range(i+1, len(truth)):
            indices.append((i,j))
            vec = np.zeros((truth.shape[0], n_points))
            vec[i] = np.cos(t)
            vec[j] = np.sin(t)
            err_vec = np.sqrt(La)@vec
            offset = (U@err_vec)

            ellipses.append(truth + offset.transpose())
            
            diff = j-i
            ellipses_2D.append(err_vec.transpose()[:,i:j+1:diff])

    return ellipses, indices, ellipses_2D, U


def get_minimum_test_point_set(cov, truth):

    points = []

    for i in range(len(truth)):
        points.append([1. if l==i else 0. for l in range(len(truth))])
        for j in range(i+1, len(truth)):
            points.append([-1/sqrt(2) if l==i or l==j else 0. for l in range(len(truth))])
            points.append([1/sqrt(2) if l==i or l==j else 0. for l in range(len(truth))])

    points_np = np.array(points).transpose()

    U, La = get_eigen_decomposition(cov)
    errvec = np.sqrt(La)@points_np
    offset = U@errvec
    
    return truth + offset.transpose()

def test_ellipses(cost_f, ellipses):

    vals = np.empty((len(ellipses), ellipses[0].shape[0]))
    for i, e in enumerate(ellipses):
        for j, param in enumerate(e):
            vals[i][j] = cost_f(*param)
    return vals


def compare_2d_ellipses(minimizer, truth, minimizer_orig, name=None, plot_submatrix=False):

    indices = get_sub_views(truth)

    for ind in indices:
        minimizer.reset()
        ind0 = ind[0]
        ind1 = ind[1]

        for i in range(len(minimizer.values)):
            minimizer.fixed[i] = True
            if i==ind0 or i==ind1:
                minimizer.fixed[i] = False

        minimizer.migrad()
        minimizer.hesse()

       # print(minimizer.covariance)

        cov = np.array(minimizer.covariance)
       # zero_correlation = np.abs(np.array(minimizer.covariance.correlation()))<0.05
       # cov[zero_correlation] = 0.

        sub_cov = get_sub_cov(cov, ind0, ind1)
        truth_sub = truth[[ind0, ind1]]

        if plot_submatrix:
            sub_name = None
            if name is not None:
                sub_name = f'{name}_x{ind0}_x{ind1}_sub_matrix'
            plot_covariance_matrix(sub_cov, np.array(minimizer.parameters)[[ind0, ind1]], name=sub_name)

        ellipse = get_2d_error_ellipse(sub_cov, truth_sub)
        
        minimizer_orig.draw_contour(minimizer_orig.parameters[ind0],minimizer_orig.parameters[ind1])
        plt.plot(ellipse[:,0], ellipse[:,1], label='error ellipse')
        plt.legend()
        plt.tight_layout()

        if name is not None:
            plt.savefig(f'{name}_x{ind0}_x{ind1}.png', dpi=400)
        
        plt.show()


def test_maximum_discrepancy(cost_f, ellipses):
    vals = test_ellipses(cost_f, ellipses)
    return get_maximum_sigma_discrepancy(-vals)


def calc_maximum_hessian_discrepancy(cov, cost_f, truth, n_points='min'):

    if n_points=='min':
        ellipses = [get_minimum_test_point_set(cov, truth)]
    else:
        ellipses, _, _, _ = get_error_ellipses(cov, truth, n_points=n_points, phi_max=2*np.pi, endpoint=False)
    
    return test_maximum_discrepancy(cost_f, ellipses)


def hessian_is_valid_approximation(cov, cost_f, truth, margin, n_points='min', return_max_discrepancy=False):

    max_discrepancy = calc_maximum_hessian_discrepancy(cov, cost_f, truth, n_points=n_points)

    lower = 1. - margin
    upper = 1. + margin
    in_valid_band = lower<max_discrepancy and max_discrepancy<upper

    if return_max_discrepancy:
        return in_valid_band, max_discrepancy
    else:
        return in_valid_band


def get_maximum_sigma_discrepancy(vals):

    sigma_vals = llh_to_sigma(vals, 0.)
    diff = sigma_vals - 1.
    return sigma_vals.flatten()[np.argmax(abs(diff))]

    
def plot_error_ellipses_with_llh_vals(cov, cost_f, truth, n_points, phi_max, endpoint, ax_names=None, use_sigma=False, name=None):

    ellipses, indices, ellipses_2D, U = get_error_ellipses(cov, truth, n_points=n_points, phi_max=phi_max, endpoint=endpoint)
    vals = test_ellipses(cost_f, ellipses)
    _, _, ellipses_2D_plot, U = get_error_ellipses(cov, truth, n_points=200, phi_max=2*np.pi, endpoint=True)

    if ax_names is not None:
        plot_U_matrix(U, ax_names, name)

    for i, e in enumerate(ellipses):
        ind0, ind1 = indices[i]

        if use_sigma:
            c = llh_to_sigma(-vals[i], 0.)
            c_central = 1.
            c_label = r'$\sigma$ level'
        else:
            c = -vals[i]
            c_central = confidence_to_threshold(0., sigma_to_confidence(1.), 1)
            c_label = 'log-likelihood'

        colors = np.round(c,4)
        min_c = np.min(colors)
        max_c = np.max(colors)
        d_min = abs(c_central-min_c)
        d_max = abs(c_central-max_c)
        max_d = max(d_min, d_max)

        vmin = round(c_central-max_d,8)
        vmax = round(c_central+max_d, 8)


        plt.plot(ellipses_2D_plot[i][:,0], ellipses_2D_plot[i][:,1], c='black', label=r'predicted 1$\sigma$ ellipse')
        plt.scatter(ellipses_2D[i][:,0], ellipses_2D[i][:,1], c=colors, zorder=3, label='actual likelihood', cmap=sns.color_palette("icefire", as_cmap=True), vmin=vmin, vmax=vmax)
        plt.xlabel(f'x{ind0}')
        plt.ylabel(f'x{ind1}')
        cbar = plt.colorbar()

        cbar.set_label(c_label)
        plt.ylim((plt.ylim()[0], plt.ylim()[1]*1.35))
        plt.legend(loc='upper left')
        plt.tight_layout()

        if name is not None:
            plt.savefig(f'{name}_x{ind0}_x{ind1}.png', dpi=400)
        plt.show()


def plot_covariance_matrix(cov, ax_names, name=None):

    fig, ax = plt.subplots()

    corr = get_correlation_matrix(cov)
    diagonal = np.diag(np.ones(cov.shape[0]))==1
    corr[diagonal] = np.nan

    im = ax.imshow(corr, zorder=2, cmap=sns.color_palette("vlag", as_cmap=True), vmin=-1, vmax=1)

    for i in range(cov.shape[0]):
        for j in range(cov.shape[1]):
            text = ax.text(j, i, f'{cov[i, j]:5.2e}',
                        ha="center", va="center", color="black")

    ax.set_xticks(np.arange(len(ax_names)), labels=ax_names)
    ax.set_yticks(np.arange(len(ax_names)), labels=ax_names)

    cbar = plt.colorbar(im)
    cbar.set_label('correlation')

    plt.tight_layout()

    if name is not None:
        plt.savefig(f'{name}.png', dpi=400)

    plt.show()


def plot_U_matrix(U, ax_names, name=None):

    int_names = [f'x{i}' for i in range(len(ax_names))]

    fig, ax = plt.subplots()
    im = ax.imshow(U**2, zorder=2, cmap=sns.color_palette("Blues", as_cmap=True))
    ax.set_yticks(np.arange(len(ax_names)), labels=ax_names)
    ax.set_xticks(np.arange(len(int_names)), labels=int_names)

    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            text = ax.text(j, i, f'{U[i, j]**2:5.2f}',
                        ha="center", va="center", color="black")

    cbar = fig.colorbar(im)
    cbar.set_label('fraction')
    plt.tight_layout()

    if name is not None:
        plt.savefig(f'{name}_U.png', dpi=400)

    plt.show()