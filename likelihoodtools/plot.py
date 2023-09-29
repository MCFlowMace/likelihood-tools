
"""

Author: F. Thomas
Date: November 26, 2021

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from .likelihood import confidence_to_threshold, sigma_to_confidence

import seaborn as sns
sns.set()

    
def scaled_plot(f_plot, scaling):
    
    rc_context = {'figure.figsize':np.array(plt.rcParams['figure.figsize'])*scaling,
                 'lines.linewidth':plt.rcParams['lines.linewidth']*scaling}

    with sns.plotting_context("notebook", font_scale=scaling), plt.rc_context(rc_context):
        f_plot()
 

def make_2d_llh_plot(fit_result, scale_fig=1.0, debug=False, name=None):
    
    llh_scan = fit_result.llh_scan
    llh = llh_scan.llh
    axes = llh_scan.axes
    
    if llh_scan.llh_f is not None:
        llh = llh_scan.llh_f(axes)
    
    ax_names = llh_scan.ax_names
    truth = llh_scan.truth
    
    if len(axes) != 2:
        raise ValueError(f'Attempting 2d plot on likelihood scan of dimension {len(axes)}')
    
    best_fit = fit_result.best_fit_view
    errors = fit_result.errors_view
    confidence_thresholds = fit_result.llh_vals[:-1]
    confidence_levels = fit_result.confidence_levels
    use_sigma_confidence = fit_result.use_sigma_confidence
    
    ll = (best_fit[0]-errors[0,0,0], best_fit[1]-errors[0,1,0])
    width = errors[0,0,0]+errors[0,0,1]
    height = errors[0,1,0]+errors[0,1,1]
    
    
    delta_x = axes[0][1]-axes[0][0]
    delta_y = axes[1][1]-axes[1][0]

    
    def f_plot_main():
        rec = patches.Rectangle(ll, width, height,linewidth=1, edgecolor='r', facecolor='none', zorder=3)

        fig, ax = plt.subplots()
        
        im = ax.imshow(llh.transpose(),origin='lower', 
                        extent=(axes[0][0]-delta_x/2,axes[0][-1]+delta_x/2,
                                axes[1][0]-delta_y/2,axes[1][-1]+delta_y/2), 
                        aspect='auto', zorder=2)
        ax.plot(truth[0], truth[1], marker='x', color='r', label='truth', ls='None')
        
        best_fit_label = 'best fit'
        
        if llh_scan.fixed_view_parameters is not None:
            for key in llh_scan.fixed_view_parameters:
                best_fit_label = best_fit_label + f'\n{key}={llh_scan.fixed_view_parameters[key][0]:.2f}'
        
        ax.plot(best_fit[0], best_fit[1], marker='*', color='b', label=best_fit_label, ls='None')
        #ax.add_patch(rec)

        c_line = ax.contour(llh.transpose(), confidence_thresholds,
                            colors=['darkred', 'red', 'darkorange', 'orange'],
                            origin='lower', extent=(axes[0][0]-delta_x/2,axes[0][-1]+delta_x/2,
                                                    axes[1][0]-delta_y/2,axes[1][-1]+delta_y/2)
                           )

        fmt = {}
        for i, l in enumerate(c_line.levels):
            if not use_sigma_confidence:
                s_confidence = f'{confidence_levels[i]*100:.1f}%'
            else:
                s_confidence = f'{confidence_levels[i]:.1f}$\sigma$'
            fmt[l] = s_confidence

        ax.clabel(c_line, c_line.levels, inline=True, fmt=fmt, fontsize=10)

        ax.set_xlabel(ax_names[0])
        ax.set_ylabel(ax_names[1])

        ax.ticklabel_format(style='sci', scilimits=(-2,4))
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('log-likelihood')
        plt.legend()
        plt.tight_layout()
        
        if name is not None:
            plt.savefig(name + '.png',dpi=400)
        plt.show()
    
    def f_plot_debug():
        
        fig, ax = plt.subplots()

        im = ax.imshow((llh>confidence_thresholds[0]).transpose(),origin='lower', 
                       extent=(axes[0][0]-delta_x/2,axes[0][-1]+delta_x/2,
                                axes[1][0]-delta_y/2,axes[1][-1]+delta_y/2), 
                        aspect='auto', zorder=2)
    
        rec = patches.Rectangle(ll, width, height,linewidth=1, edgecolor='r', facecolor='none', zorder=3)
        #ax.add_patch(rec)
        ax.set_xlabel(ax_names[0])
        ax.set_ylabel(ax_names[1])
        cbar = fig.colorbar(im, ax=ax)
        
        if name is not None:
            plt.savefig(name+'_debug.png', dpi=400)
        plt.show()
        
    scaled_plot(f_plot_main, scale_fig)
    
    if debug:
        scaled_plot(f_plot_debug, scale_fig)

        
        
def make_1d_llh_plot(fit_result, ylim=None,  scale_fig=1.0, name=None, hlines=[]):
    
    llh_scan = fit_result.llh_scan
    
    llh = llh_scan.llh
    axes = llh_scan.axes
    
    if llh_scan.llh_f is not None:
        llh = llh_scan.llh_f(axes)
    
    ax_names = llh_scan.ax_names
    truth = llh_scan.truth
    
    if len(axes) != 1:
        raise ValueError(f'Attempting 1d plot on likelihood scan of dimension {len(axes)}')
    
    best_fit = fit_result.best_fit_view
    errors = fit_result.errors_view
    confidence_thresholds = fit_result.llh_vals[:-1]
    confidence_levels = fit_result.confidence_levels

    left_val = best_fit-errors[:,0,0]
    right_val = best_fit+errors[:,0,1]
    
    def f_plot():

        plt.plot(axes[0], llh)

        if ylim is not None:
            plt.ylim(ylim)

        ylim_min, ylim_max = plt.ylim()
        ymin = ylim_min+0.05*(ylim_max-ylim_min)
        xmin = min(axes[0])

        plt.axvline(truth, color='black', ls='--', label='truth')
        
        best_fit_label = 'best fit'
        
        if llh_scan.fixed_view_parameters is not None:
            for key in llh_scan.fixed_view_parameters:
                best_fit_label = best_fit_label + f'\n{key}={llh_scan.fixed_view_parameters[key][0]:.2f}'
        
        plt.axvline(best_fit, color='r', label=best_fit_label)

        for i, l in enumerate(confidence_levels):

            plt.axvspan(left_val[i], right_val[i], alpha=0.3)
            plt.text(right_val[i],ymin, f'{l*100:.1f}%', rotation='vertical')

        for hline in hlines:
            line_level = confidence_to_threshold(fit_result.llh_vals[-1], sigma_to_confidence(hline), 1)
            plt.axhline(line_level, ls='--')
            plt.text(xmin,line_level, f'{hline:.1f}$\sigma$')

        plt.ticklabel_format(style='sci', scilimits=(-2,4))
        plt.ylabel('log-likelihood')
        plt.xlabel(ax_names[0])
        plt.legend(loc='upper right')
        plt.tight_layout()
                
        if name is not None:
            plt.savefig(name+'.png', dpi=400)
            
        plt.show()
    
    scaled_plot(f_plot, scale_fig)
