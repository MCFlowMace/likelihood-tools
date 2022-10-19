
"""

Author: F. Thomas
Date: November 26, 2021

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from .data import sort_for_key

import seaborn as sns
sns.set()

SI_prefix = {'y': 1e-24,  # yocto
           'z': 1e-21,  # zepto
           'a': 1e-18,  # atto
           'f': 1e-15,  # femto
           'p': 1e-12,  # pico
           'n': 1e-9,   # nano
           'u': 1e-6,   # micro
           'm': 1e-3,   # mili
           'c': 1e-2,   # centi
           'd': 1e-1,   # deci
           'k': 1e3,    # kilo
           'M': 1e6,    # mega
           'G': 1e9,    # giga
           'T': 1e12,   # tera
           'P': 1e15,   # peta
           'E': 1e18,   # exa
           'Z': 1e21,   # zetta
           'Y': 1e24,   # yotta
    }

    
def get_annotation_x_y_pos(ax):
    
    xmin, xmax, ymin, ymax = ax.axis()
    x_diff = xmax-xmin
    y_diff = ymax-ymin

    x_pos = xmin+0.05*x_diff
    y_pos = ymin+0.95*y_diff
    
    return x_pos, y_pos
    
    
def convert_to(val, prefix):
    
    prefix_val = SI_prefix.get(prefix)
    
    if prefix_val is not None:
        val = val/prefix_val
        
    return val
    
def scaled_plot(f_plot, scaling):
    
    rc_context = {'figure.figsize':np.array(plt.rcParams['figure.figsize'])*scaling,
                 'lines.linewidth':plt.rcParams['lines.linewidth']*scaling}

    with sns.plotting_context("notebook", font_scale=scaling), plt.rc_context(rc_context):
        f_plot()
        
def plot_spectra(plotting_dict_list, event=None, extra_annotations=[], xlim=None, 
                    ylim=None, prefixX='', prefixY='', name=None, scale_fig=1.0, 
                    logscale=False, legend_location='upper right', use_relative_phase=True,
                    extra_plot_f=None):
    
    def f_plot():
        #fig = plt.figure()
            
        data_0 = plotting_dict_list[0]['data']

        unitX = data_0.unitX
        labelX = data_0.labelX

        labelY = data_0.labelY
        unitY = data_0.unitY
        
        figAbs, axAbs = plt.subplots()
        figPhase, axPhase = plt.subplots()
        
        if extra_plot_f is not None:
            extra_plot_f(figAbs, axAbs)
            extra_plot_f(figPhase, axPhase)

        for plotting_dict in plotting_dict_list:

            frequency_domain_data = plotting_dict['data']
            label = plotting_dict['label']

            if (frequency_domain_data.labelY != labelY
                or frequency_domain_data.unitY != unitY):
                raise ValueError('Different y labels or units for plots!')

            frequency = frequency_domain_data.frequencies
            spectrum = frequency_domain_data.spectra

            if (len(frequency.shape) != 1 
                or len(spectrum.shape) != 1 
                or frequency.shape != spectrum.shape):
                raise ValueError('Frequency shape=' + str(frequency.shape) + ' and spectrum shape=' + str(spectrum.shape) + ' not suited for 1-D plot!')

            frequency = convert_to(frequency, prefixX)
            spectrum = convert_to(spectrum, prefixY)
            
            phase = np.unwrap(np.angle(spectrum)*2)/2
           # phase = np.angle(spectrum)
           
            if use_relative_phase:
                phase = phase - phase[0]

            axAbs.step(frequency, np.abs(spectrum), label=label)
            axPhase.step(frequency, phase, label=label)

        if xlim is not None:
            axAbs.set_xlim(xlim[0], xlim[1])
            axPhase.set_xlim(xlim[0], xlim[1])
            
        if ylim is not None:
            axAbs.set_ylim(ylim[0], ylim[1])
            axPhase.set_ylim(ylim[0], ylim[1])

        xlabel = labelX + ' [' + prefixX + unitX + ']'
        ylabel = '|' + labelY + '| [' + prefixY + unitY + ']'
        
        if event is not None:
            annotation_pos_abs = get_annotation_x_y_pos(axAbs)
            annotation_pos_phase = get_annotation_x_y_pos(axPhase)
            annotation_label = str(event)
            
            for annotation in extra_annotations:
                annotation_label += '\n' + annotation

            axAbs.annotate(annotation_label, annotation_pos_abs, 
                            bbox=dict(facecolor='white', edgecolor='black', alpha=0.5))
                            
            axPhase.annotate(annotation_label, annotation_pos_phase, 
                            bbox=dict(facecolor='white', edgecolor='black', alpha=0.5))

        axAbs.set_xlabel(xlabel)
        axAbs.set_ylabel(ylabel)
        
        axPhase.set_xlabel(xlabel)
        axPhase.set_ylabel('phase')
        
        axAbs.legend(loc=legend_location)
        axPhase.legend(loc=legend_location)

        if logscale:
            axAbs.set_yscale('log')
            axPhase.set_yscale('log')

        if name is not None:
            figAbs.savefig(name+'_spectra_abs.svg', bbox_inches="tight")
            figPhase.savefig(name+'_spectra_phase.svg', bbox_inches="tight")

        plt.show()
        
    scaled_plot(f_plot, scale_fig)
    
def plot_beamforming(data, R, cbar_label='', ax_unit='cm',
                        scale_fig=1.0):
    
    def f_plot():
        fig, ax = plt.subplots()
        im_masked = np.ma.masked_where(data==0,data)
        im=ax.imshow(im_masked,extent=(-R,R,-R,R),origin='lower', zorder=2)

        cbar = fig.colorbar(im)
        ax.set_aspect('equal')
        ax.set_xlim(-(R+0.5),R+0.5)
        ax.set_ylim(-(R+0.5),R+0.5)
        
        xlabel='x[' + ax_unit + ']'
        ylabel='y[' + ax_unit + ']'
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        cbar.ax.set_ylabel(cbar_label)
    
    scaled_plot(f_plot, scale_fig)

def plot_TS(plotting_dict_list, xlim=None, ylim=None, prefixX='', prefixY='', 
            name=None, scale_fig=1.0, logscale=False, legend_location='upper right'):
    
    def f_plot():
        #fig = plt.figure()
        data_0 = plotting_dict_list[0]['data']

        unitX = data_0.unitX
        labelX = data_0.labelX

        labelY = data_0.labelY
        unitY = data_0.unitY
        
        figReal, axReal = plt.subplots()
        figImag, axImag = plt.subplots()

        for plotting_dict in plotting_dict_list:

            time_domain_data = plotting_dict['data']
            label = plotting_dict['label']

            if (time_domain_data.labelY != labelY
                or time_domain_data.unitY != unitY):
                raise ValueError('Different y labels or units for plots!')

            timestamps = time_domain_data.timestamps
            timeseries = time_domain_data.timeseries

            if (len(timestamps.shape) != 1 
                or len(timeseries.shape) != 1 
                or timestamps.shape != timeseries.shape):
                raise ValueError('Timestamps shape=' + str(timestamps.shape) + ' and timeseries shape=' + str(timeseries.shape) + ' not suited for 1-D plot!')

            timestamps = convert_to(timestamps, prefixX)
            timeseries = convert_to(timeseries, prefixY)

            axReal.step(timestamps, timeseries.real, label=label)
            axImag.step(timestamps, timeseries.imag, label=label)

        if xlim is not None:
            axReal.set_xlim(xlim[0], xlim[1])
            axImag.set_xlim(xlim[0], xlim[1])
            
        if ylim is not None:
            axReal.set_ylim(ylim[0], ylim[1])
            axImag.set_ylim(ylim[0], ylim[1])

        xlabel = labelX + ' [' + prefixX + unitX + ']'
        ylabel = labelY + ' [' + prefixY + unitY + ']'

        axReal.set_xlabel(xlabel)
        axReal.set_ylabel(ylabel)
        
        axImag.set_xlabel(xlabel)
        axImag.set_ylabel(ylabel)
        
        axReal.legend(loc=legend_location)
        axImag.legend(loc=legend_location)

        if name is not None:
            figReal.savefig(name+'_TSReal.svg')
            figImag.savefig(name+'_TSImag.svg')

        plt.show()
        
    scaled_plot(f_plot, scale_fig)


def plot_keys_in_dict_of_arrays(dict_of_arrays, x_key, y_keys, xunit, yunit, scale_fig=1.0, name=None):
    
    def f_plot():
        
        local_dict = dict_of_arrays

        sort_for_key(dict_of_arrays, x_key)

        for y_key in y_keys:

            plt.plot(local_dict[x_key], local_dict[y_key], label=y_key, marker='v')


        plt.xlabel(x_key + ' [' + xunit+ ']')
        plt.ylabel(yunit)

        plt.legend()
        
        if name is not None:
            plt.savefig(name+'_'+x_key+'.svg')
        plt.show()
    
    scaled_plot(f_plot, scale_fig)


def make_2d_llh_plot(fit_result, scale_fig=1.0, debug=False, name=None):
    
    llh_scan = fit_result.llh_scan
    
    llh = llh_scan.llh
    axes = llh_scan.axes
    ax_names = llh_scan.ax_names
    truth = llh_scan.truth
    
    if len(axes) != 2:
        raise ValueError(f'Attempting 2d plot on likelihood scan of dimension {len(axes)}')
    
    best_fit = fit_result.best_fit_view
    errors = fit_result.errors_view
    confidence_thresholds = fit_result.llh_vals[:-1]
    confidence_levels = fit_result.confidence_levels
    
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
            fmt[l] = f'{confidence_levels[i]*100:.0f}%'

        ax.clabel(c_line, c_line.levels, inline=True, fmt=fmt, fontsize=10)

        ax.set_xlabel(ax_names[0])
        ax.set_ylabel(ax_names[1])
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('log-likelihood')
        plt.legend()
        
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

        
        
def make_1d_llh_plot(fit_result,  scale_fig=1.0, name=None):
    
    llh_scan = fit_result.llh_scan
    
    llh = llh_scan.llh
    axes = llh_scan.axes
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

        ymin = min(llh)

        plt.axvline(truth, color='black', ls='--', label='truth')
        
        best_fit_label = 'best fit'
        
        if llh_scan.fixed_view_parameters is not None:
            for key in llh_scan.fixed_view_parameters:
                best_fit_label = best_fit_label + f'\n{key}={llh_scan.fixed_view_parameters[key][0]:.2f}'
        
        plt.axvline(best_fit, color='r', label=best_fit_label)

        for i, l in enumerate(confidence_levels):

            plt.axvspan(left_val[i], right_val[i], alpha=0.3)
            plt.text(right_val[i],ymin, f'{l*100:.0f}%', rotation='vertical')

        plt.ylabel('log-likelihood')
        plt.xlabel(ax_names[0])
        plt.legend(loc='upper right')
        
                
        if name is not None:
            plt.savefig(name+'.png', dpi=400)
            
        plt.show()
    
    scaled_plot(f_plot, scale_fig)
