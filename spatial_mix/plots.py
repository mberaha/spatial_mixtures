import numpy as np
import pymc3 as pm


def density_plot(dens_chain, true_dens, xgrid, ax, color, fill_color, title="",
                alpha=0.7):
    """
    Plots the estimated density as well as the true density on the same 
    figure.
    
    Parameters
    -----------
    dens_chain: np.array (num_steps, len(xgrid))
        density evaluation for each step of the chain on the grid 'xgrid'
    true_dens: np.array (len(xgrid),)
        true data generating density evaluated on the grid 'xgrid'
    xgrid: np.array(len(xgrid),)
        grid on which the densities are evaluated
    ax: matplotlib.pyplot.ax
        axis object where to plot the figure
    color: str
        color to be used for the mean estimated density
    fill_color: str
        color to be used to fill the 0.95 credible intervals
    title: str, default=""
        if not empty, title to set to the axis
    alpha: double, default=0.7
        alpha for the filling of the credible intervals
    """
    ax.plot(xgrid, true_dens, color="orange",
            lw=2, label="true density")
    intervals = np.array(
        [pm.stats.hpd(dens_chain[:, i], 0.95)
         for i in range(dens_chain.shape[1])])
    ax.plot(xgrid, np.mean(dens_chain, 0), lw=2, label="estimated density",
            color=color)
    ax.fill_between(
        xgrid, intervals[:, 0], intervals[:, 1], alpha=alpha, color=fill_color,
        label="0.95 credible interval")
    if title:
        ax.set_title(title)
