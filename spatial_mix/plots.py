import numpy as np
import pymc3 as pm


def density_plot(dens_chain, true_dens, xgrid, ax, color, fill_color, title="",
                alpha=0.7):
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
