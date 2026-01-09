from __future__ import annotations

import numpy as np
import pandas as pd 
import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.colors import Normalize, LogNorm, SymLogNorm
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.ticker import NullFormatter
import matplotlib as mpl
import matplotlib.ticker as mticker
from matplotlib.colors import Colormap, ListedColormap
from matplotlib.path import Path        # <-- Path defined here
from matplotlib.markers import MarkerStyle
import matplotlib.cm as cm
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from . import config

class KIT_heatmap:
    """Utilities to bin data and draw 2D heatmaps with flexible ticks/scale."""

    # ---------- binning ---------- #
    @staticmethod
    def bin_2d(df, x_cols, y_cols, x_bins, y_bins, dropna=True):
        """
        Bivariate 2D histogram with edge-based clipping.

        Returns
        -------
        H : (nx, ny) ndarray            # counts with x along axis 0, y along axis 1
        xedges, yedges : 1D ndarrays    # bin edges actually used
        n_base : int                    # number of finite (x,y) pairs BEFORE clipping to edges
        """
        # Get raw arrays (accept column name or raw array/Series)
        x = df[x_cols].to_numpy() if isinstance(x_cols, str) else np.asarray(x_cols)
        y = df[y_cols].to_numpy() if isinstance(y_cols, str) else np.asarray(y_cols)

        # finite mask
        mask = np.isfinite(x) & np.isfinite(y) if dropna else np.ones_like(x, dtype=bool)
        x, y = x[mask], y[mask]
        n_base = len(x)  # used for normalization (pre-clip size)

        # Resolve bin edges: allow int (compute edges) or explicit sequences (use as-is)
        xedges = np.histogram_bin_edges(x, bins=x_bins) if np.isscalar(x_bins) else np.asarray(x_bins, dtype=float)
        yedges = np.histogram_bin_edges(y, bins=y_bins) if np.isscalar(y_bins) else np.asarray(y_bins, dtype=float)

        # Clip to provided/derived edges so only in-range samples are binned
        in_x = (x >= xedges[0]) & (x <= xedges[-1])
        in_y = (y >= yedges[0]) & (y <= yedges[-1])
        keep = in_x & in_y
        x_clip, y_clip = x[keep], y[keep]

        # 2D histogram using those edges
        H, xedges, yedges = np.histogram2d(x_clip, y_clip, bins=[xedges, yedges])

        return H, xedges, yedges, n_base

    # ---------- utilities ---------- #
    @staticmethod
    def _centers(edges):
        """Return bin centers given bin edges."""
        edges = np.asarray(edges, dtype=float)
        return 0.5 * (edges[:-1] + edges[1:])

    @staticmethod
    def _data_to_pos(vals, edges, tick_at="center"):
        """
        Map real values (vals) to heatmap positions given bin edges.
        tick_at = "center" or "edge"
        """
        vals = np.asarray(vals, dtype=float)
        edges = np.asarray(edges, dtype=float)
        idx = np.searchsorted(edges, vals, side="right") - 1
        idx = np.clip(idx, 0, len(edges) - 2)

        if tick_at == "edge":
            # allow a tick exactly on the final edge
            last_mask = np.isclose(vals, edges[-1])
            out = idx.astype(float)
            out[last_mask] = len(edges) - 1
            return out
        else:  # "center"
            return idx + 0.5

    def _make_default_ticks(self, edges, tickstep=1, integer=False, tick_at="center"):
        """
        Default tick positions and labels.
        tick_at = "center" → ticks at bin centers
        tick_at = "edge"   → ticks at bin left edges + final right/top edge
        """
        centers = self._centers(edges)

        if tick_at == "edge":
            pos = list(np.arange(len(edges) - 1)[::tickstep])
            labels = [
                (str(int(edges[i])) if integer else f"{edges[i]:.3g}")
                for i in range(0, len(edges) - 1, tickstep)
            ]
            pos.append(len(edges) - 1)
            labels.append(str(int(edges[-1])) if integer else f"{edges[-1]:.3g}")
        else:
            pos = list(np.arange(len(centers))[::tickstep] + 0.5)
            labels = [
                (str(int(round(v))) if integer else f"{v:.3g}")
                for v in centers[::tickstep]
            ]
        return pos, labels

    def _make_custom_ticks(self,
                           ticks,
                           edges,
                           labels=None,
                           integer=False,
                           space="data",
                           tick_at="center"):
        """
        Custom tick positions and labels.
        space = "data"  → ticks given in data space, convert to heatmap coords
        space = "index" → ticks already in heatmap coords
        """
        if space == "data":
            pos = self._data_to_pos(ticks, edges, tick_at=tick_at)
            if labels is None:
                ticks_arr = np.asarray(ticks)
                labels = [str(int(round(v))) if integer else f"{v:g}" for v in ticks_arr]
        else:
            pos = ticks
        return pos, labels

    @staticmethod
    def _resolve_cmap(cmap, bad_color="white"):
        """
        Accepts a Colormap object, a Matplotlib colormap name, a Seaborn palette name,
        or a list of colors. Returns a Matplotlib Colormap with bad set to bad_color.
        """
        if isinstance(cmap, Colormap):
            cm = cmap
        elif isinstance(cmap, str):
            try:
                cm = mpl.colormaps[cmap]
            except Exception:
                cm = sns.color_palette(cmap, as_cmap=True)
        elif isinstance(cmap, (list, tuple)):
            cm = ListedColormap(cmap)
        else:
            cm = mpl.colormaps['YlGnBu']
        cm = cm.copy()
        cm.set_bad(bad_color)
        return cm

    # ---------- data prep ---------- #
    def gen_heatmap_data(self,
                         df,
                         x_cols,
                         y_cols,
                         x_bins,
                         y_bins,
                         normalize=True,
                         log10=True,
                         symlog=False,
                         linthresh=1e-3,
                         vmin=None,
                         vmax=None):
        """
        Generate binned heatmap data from a dataframe.

        Returns
        -------
        H : 2D array
            Processed heatmap values (normalized, log10, masked, transposed).
        xedges, yedges : arrays
            Bin edges along x and y.
        n_base : int
            Base normalization count (from bin_2d).
        norm : Normalize or None
            SymLogNorm if symlog=True, else None.
        """
        H, xedges, yedges, n_base = self.bin_2d(df, x_cols, y_cols, x_bins, y_bins)

        # normalize
        if normalize and n_base > 0:
            H = H / n_base

        # log scaling (only once!)
        if log10 and not symlog:
            with np.errstate(divide='ignore', invalid='ignore'):
                H = np.where(H > 0, np.log10(H), np.nan)

        # symlog normalization
        norm = None
        if symlog:
            vmin_auto, vmax_auto = np.nanmin(H), np.nanmax(H)
            if np.isfinite(vmin_auto) and np.isfinite(vmax_auto) and vmin_auto != vmax_auto:
                norm = mpl.colors.SymLogNorm(linthresh=linthresh,
                                             linscale=1.0,
                                             vmin=vmin if vmin is not None else vmin_auto,
                                             vmax=vmax if vmax is not None else vmax_auto)

        # mask outside vmin/vmax
        if vmin is not None:
            H = np.where(H < vmin, np.nan, H)
        if vmax is not None:
            H = np.where(H > vmax, np.nan, H)

        return H.T, xedges, yedges, n_base, norm

    # ---------- main heatmap ---------- #
    def visual_heatmap_2d(self,
                          ax,
                          df,
                          x_cols,
                          y_cols,
                          x_bins,
                          y_bins,
                          normalize=True,
                          log10=True,
                          cbar=True,
                          lw=2,
                          cbar_label="",
                          xtickstep=1,
                          ytickstep=1,
                          integer_ticks=False,
                          symlog=False,
                          linthresh=1e-3,
                          xticks=None,
                          yticks=None,
                          xticklabels=None,
                          yticklabels=None,
                          tick_space="index",
                          tick_at="center",                  # "center" or "edge"
                          cbar_orientation="vertical",
                          cbar_ticks=None,
                          cbar_ticklabels=None,
                          return_cbar=False,
                          cmap="YlGnBu",
                          cmap_bad="white",
                          bin_minor_ticks_x=False,
                          bin_minor_ticks_y=False,
                          bin_minor_len=3,
                          bin_minor_w=0.8,
                          # --- tick appearance ---
                          tick_labelsize=10,
                          tick_length=4,
                          tick_width=1.0,
                          # --- color scaling ---
                          vmin=None,
                          vmax=None,
                          return_heatmap_data=False):
        """
        Draw heatmap with flexible ticks, colorbar, and optional manual vmin/vmax.
        """
        H, xedges, yedges, n_base, norm = self.gen_heatmap_data(
            df, x_cols, y_cols, x_bins, y_bins,
            normalize=normalize, log10=log10,
            symlog=symlog, linthresh=linthresh,
            vmin=vmin, vmax=vmax
        )

        # resolve cmap & draw
        cmap_resolved = self._resolve_cmap(cmap, bad_color=cmap_bad)
        sns.heatmap(
            H,
            cmap=cmap_resolved,
            cbar=cbar,
            ax=ax,
            norm=norm,
            vmin=vmin if not symlog else None,
            vmax=vmax if not symlog else None,
            cbar_kws={"label": cbar_label, "orientation": cbar_orientation}
        )
        ax.invert_yaxis()

        # ---- major ticks ----
        if xticks is None:
            xt, xlbl = self._make_default_ticks(xedges, tickstep=xtickstep,
                                                integer=integer_ticks, tick_at=tick_at)
        else:
            xt, xlbl = self._make_custom_ticks(xticks, xedges, labels=xticklabels,
                                               integer=integer_ticks, space=tick_space, tick_at=tick_at)
        ax.set_xticks(xt)
        if xlbl is not None:
            ax.set_xticklabels(xlbl, rotation=0)

        if yticks is None:
            yt, ylbl = self._make_default_ticks(yedges, tickstep=ytickstep,
                                                integer=integer_ticks, tick_at=tick_at)
        else:
            yt, ylbl = self._make_custom_ticks(yticks, yedges, labels=yticklabels,
                                               integer=integer_ticks, space=tick_space, tick_at=tick_at)
        ax.set_yticks(yt)
        if ylbl is not None:
            ax.set_yticklabels(ylbl, rotation=0)

        # ---- minor ticks ----
        if tick_at == "center":
            x_minor = np.arange(len(xedges) - 1) + 0.5
            y_minor = np.arange(len(yedges) - 1) + 0.5
        else:  # "edge"
            x_minor = np.arange(len(xedges))
            y_minor = np.arange(len(yedges))

        if bin_minor_ticks_x:
            ax.set_xticks(x_minor, minor=True)
            ax.xaxis.set_minor_formatter(mticker.NullFormatter())
            ax.tick_params(axis='x', which='minor', length=bin_minor_len, width=bin_minor_w)

        if bin_minor_ticks_y:
            ax.set_yticks(y_minor, minor=True)
            ax.yaxis.set_minor_formatter(mticker.NullFormatter())
            ax.tick_params(axis='y', which='minor', length=bin_minor_len, width=bin_minor_w)

        # ---- major tick appearance ----
        ax.tick_params(axis='both', which='major',
                       labelsize=tick_labelsize,
                       length=tick_length,
                       width=tick_width)

        # ---- colorbar ----
        cbar_obj = None
        if cbar:
            for m in list(ax.collections) + list(ax.images):
                cb = getattr(m, "colorbar", None)
                if isinstance(cb, mpl.colorbar.Colorbar):
                    cbar_obj = cb
                    break

            if cbar_obj is not None:
                if cbar_ticks is not None:
                    cbar_obj.set_ticks(cbar_ticks)
                if cbar_ticklabels is not None:
                    cbar_obj.set_ticklabels(cbar_ticklabels)
                cbar_obj.update_ticks()

        if return_cbar and return_heatmap_data:
            return cbar_obj, H
        elif return_cbar:
            return cbar_obj
        elif return_heatmap_data:
            return H
        else:
            return None



#FUNCTIONS FOR CREATING PANELS
def make_axes_layout(shape_list, 
                     order = 'row', 
                     figsize = (10, 8), 
                     gridspec_kw = None):
    """
    Create a figure with flexible subplot layout.

    Parameters
    ----------
    shape_list : list of int
        If order='row': list of number of axes per row (e.g. [1,3,3]).
        If order='col': list of number of axes per column.
    order : {'row','col'}, default 'row'
        Whether to interpret shape_list by rows or columns.
    figsize : tuple, default (10,8)
        Figure size.
    gridspec_kw : dict, optional
        Additional keyword arguments for matplotlib.gridspec.GridSpec
        (e.g., width_ratios, height_ratios, wspace, hspace).

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : list of Axes
        Flattened list of Axes in creation order.
    """
    if gridspec_kw is None:
        gridspec_kw = {}

    if order == 'row':
        nrows = len(shape_list)
        ncols = max(shape_list)
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(nrows, ncols, figure=fig, **gridspec_kw)

        axes = []
        for irow, n_in_row in enumerate(shape_list):
            if n_in_row == ncols:
                for j in range(n_in_row):
                    ax = fig.add_subplot(gs[irow, j])
                    axes.append(ax)
            else:
                width = ncols // n_in_row
                for j in range(n_in_row):
                    start = j * width
                    end = start + width
                    ax = fig.add_subplot(gs[irow, start:end])
                    axes.append(ax)
        return fig, axes

    elif order == 'col':
        ncols = len(shape_list)
        nrows = max(shape_list)
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(nrows, ncols, figure=fig, **gridspec_kw)

        axes = []
        for jcol, n_in_col in enumerate(shape_list):
            if n_in_col == nrows:
                for i in range(n_in_col):
                    ax = fig.add_subplot(gs[i, jcol])
                    axes.append(ax)
            else:
                height = nrows // n_in_col
                for i in range(n_in_col):
                    start = i * height
                    end = start + height
                    ax = fig.add_subplot(gs[start:end, jcol])
                    axes.append(ax)
        return fig, axes

    else:
        raise ValueError("order must be 'row' or 'col'")

def axis_add_tick(ax,i):
    # get existing ticks and labels
    ticks = list(ax.get_yticks())
    labels = [str(t) for t in ticks]

    # add 0 if missing
    if i not in ticks:
        ticks.append(i)
        labels.append(str(i))

    # sort them together
    ticks, labels = zip(*sorted(zip(ticks, labels)))

    # apply
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)


def remove_axis_ticktext(ax, axis='x'):
    if axis == 'x':
        ax.xaxis.set_major_formatter(NullFormatter())
    elif axis == 'y':
        ax.yaxis.set_major_formatter(NullFormatter())

def set_leg_bbox(ax, 
                 bx = 1.05, 
                 by = 1):
    
    leg = ax.get_legend()
    
    if leg is not None:
        leg.set_bbox_to_anchor((bx, by)) 

def set_percent_yticks(ax, decimals=2, factor=100):
    """Format current y-ticks as percentage strings."""
    yticks = ax.get_yticks()
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{y*factor:.{decimals}f}" for y in yticks])

def set_percent_xticks(ax, decimals=2, factor=100):
    """Format current x-ticks as percentage strings."""
    xticks = ax.get_xticks()
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{x*factor:.{decimals}f}" for x in xticks])


#FUNCTION FOR CREATING DATA FOR VISUALIZATION
def ax_visual_label(ax, DICT_label, legend_pars): 
    Patches = [mpatches.Patch(color=l, label=c)  for c,l in DICT_label.items()]       
    ax.legend(handles= Patches, 
              loc = legend_pars['loc'], 
              fontsize = legend_pars['fontsize'],
              title_fontsize = legend_pars['title_fontsize'])

def ax_visual_xticklabel(ax, DICT_xtl): 
    xt, xtl, rot, size = (DICT_xtl[c] for c in ['xt', 'xtl','rot','size'])
    ax.set_xticks(xt)
    ax.set_xticklabels(xtl, rotation = rot, size = size)

def ax_visual_yticklabel(ax, DICT_ytl): 
    yt, ytl, rot, size = (DICT_ytl[c] for c in ['yt', 'ytl','rot','size'])
    ax.set_yticks(yt)
    ax.set_yticklabels(ytl, rotation = rot, size = size)

def ax_visual_ticklabel(ax, DICT_xtl, axis='cx'): 
    xt, xtl, rot, size = (DICT_xtl[c] for c in ['t', 'tl','rot','size'])
    if axis=='x':
        ax.set_xticks(xt)
        ax.set_xticklabels(xtl, rotation = rot, size = size)
    if axis=='y':
        ax.set_yticks(xt)
        ax.set_yticklabels(xtl, rotation = rot, size = size)   

def ax_visual_labeltitles(ax, DICT_lt): 
    xl = DICT_lt['xlabel']
    yl = DICT_lt['ylabel']
    title = DICT_lt['title']
    size_l = DICT_lt['label_size']
    size_t = DICT_lt['title_size']
    ax.set_xlabel(xl, size = size_l)
    ax.set_ylabel(yl, size = size_l)
    ax.set_title(title, size = size_t)

def ax_visual_legend(ax, 
                     DICT_legend, 
                     List_fc = None): 
    
    Colors = DICT_legend['colors']
    Indicators = DICT_legend['classes']    
    Patches = [mpatches.Patch(edgecolor = c, 
                              facecolor= c, 
                              label=l)  for c,l in zip(Colors, Indicators)]
    if List_fc is not None:
        Patches = [mpatches.Patch(label=l, 
                                  edgecolor = c, 
                                  facecolor=f)  for c,l,f in zip(Colors, Indicators, List_fc)]

    ax.legend(handles = Patches, 
              title   = DICT_legend['title'], 
              loc            = DICT_legend['loc'], 
              fontsize       = DICT_legend['fontsize'],
              title_fontsize = DICT_legend['title_fontsize'])


def gen_DICT_ax_visual(ax_viz = 'label_titles', 
                       return_all = False):

    DICT_ax_visual = {'label_ticks': {'t': '', 
                                      'tl': '',
                                      'rot': 90,
                                      'size': 20},
                      
                      'label_titles': {'xlabel': '', 
                                       'ylabel': '',
                                       'title': '',
                                       'label_size': 20,
                                       'title_size': 25}, 
                    
                      'legend' : {'classes': [],
                                  'colors': [], 
                                  'title': '', 
                                  'loc': 'upper left', 
                                  'fontsize': 15, 
                                  'title_fontsize': 15}}
    
    if return_all:
        return DICT_ax_visual
    
    return DICT_ax_visual[ax_viz]


def subset_df_feature(df,f):
    '''
    df : dataframe
    f  : feature name 
    '''
    #feature unique values 
    f_vals = df[f].unique()
    #subset dataframe according to feature f records 
    return {v: df[df[f]==v] for v in f_vals}


def visual_imshow(ax, 
                  df_tbr, 
                  Colors, 
                  legend_pars = None):
    '''
    legend_pars has keys: 'loc', 'fontsize', 'title_fontsize'
    '''
    
    cmap = ListedColormap(Colors)  # 0 -> blue, 1 -> yellow
    boundaries = np.arange(0,len(Colors)+1)-0.5   # Define boundaries that map values 1 -> blue, 2 -> red
    norm = BoundaryNorm(boundaries, cmap.N)

    ax.imshow(df_tbr, 
               aspect='auto', 
               cmap=cmap, 
               norm = norm,
               origin = 'lower', 
               interpolation = None,
               interpolation_stage='rgba')
    
    

def get_temporal_ticks(Inds, 
                       only_minute = True, 
                       and_first_day_hour = False, 
                       and_first_month_day = False):

    #show bool of indexes
    Inds_sb = Inds.minute == 0
    if and_first_day_hour:
        Inds_sb = Inds_sb & (Inds.hour == 0)
    if and_first_month_day:
        Inds_sb = Inds_sb & (Inds.day==1) 

    return Inds, Inds_sb
    
def set_temporal_xticks(ax, 
                        Inds, 
                        Inds_sb,
                        axis = 'x',
                        str_format = '%Y-%m-%d %H'):

    #tick's index
    x_ti = np.arange(0,len(Inds))[Inds_sb]
    #tick's vals 
    x_tv = np.array(Inds)[Inds_sb]
    if str_format is not None:
        x_tv = pd.to_datetime(x_tv).strftime(str_format)
    if axis=='y':
        ax.set_yticks(x_ti,x_tv , rotation = 90, size = 15)
    else:     
        ax.set_xticks(x_ti,x_tv , rotation = 90, size = 15)
    

def plot_stacked_bar(df_count_, 
                     Colors, 
                     Labels, 
                     xs = None, 
                     x_nu = 0, 
                     bar_width = 1, 
                     legend=  False, 
                     legend_loc= 'upper left', log_scale=  False): 
    '''
    plot a count-dataset as a stacked bar
    '''
    df_count = df_count_.copy()
    
    #getting the x-range 
    Cols = df_count.columns 
    x = range(len(Cols))
    if xs is not None: 
        x = xs 

    x = [x_i + x_nu for x_i in x]
    
    #plotting the first count row 
    r = 0
    y_bottom = df_count.iloc[r].values
    plt.bar(x, 
            y_bottom,
            color = Colors[r], 
            label = Labels[r], 
            width = bar_width)
    
    #plotting in order the subsequent count rows
    for r in range(len(df_count))[1:]:
        
        y_up = df_count.iloc[r]
        
        plt.bar(x, 
                y_up, 
                bottom = y_bottom, 
                color  = Colors[r],
                label  = Labels[r], 
                width  = bar_width)

        y_bottom += y_up

    if legend: 
        plt.legend(loc = legend_loc, fontsize = 15)
    if log_scale:
        plt.yscale('log')

def convert_num_colors(numbers):
    # Choose a matplotlib colormap (e.g., 'viridis', 'plasma', 'inferno', etc.)
    cmap = plt.get_cmap('viridis')
    # Normalize the data to range between 0 and 1
    from matplotlib.colors import LogNorm
    norm = Normalize(vmin=min(numbers), vmax=max(numbers))
    # Map the normalized data to colors using the colormap
    colors = cmap(norm(numbers))

    # Create a colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    return colors, sm

def gen_colors(num_colors):
    # Generate a color map with the specified number of colors
    cmap = plt.get_cmap('tab20')  # You can use 'tab10', 'tab20', 'Set3', or any other distinct colormap
    colors = [cmap(i) for i in np.linspace(0, 1, num_colors)]
    
    return colors

def map_nc(numbers, log = False, mm_set = None): 
    
    import matplotlib.colors as mcolors
    # Create a colormap (e.g., 'viridis')
    cmap = plt.get_cmap('viridis')
    # Normalize the numbers to a range between 0 and 1
    mm = (min(numbers), max(numbers))
    if mm_set is not None:
        mm = mm_set
                    
    norm = plt.Normalize(mm[0], mm[1])
    if log: 
        norm = mcolors.LogNorm(vmin = mm[0], 
                               vmax = mm[1])
        
    # Map the numbers to colors using the colormap
    colors = cmap(norm(numbers))
    # Create a colorbar with the original range of values
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Required for ScalarMappable
    
    return colors, sm


############################################
##### CONTACT -EPIDEMIC VISUALIZATION ######
############################################


#Define the function for visualizing each daily contact using ax object
def visual_contact_daily(ax, df_cm0_, USERS_select, cbar=True):
    df_cm0 = df_cm0_.pivot(index = 'u1', columns = 'u2', values = 'n_minutes').fillna(0).reindex(USERS_select,columns = USERS_select).fillna(0)
    #df_cm0+=1e-6
    cax = ax.matshow(df_cm0.values, 
                     cmap=cm.jet, 
                     #vmin=0, vmax=3600, 
                     norm=LogNorm(vmin=1, vmax=3600),
                     origin='lower', alpha=1.0)
    if cbar:
        cbar = plt.colorbar(cax, ax=ax, orientation='vertical', fraction=0.046, pad=0.1)
        #cbar.set_ticks(np.arange(1, 3601, 120))
        cbar.set_ticks([1, 60, 300, 600, 1800, 3600])  # Adjust tick values as needed
        cbar.set_ticklabels([1, 60, 300, 600, 1800, 3600]) 
        cbar.set_label('n_minutes', fontsize=  15)


def visual_epid_count(ax, ts_SI, Dates, USERS_select):
    ax.plot(Dates, ts_SI[:, 0][:-1], label='S', color='green')
    ax.plot(Dates, ts_SI[:, 1][:-1], label='I', color='red')
    ax.set_xticks(Dates)
    ax.set_xticklabels(Dates, rotation=90)
    ax.set_ylim(0, len(USERS_select))
    ax.legend()

def visual_R0(ax,ts_SI, Dates):
    ax.plot(Dates, ts_SI[:, 2][:-1], color='blue')
    ax.set_xticks(Dates)
    ax.set_xticklabels(Dates, rotation=90)

def visual_shaded_area(ax, x, x_minus, x_plus, color = 'blue', label = '', fill_btw = True, visual_line = True):
    if visual_line:
        ax.plot(x, color=color, label = label)
    if fill_btw:
        ax.fill_between(range(len(x)), x_minus, x_plus, color=color, alpha=0.2)

def visual_curves_SI(ax, COLLECT_simulations, USERS_select, Dates):
    q25, q50, q75 = np.percentile(COLLECT_simulations, [25, 50, 75], axis=0, method = 'nearest')
    visual_shaded_area(ax, q50[:,0], q25[:,0], q75[:,0], color = 'green', label ='S')
    visual_shaded_area(ax, q50[:,1], q25[:,1], q75[:,1], color = 'red', label ='I')
    ax.set_xticks(range(len(Dates)), Dates)
    ax.set_xticklabels(Dates, rotation=90)
    ax.set_ylim(0, len(USERS_select))    
    ax.set_ylabel('user count')
    ax.legend()
    
def visual_curves_R0(ax, COLLECT_simulations, Dates):
    q25, q50, q75 = np.percentile(COLLECT_simulations, [25, 50, 75], axis=0, method = 'nearest')
    visual_shaded_area(ax, q50[:,2], q25[:,2], q75[:,2], color = 'blue', label ='R0')
    ax.set_xticks(range(len(Dates)), Dates)
    ax.set_xticklabels(Dates, rotation=90)
    ax.legend()

def visual_epid_simulation(axes, COLLECT_simulations, Dates):
    '''
    visualize iterated epidemiological simulation
    '''    
    visual_curves_SI(axes[0], COLLECT_simulations, Dates)
    visual_curves_R0(axes[1], COLLECT_simulations, Dates)


############################################
##### GENERAL VISUALIZATION FUNCTIONS ######
############################################

def ax_legend_level(ax, 
                    loc = 'lower left', 
                    fontsize = 10,
                    include_complete = True):
    
    Classes = ['ground truth'] + [convert_to_percent_range(str(l)) for l in Levels]       
    Colors  = ['blue']     + [DICT_colors_level[l] for l in Levels]

    if not include_complete:
        Classes = Classes[1:]
        Colors = Colors[1:]    
    
    DICT_legend = gen_DICT_ax_visual('legend')
    DICT_legend.update({'classes': Classes, 
                        'colors': Colors, 
                        'loc': loc, 
                        'fontsize':10})
    
    ax_visual_legend(ax, DICT_legend)

def ax_legend_sparsification(ax, 
                             loc = 'upper right', 
                             fontsize = 10, 
                             include_complete = True):

    Classes = ['ground truth'] + [DICT_rename_ss_brief[s] for s in List_ss] 
    Colors  = ['blue']     + [DICT_colors_ss[s] for s in List_ss]      
    if not include_complete:
        Classes = Classes[1:]
        Colors = Colors[1:]  
        
    DICT_legend = gen_DICT_ax_visual('legend')
    DICT_legend.update({'classes': Classes,
                        'colors': Colors, 
                        'title': '', 
                        'loc': loc, 
                        'fontsize':fontsize})
    ax_visual_legend(ax, DICT_legend)

def ax_legend_emv(ax, 
                  EMVs, 
                  loc = 'upper right', 
                  fontsize = 10, 
                  title_fontsize = 10,
                  title= '',
                  include_complete = True):

    Classes = ['ground truth'] +  [DICT_rename_EMVs[s] for s in EMVs]
    Colors  = ['blue'] +      [DICT_colors_emv[s] for s in EMVs]  
    if not include_complete:
        Classes = Classes[1:]
        Colors = Colors[1:]  
                                 
    DICT_legend = gen_DICT_ax_visual('legend')
    DICT_legend.update({'classes': Classes,
                        'colors': Colors, 
                        'title': title, 
                        'loc': loc, 
                        'fontsize':fontsize,
                        'title_fontsize': title_fontsize})
    ax_visual_legend(ax, DICT_legend)

def ax_visual_line_legend(ax, DICT_legend):
    
    Colors = DICT_legend['colors']
    Indicators = DICT_legend['classes']
    LineStyles = DICT_legend.get('linestyles', ['solid'] * len(Colors))  # Default to solid if not specified

    Patches = [mlines.Line2D([], [], color=c, linestyle=ls, label=l, linewidth=2) 
               for c, l, ls in zip(Colors, Indicators, LineStyles)]

    ax.legend(handles=Patches, 
              title=DICT_legend['title'],
              loc=DICT_legend['loc'], 
              fontsize=DICT_legend['fontsize'],
              title_fontsize=DICT_legend['title_fontsize'])

def scatter_df(ax, 
               df, x, y, 
               title = '', 
               s = 1,
               c = 'blue',
               x_rename = None, 
               y_rename = None, 
               label_size =  15,
               title_size = 20,
               cmap = None,
               vmin = None,
               vmax = None, 
               colorbar = False, 
               colorbar_label = ''): 
    
    X = df[x].values
    Y = df[y].values
    
    sc = ax.scatter(X, Y,  
                    c = c, 
                    s = s, 
                    cmap = cmap,
                    vmin = vmin, 
                    vmax = vmax)

    if colorbar:
        cbar = plt.colorbar(sc, ax = ax)
        cbar.set_label(colorbar_label)
        
    if x_rename is None:
        x_rename = x
    if y_rename is None:
        y_rename = y

    ax_visual_labeltitles(ax, {'xlabel': x_rename, 
                               'ylabel': y_rename,
                               'title': title, 
                               'label_size': label_size, 
                               'title_size': title_size})
    

def rescale_ax_ticks(ax, 
                     int_scale = 4, 
                     digit_round = 0, 
                     scient_not_drop = False, 
                     axis = 'x'):
    '''
    scient_not_drop: drop scientific notation if already rescaled and put it into 
    else rescale the ticks accordingly
    '''

    if scient_not_drop:
        if axis=='y':
            ax.yaxis.offsetText.set_visible(False)
            ylabel = ax.get_ylabel()
            ax.set_ylabel(f"{ylabel} $(x10^{int_scale})$")
        if axis=='x':
            ax.xaxis.offsetText.set_visible(False)
            xlabel = ax.get_xlabel()
            ax.set_xlabel(f"{xlabel} $(x10^{int_scale})$")
        
    else:
        if axis=='y':
            yticks = ax.get_yticks()
            yticks_rescaled = [f'{np.round(y/(10**int_scale),digit_round)}' for y in yticks]
            ax.set_yticklabels(yticks_rescaled)
            ylabel = ax.get_ylabel()
            ax.set_ylabel(f"{ylabel} $(x10^{int_scale})$")
        if axis=='x': 
            xticks = ax.get_xticks()
            xticks_rescaled = [f'{np.round(x/(10**int_scale),digit_round)}' for x in xticks]
            ax.set_xticklabels(xticks_rescaled)
            xlabel = ax.get_xlabel()
            ax.set_xlabel(f"{xlabel} $(x10^{int_scale})$")


#Visualize scatterplot and boxplot together
def viz_scatter_boxplot(ax,
                        List_Ms, 
                        Ms_ind,
                        Colors, 
                        Colors_scatter = 'black',
                        vmin = 0, 
                        vmax = 15, 
                        cmap = 'viridis',
                        cbar = True,
                        cbar_label = '',
                        scatter_size = 1):
    '''
    Visualize the boxplots of a list of values in List_Ms corresponding to indexes Ms_ind,
    Colors are the colors of the boxplots
    Colors_scatter are the colors of the scatter points (can be also a list)
        cmap : of the Colors_scatter
    '''

    for i, c, Ms in zip(Ms_ind, Colors, List_Ms):
        
        ax.boxplot(Ms, 
                   positions = [i], 
                   widths = 0.5, 
                   showfliers = False, 
                   patch_artist=True,
                   boxprops=dict(facecolor= 'none', color= c),
                   capprops=dict(color=c),
                   whiskerprops=dict(color=c),
                   flierprops=dict(color=c, markeredgecolor=c),
                   medianprops=dict(color=c))

        X_vals = [i + np.random.normal(scale = 0.05) for j in range(len(Ms))]
        if len(Colors_scatter)>1:
            IND = np.argwhere(Ms_ind==i)[0][0]
            sc = ax.scatter(X_vals, 
                            Ms, 
                            c = Colors_scatter[IND],
                            vmin = vmin, vmax = vmax, cmap = cmap, 
                            s = scatter_size)
            if cbar and IND==0:
                cbar = plt.colorbar(sc, ax=ax)
                cbar.set_label(cbar_label, size=14)
            
        else:
            ax.scatter(X_vals, 
                       Ms, 
                       color = Colors_scatter, 
                       s = scatter_size)

def change_first_xtick(ax, newlabel='newlabel'):
    xticks = ax.get_xticks()  # Get current x-tick positions
    xticklabels = ax.get_xticklabels()  # Get current x-tick labels
    ax.set_xticks(xticks)  # Update x-ticks
    xticklabels[0] = newlabel  # Modify first tick label
    ax.set_xticklabels(xticklabels)  # Update x-tick labels

#Visualize scatterplot and boxplot together
def viz_scatter_boxplot_new(ax,
                            List_Ms, 
                            Ms_ind,
                            Colors = 'black', 
                            Colors_scatter = 'black',
                            scatter = True,
                            scatter_size = 1, 
                            x_Labels = None, 
                            x_label_size = 0,
                            x_label_rot = 0):
    '''
    Visualize the boxplots of a list of values in List_Ms corresponding to indexes Ms_ind
    Colors are the colors of the boxplots
    Colors_scatter are the colors of the scatter points (can be also a list)
        cmap : of the Colors_scatter
    '''
    
    if not isinstance(Colors, list):
        Colors = [Colors] * len(List_Ms)
    if not isinstance(Colors_scatter, list):
        Colors_scatter = [Colors_scatter] * len(List_Ms)

    for i, Ms, c, c_scatter in zip(Ms_ind, 
                                   List_Ms, 
                                   Colors, 
                                   Colors_scatter):
        
        ax.boxplot(Ms, 
                   positions = [i], 
                   widths = 0.5, 
                   showfliers = False, 
                   patch_artist = True,
                   boxprops = dict(facecolor = 'none', color = c),
                   capprops = dict(color=c),
                   whiskerprops = dict(color=c),
                   flierprops = dict(color=c, markeredgecolor=c),
                   medianprops = dict(color=c))

        #add scatter points within the boxplot
        if scatter:
            X_vals = [i + np.random.normal(scale = 0.05) for j in range(len(Ms))]
            ax.scatter(X_vals, 
                       Ms, 
                       color = c_scatter, 
                       s = scatter_size)
            
    if x_Labels is not None: 
        ax_visual_ticklabel(ax, {'t': Ms_ind, 
                                 'tl': x_Labels,
                                 'rot': x_label_rot,
                                 'size': x_label_size}, axis = 'x')

def viz_bar_series(ax, 
                   s, 
                   s_ind = None, 
                   Colors = 'black',
                   x_Labels= None, 
                   x_label_rot=0,
                   x_label_size=10):
    '''
    visual a series values with an ax.bar plot with customized colors and labeling
    s: series
    s_ind : x-ticks of bar plots
    Colors : can be also a list equal to the length of the series
    '''

    if not isinstance(Colors, list):
        Colors = [Colors] * len(s)
        
    if x_Labels is not None:
        s = s.loc[x_Labels]
    else:
        x_Labels = s.index
        
    if s_ind is None:
        s_ind = range(len(s))
        
    for i,val,c in zip(s_ind, s.values, Colors):
        ax.bar([i], val, color = c)
        
    ax_visual_ticklabel(ax, {'t': s_ind, 
                             'tl': x_Labels,
                             'rot': x_label_rot,
                             'size': x_label_size}, axis = 'x')



def custom_boxplot_from_stats(ax, 
                              stats, 
                              positions=None, 
                              color='black', 
                              width=.8, 
                              linewidth=1,
                              face_alpha=1,
                              median_color='black',
                              bar_as_median=False,
                              bar_errorbar = True, 
                              capsize = 1):
    """
    Draw a boxplot (default) or a barplot-with-errorbars (if bar_as_median=True)
    given precomputed stats (dicts with q1, q3, med, whislo, whishi).

    color: str or list[str] — single color for all boxes or one per box.
    median_color: str — color for median lines (default 'black').
    bar_as_median: bool — if True, plot median as bar height with CI error bars instead of a box.
    """
    
    n = len(stats)
    if positions is None:
        positions = list(range(1, n+1))

    # normalize color input
    if isinstance(color, (list, tuple)):
        if len(color) != n:
            raise ValueError("If 'color' is a list, its length must equal len(stats).")
        colors = list(color)
    else:
        colors = [color] * n

    if bar_as_median:
        # Draw bars at median with error bars from whislo/whishi
        medians = [s['med'] for s in stats]
        err_low = [s['med'] - s['whislo'] for s in stats]
        err_high = [s['whishi'] - s['med'] for s in stats]
        ax.bar(positions, medians,
               color=colors,
               alpha=face_alpha,
               width=width,
               edgecolor='none',
               zorder=2)
        if bar_errorbar:
            ax.errorbar(positions, medians,
                        yerr=[err_low, err_high],
                        fmt='none',
                        ecolor=median_color,
                        elinewidth=linewidth,
                        capsize=capsize,
                        zorder=3)
        
        return None  # nothing to return in this mode

    # --- default boxplot branch ---
    bp = ax.bxp(stats,
                positions=positions,
                widths=width,
                patch_artist=True,
                showfliers=False)

    for i in range(n):
        c = colors[i]
        bp['boxes'][i].set(facecolor=c, edgecolor=c, linewidth=linewidth, alpha=face_alpha)
        bp['medians'][i].set(color=median_color, linewidth=linewidth)
        #if i < len(bp['fliers']):
        #    bp['fliers'][i].set(marker='o', color=c, alpha=0.7)

        wi0, wi1 = 2*i, 2*i + 1
        bp['whiskers'][wi0].set(color=c, linewidth=linewidth)
        bp['whiskers'][wi1].set(color=c, linewidth=linewidth)
        bp['caps'][wi0].set(color=c, linewidth=linewidth)
        bp['caps'][wi1].set(color=c, linewidth=linewidth)

    return bp

def caret_marker(direction = "up", 
                 width = 1.0, 
                 height = 1.0):
    """
    Symmetric triangular caret centered at (0,0).
    direction: "up" or "down"
    width, height: relative shape proportions.
    """
    w = 0.9 * width
    h = 0.9 * height
    verts = np.array([
        [ 0.0,  h/2],   # tip
        [-w/2, -h/2],   # left base
        [ w/2, -h/2],   # right base
        [ 0.0,  0.0],   # ignored for CLOSEPOLY
    ])
    if direction == "down":
        verts[:, 1] *= -1
    codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
    return MarkerStyle(Path(verts, codes))

def visual_metric(ax, 
                  df_stats, 
                  metric, 
                  List_ss,
                  X,
                  *,
                  gt_tick = [0],
                  DICT_colors_ss = config.DICT_colors_ss,
                  COLOR_GT_std = 'black', #config.COLOR_GT,
                  col_gt = 'Complete',
                  visual_bxp_groundtruth=True,
                  visual_std_groundtruth=True,
                  visual_mean=True,
                  visual_mean_std=False,
                  visual_bxp=True,
                  set_yax_percent=False,
                  # --- NEW ---
                  color_mean_std='black',          # can be str or list per-level
                  mean_marker_size=5,
                  cap_std_marker_size=8,
                  err_linewidth=1.2, 
                  marker_std_up = caret_marker('up'),
                  marker_std_down = caret_marker('down'),
                  xtl_noperc = False,
                  face_alpha = 1, 
                  linewidth =1,
                  capsize=1,
                  bar_as_median = False,
                  bar_errorbar_gt = True):
    """
    df_stats must have as index [s, l]; sparsity and sparsity level 
    when analyzing debiasing it should be [emv, l] : debiasing approach and sparsity level
    
    Visualize boxplots and/or mean (+/- std) for each sparsity model in List_ss.

    Colors:
      - DICT_colors_ss[s] can be a single color or a list of colors (len == len(Levels)).
      - color_mean_std can also be a single color or a list per-level. If None, it follows DICT_colors_ss[s].
    """
    # standard matplotlib naming of stats for IQR and CI
    bxp_stats = ['whislo', 'q1', 'med', 'q3', 'whishi']
    
    # groundtruth stats
    s_bxp_gt = [{stat: df_stats.loc[col_gt, col_gt][f'{metric}_{stat}']
                 for stat in bxp_stats}]
    s_bxp_gt[0].update({'fliers': []})
    
    s_mean_gt = [df_stats.loc[col_gt, col_gt][f'{metric}_mean']]
    s_std_gt  = [df_stats.loc[col_gt, col_gt][f'{metric}_std']]

    Levels = config.Levels
    n_levels = len(Levels)

    def _normalize_colors(c):
        """Return a list of length n_levels from a color or color list."""
        if isinstance(c, (list, tuple)):
            if len(c) != n_levels:
                raise ValueError(f"Color list must match len(Levels)={n_levels}")
            return list(c)
        else:
            return [c] * n_levels

    for j, s in enumerate(List_ss):
        print(s)
        # base color(s) for this sparsity model
        base_color = DICT_colors_ss[s]
        print(base_color)
        colors_levels = _normalize_colors(base_color)

        # mean/std colors per level (follow base if None; else normalize)
        if color_mean_std is None:
            ms_colors = colors_levels
        else:
            ms_colors = _normalize_colors(color_mean_std)

        # [0] gather stats per level
        s_bxp = [{k: df_stats.loc[s, str(l)][f'{metric}_{k}'] for k in bxp_stats}
                 for l in Levels]
        for d in s_bxp:
            d.update({'fliers': []})

        # [1] boxplots (support per-level colors)
        if visual_bxp:
            custom_boxplot_from_stats(ax,
                                      s_bxp,
                                      positions=X + j,
                                      color=colors_levels, 
                                      face_alpha = face_alpha, 
                                      bar_as_median = bar_as_median,
                                      capsize=capsize, 
                                      linewidth = linewidth) # <-- list or single OK
            
            # median ground-truth line
            gt_median = df_stats.loc[col_gt, col_gt][f'{metric}_med']
            ax.axhline(gt_median, color=config.COLOR_GT, linewidth=.5)

        if visual_bxp_groundtruth:
            custom_boxplot_from_stats(ax,
                                      s_bxp_gt,
                                      positions=[0],
                                      color=config.COLOR_GT,
                                      capsize=capsize, 
                                      linewidth = linewidth,
                                      bar_as_median = bar_as_median,
                                      bar_errorbar = bar_errorbar_gt)

        # [2] mean and std
        if visual_mean or visual_mean_std:
            s_mean = [df_stats.loc[s, str(l)][f'{metric}_mean'] for l in Levels]
            s_std  = [df_stats.loc[s, str(l)][f'{metric}_std']  for l in Levels]
            x_pos  = np.full(n_levels, X + j, dtype=float)

        
        
        if visual_mean:
            # per-level mean markers
            for xi, yi, c in zip(x_pos, s_mean, ms_colors):
                ax.scatter(xi, yi, color = c, s = mean_marker_size, marker='o', zorder=3)

            # ground truth mean
            ax.scatter(gt_tick, s_mean_gt, color=COLOR_GT_std, s=mean_marker_size, marker='o', zorder=3)

            if visual_std_groundtruth:
                # GT std line + caret caps
                ax.errorbar(gt_tick,
                            s_mean_gt,
                            yerr=s_std_gt,
                            fmt='none', ecolor=COLOR_GT_std,
                            elinewidth = err_linewidth, capsize=0, zorder=2)
                
                ax.scatter(gt_tick, [s_mean_gt[0] + s_std_gt[0]],
                           marker= marker_std_up,   color=COLOR_GT_std,
                           s=cap_std_marker_size, zorder=4)
                ax.scatter(gt_tick, [s_mean_gt[0] - s_std_gt[0]],
                           marker= marker_std_down, color=COLOR_GT_std,
                           s=cap_std_marker_size, zorder=4)

        if visual_mean_std:
            # loop per level so each errorbar uses its own color
            for xi, yi, si, c in zip(x_pos, s_mean, s_std, ms_colors):
                ax.errorbar([xi], [yi], yerr=[si],
                            fmt='none', ecolor=c,
                            elinewidth=err_linewidth, capsize=0, zorder=2)
                ax.scatter(xi, yi + si, marker = marker_std_up,
                           color=c, s=cap_std_marker_size, zorder=4)
                ax.scatter(xi, yi - si, marker = marker_std_down,
                           color=c, s=cap_std_marker_size, zorder=4)

    # x-ticks
    DICT_xtl = gen_DICT_ax_visual('label_ticks')
    DICT_xtl['t']  = [0] + list(X)
    xtl = ['0-5%'] + config.Levels_str
    
    if xtl_noperc:
        xtl = [x[:-1] for x in xtl]
        
    DICT_xtl['tl'] = xtl
    DICT_xtl['rot'] = 45
    ax_visual_ticklabel(ax, DICT_xtl, axis='x')

    # y ticks as percents
    if set_yax_percent:
        DICT_ytl = gen_DICT_ax_visual('label_ticks')
        yticks = ax.get_yticks()
        yticks = yticks[(yticks >= 0) & (yticks <= 1)]
        DICT_ytl['t']  = yticks
        DICT_ytl['tl'] = (yticks * 100).astype(int)
        DICT_ytl['rot'] = 0
        ax_visual_ticklabel(ax, DICT_ytl, axis='y')


#######################
#### AX RESTYLING #####
#######################



def restyle_ax(ax,
               title_size =        config.ax_title_size,
               label_size =        config.ax_label_size,
               text_size  =        config.ax_text_size,
               tick_size  =        config.ax_tick_size,
               legend_font_size  = config.ax_legend_font_size,
               legend_title_size = config.ax_legend_title_size,               
               font_family='Liberation Sans', 
               other_text = True):
    """
    Apply consistent font sizes and font family to ticks, labels, title, legend, and texts in an Axes.
    """

    # Axis labels
    ax.xaxis.label.set_size(label_size)
    ax.xaxis.label.set_family(font_family)
    ax.yaxis.label.set_size(label_size)
    ax.yaxis.label.set_family(font_family)

    # Title
    ax.title.set_size(title_size)
    ax.title.set_family(font_family)

    # Tick labels
    ax.tick_params(axis='both', which='major', labelsize=tick_size)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontsize(tick_size)
        tick.set_family(font_family)

    # Legend
    legend = ax.get_legend()
    if legend:
        legend.get_title().set_fontsize(legend_title_size)
        legend.get_title().set_family(font_family)
        for text in legend.get_texts():
            text.set_fontsize(legend_font_size)
            text.set_family(font_family)
    if other_text:
        # Other text
        for text in ax.texts:
            text.set_fontsize(text_size)
            text.set_family(font_family)

#####################
#### RIDGEPLOTS #####
#####################


#histogram creation function
def compute_discrete_ridge_data(
    df,
    by,
    column,
    bins,
    normalize = 'max'):
    
    groups = list(df[by].dropna().unique())
    H = []
    for g in groups:
        h, _ = np.histogram(df.loc[df[by] == g, column], bins=bins)
        H.append(h.astype(float))
        
    H = np.vstack(H)
    widths = np.diff(bins)
    
    if normalize == 'density':
        H = (H / H.sum(1, keepdims=True)) / widths
    elif normalize == 'max':
        m = H.max()
        if m > 0:
            H = H / m
    x = (bins[:-1] + bins[1:]) / 2
    
    return x, H, groups

def plot_discrete_ridges(ax,
                         x,
                         H,
                         groups = None,
                         overlap = 1.4,
                         alpha = 0.55,
                         color = cm.Blues,        # can be str, list, or colormap
                         linewidth = 1.0,
                         show_labels = True,
                         mask = None,             # boolean list/array of len(groups)
                         x_label = 0,
                         labels = None,           # NEW: list/array of custom labels per group
                         label_rotation = 0,      # NEW: rotation (deg)
                         label_size = 10):        # NEW: fontsize

    spacing = 1.0 / overlap

    if groups is None:
        groups = list(range(len(H)))

    if mask is None:
        mask = [True] * len(groups)
    elif len(mask) != len(groups):
        raise ValueError("`mask` must have same length as groups")

    if isinstance(color, str):
        colors = [color] * len(groups)
    elif hasattr(color, "__call__"):
        colors = [color(i / max(1, len(groups)-1)) for i in range(len(groups))]
    elif isinstance(color, (list, tuple)) and len(color) == len(groups):
        colors = color
    else:
        raise ValueError("`color` must be a string, colormap, or list of len(groups).")

    n = len(groups)
    for idx in range(n - 1, -1, -1):
        g    = groups[idx]
        h    = H[idx]
        c    = colors[idx]
        keep = mask[idx]
        y0 = idx * spacing
        ridge_alpha = alpha if keep else 0.0

        x = np.asarray(x)
        w = np.median(np.diff(x)) if len(x) > 1 else 1.0
        edges = np.concatenate([x - w/2, [x[-1] + w/2]])
        h_ext = np.concatenate([h, [h[-1]]])

        if keep:
            ax.axhline(y0, color='black', lw=0.5, zorder=0)

        ax.fill_between(
            edges, y0, y0 + h_ext,
            step='post', alpha=ridge_alpha,
            facecolor=c, edgecolor='black', zorder=1
        )

        ax.plot(
            edges, y0 + h_ext,
            drawstyle='steps-post', color='black',
            lw=linewidth, alpha=ridge_alpha, zorder=2
        )

        # --- generalized label block ---
        if keep and show_labels:
            txt = (labels[idx] if (labels is not None and len(labels) == n)
                   else str(g))
            ax.text(
                x_label, y0, txt,
                va='center', ha='right',
                rotation=label_rotation,
                fontsize=label_size,
                zorder=3
            )

def discrete_ridge_hist(
    ax,
    df,
    by,
    column,
    bins,
    overlap=1.4,
    alpha=0.55,
    color=cm.Blues,
    normalize='max',
    linewidth=1.0,
    show_labels = False,
    x_label = 0,
    labels = None,           # NEW: list/array of custom labels per group
    label_rotation = 0,      # NEW: rotation (deg)
    label_size = 10):        # NEW: fontsize

    x, H, groups = compute_discrete_ridge_data(
        df=df,
        by=by,
        column=column,
        bins=bins,
        normalize=normalize
    )
    plot_discrete_ridges(
        ax=ax,
        x=x,
        H=H,
        groups=groups,
        overlap=overlap,
        alpha=alpha,
        color=color,
        linewidth=linewidth,
        show_labels=show_labels,
        x_label = x_label,
        labels = labels,           # NEW: list/array of custom labels per group
        label_rotation = label_rotation,      # NEW: rotation (deg)
        label_size = label_size)        # NEW: fontsize
    
def plot_level_ridges(
    df_freq,
    ax,
    l,
    ss=None,                    # list of scenario keys (defaults to global _ss if present)
    x=None,                     # x positions; defaults to range over df_freq columns
    overlap=4,
    alpha=0.8,
    color=None,
    linewidth=1,
    show_labels=False,
    x_label=0,
    mask=None,
    labels=None,                # custom labels per ridge (len == len(ss))
    label_rotation=0,
    label_size=10
):
    if ss is None:
        ss = _ss  # fallback to existing global if user keeps it
    if x is None:
        x = range(df_freq.shape[1])

    H_l = [df_freq.loc[(s, str(l))].values for s in ss]

    plot_discrete_ridges(
        ax=ax,
        x=x,
        H=H_l,
        groups=ss,
        overlap=overlap,
        alpha=alpha,
        color=color,
        linewidth=linewidth,
        show_labels=show_labels,
        x_label=x_label,
        mask=mask,
        labels=labels,
        label_rotation=label_rotation,
        label_size=label_size
    )

def plot_missingness_ridges(
    df_freq,
    ax,
    s,
    levels=None,                # list/iterable of Levels (defaults to global Levels if present)
    x=None,
    overlap=4,
    alpha=0.8,
    color=None,
    linewidth=1,
    show_labels=False,
    x_label=0,
    mask=None,
    labels=None,                # custom labels per level (len == len(levels))
    label_rotation=0,
    label_size=10):
    
    if levels is None:
        levels = config.Levels  # fallback to existing global if user keeps it
    if x is None:
        x = range(df_freq.shape[1])

    H_l = [df_freq.loc[(s, str(l))].values for l in levels]

    plot_discrete_ridges(
        ax=ax,
        x=x,
        H=H_l,
        groups=[str(l) for l in levels],
        overlap=overlap,
        alpha=alpha,
        color=color,
        linewidth=linewidth,
        show_labels=show_labels,
        x_label=x_label,
        mask=mask,
        labels=labels,
        label_rotation=label_rotation,
        label_size=label_size
    )


def plot_groundtruth_ridges(
    df_freq,
    ax,
    groundtruth=('Complete', 'Complete'),
    k=3,                        # number of repeated ridges (ignored if mask is given)
    x=None,
    overlap=4,
    alpha=0.8,
    color=None,
    linewidth=1,
    show_labels=False,
    x_label=0,
    mask=None,                  # e.g., [True, False, False]
    labels=None,                # custom labels per repeated ridge (len == len(mask or k))
    label_rotation=0,
    label_size=10
):
    if x is None:
        x = range(df_freq.shape[1])

    v = df_freq.loc[groundtruth].values

    # Determine how many layers to plot
    if mask is not None:
        k_eff = len(mask)
    else:
        k_eff = k
        mask = [True] + [False] * (k_eff - 1)  # default: show only first ridge

    H_gt = [v] * k_eff
    groups = [f"GT{i+1}" for i in range(k_eff)]  # placeholder group names unless labels provided

    plot_discrete_ridges(
        ax=ax,
        x=x,
        H=H_gt,
        groups=groups,
        overlap=overlap,
        alpha=alpha,
        color=color,
        linewidth=linewidth,
        show_labels=show_labels,
        x_label=x_label,
        mask=mask,
        labels=labels,
        label_rotation=label_rotation,
        label_size=label_size
    )


def annotate_axes(
    axes,
    texts=None,
    dates=None,
    xs=None,
    tick_size = .7,
    y_tick_height = -0.1,
    y_tick_height_last_axis = -0.1,
    y_date_labels = 0):
    """
    Annotate axes with optional category labels on the left
    and date ticks/labels on the bottom axis.

    Parameters
    ----------
    axes : list of Axes
        Axes objects to annotate.
    texts : list of str or None, optional
        Labels for each subplot (reversed order is used).
        If None, no text is drawn.
    dates : sequence of datetime, optional
        Full sequence of dates for labeling.
    xs : list of int, optional
        Positions in the date sequence to mark and label.
    """
    
    if xs is None:
        xs = []
    xs_dates = []
    if dates is not None and xs:
        xs_dates = [dates[i-1].strftime("%-d %b") for i in xs]

    # category labels on the left
    if texts is not None:
        for ax, t in zip(axes, texts[::-1]):
            ax.text(
                1, .6, t,
                transform=ax.get_xaxis_transform()
            )

    #xlimits
    for ax in axes:
        ax.set_xlim(1, 27.5)

    # tick marks
    for ax in axes[:-1]:
        if xs:
            ax.plot(
                xs, [y_tick_height]*len(xs),
                linestyle='none', marker='|',
                markersize= tick_size, color='black',
                zorder = 999,
                clip_on=False
            )

    # date labels only on the bottom axis
    if xs_dates:
        
        ax = axes[-1]
        ax.plot(xs, 
                [y_tick_height_last_axis]*len(xs),
                linestyle='none', 
                marker = '|',
                markersize = tick_size, 
                color = 'black',
                zorder = 999,
                clip_on = False)
        
        for x, label in zip(xs, xs_dates):
            ax.text(x, 
                    y_date_labels, 
                    label,
                    rotation = 45, 
                    fontsize = 10,
                    ha = 'right', 
                    va = 'top',
                    transform=ax.get_xaxis_transform(),
                    clip_on=False)

def panel_metric_dynamic(axes, 
                         c_freq, 
                         c,
                         _ss = config.List_ss_rename,
                         DICT_colors = config.DICT_colors_ss):
                         
    '''
    axes: axes stacked vertically
    c_freq: fequency of c
        - must have index (sparsity, sparsity_level)
    c : dynamic metric ('day_peak', 'day_last_case', 'day_last_recovery'])
    _ss : list of sparsity approaches or debiasing approaches
    '''

    overlap   = 1.5
    alpha     = 0.8
    linewidth = 1
    colors = [DICT_colors[s] for s in _ss]
    
    for s, ax in zip(_ss, axes[:-1]):
        plot_missingness_ridges(
            df_freq=c_freq,
            ax=ax,
            s=s,
            overlap=overlap,
            alpha=alpha,
            color= DICT_colors[s],
            linewidth=.01,
            show_labels = True, 
            label_rotation = 0,
            labels = ['10-20', '','','', '50-60'])
        
        ax.axis('off')
    
    ax = axes[-1]
    plot_groundtruth_ridges(
        df_freq=c_freq,
        ax=ax,
        groundtruth=('ground truth', 'ground truth'),
        overlap=overlap,
        alpha=alpha,
        color=config.COLOR_GT,
        linewidth=.01,
        mask = [False,False,False,False,True], 
        show_labels = True, 
        labels = ['','','','','0-5'])
    ax.axis('off')
    
    _xs = [1, 8, 15, 22, 27]
    annotate_axes(axes, 
                  dates = config.Dates_plus1, 
                  xs= _xs,
                  tick_size = 5,
                  y_tick_height = -.1, 
                  y_tick_height_last_axis = 2.6, 
                  y_date_labels = .7)
    
    for ax, color in zip(axes, colors):
    
        rect = patches.Rectangle((0, .05),          # bottom-left in axes coords
                                 1, .95,            # full width & height
                                 transform = ax.transAxes,  # use axes coordinates
                                 facecolor = color,
                                 alpha = .4,
                                 zorder = -1)
        ax.add_patch(rect)
    
    
    for ax in axes:
        restyle_ax(ax,
                   text_size  = config.ax_tick_size, 
                   title_size = config.ax_label_size)

####################################    
##### CONTACT VISUALIZATION ########
####################################

def draw_filled_line(
    ax,
    x,
    y,
    color,
    base=0.03,
    alpha=0.2,
    lw=1.5,
    s=10,
    marker='o',
    facecolor='none'):
    """Fill under curve to a base line, then draw line and markers."""
    
    ax.fill_between(
        x, y, base,
        color=color,
        alpha=alpha
    )

    ax.plot(
        x, y,
        color=color,
        linewidth=lw
    )

    ax.scatter(
        x, y,
        edgecolor=color,
        facecolor=facecolor,  # use facecolor argument
        s=s,
        marker=marker
        
    )

def to_12h_label(h):
    """Return '12 am/pm' style label for hour h in [0..23]."""
    if h == 0:
        return "12 am"
    if h < 12:
        return f"{h} am"
    if h == 12:
        return "12 pm"
    return f"{h-12} pm"

def make_xtick_labels(ticks, start_hour):
    """Shift ticks by start_hour and format as 12-hour labels."""
    shifted = [ (t + start_hour) % 24 for t in ticks ]
    return [to_12h_label(h) for h in shifted]


def legend_weekend_weekday(ax, 
                           loc='upper left'):
    legend_elements = [
        Line2D([0], [0],
               marker='o',
               color='black',       # line color (unused here)
               markerfacecolor='black',
               markeredgecolor='black',
               linestyle='None',
               label='weekday'),
        Line2D([0], [0],
               marker='D',
               color='black',
               markerfacecolor='none',   # hollow marker
               markeredgecolor='black',
               linestyle='None',
               label='weekend')]
    
    ax.legend(handles=legend_elements, loc = loc)


def panel_csa_level(ax, 
                    df_csa, 
                    level, 
                    contact_metric= 'share', 
                    percent_yticks = True):
    '''
    compare the csa (contact share average)
    over different sparsity approaches for a given sparsity-level
    '''

    def get_csa(df_csa, 
                l,
                wp = 'weekday',
                s = 'Data driven',
                hour_order = range(24)):
        '''
        get csa values for a specific (wp, s, l)
        hour_order : ordering of the hour probabilities
        '''
        
        df_wp = df_csa[df_csa['weekperiod']==wp]
        df_sl = df_wp[(df_wp['sparsity']==s) & (df_wp['sparsity_level'] == str(l))]
    
        df_sl = df_sl.set_index('hourofday').loc[hour_order]
        return df_sl[contact_metric].values
    
    start_hour = 8
    hour_order = list(range(start_hour,24)) + list(range(0,start_hour))
    
    for marker, wp in zip(['o','D'], 
                          ['weekday','weekend']):
    
        x = range(24)
        for s in config.List_ss_rename:
            y_csa = get_csa(df_csa, 
                            l = level,
                            wp = wp, 
                            s = s,
                            hour_order= hour_order)
    
            c = config.DICT_colors_ss[s]
            draw_filled_line(ax, 
                             x, y_csa, 
                             color = c ,
                             base=0, alpha= 0.1, lw=.5, s=20, 
                             marker= marker, 
                             facecolor = c if (wp =='weekday') else 'none')
    
    _xticks = [0, 6, 12, 18, 23]
    _xticks_labels = make_xtick_labels(_xticks, start_hour)
    ax.set_xticks(_xticks)
    ax.set_xticklabels(_xticks_labels, rotation=45)
    
    # titles / labels
    ax.set_xlabel("hour of day")
    if percent_yticks: 
        set_percent_yticks(ax, decimals=0)

    ax.grid(axis="x", linestyle="-", color="gray", alpha=0.7)


from sklearn.metrics import r2_score

def rsq(df, col1, col2, corr_type = 'pearson_r2'):
    if corr_type =='pearson_r2':
        return (df[col1].corr(df[col2]))
    else:
        return r2_score(df[col1], df[col2])



def _binned_percentile_yvals(df, 
                             x_col, y_col, 
                             x_bins, 
                             percentile,
                             use_weight=True, 
                             weight_col="weight"):
    """
    Return a 1D float array of length len(x_bins) with NaN at the left edge.
    """
    df = df[[x_col, y_col] + ([weight_col] if use_weight else [])].dropna()
    x_idx = np.digitize(df[x_col].to_numpy(), x_bins)

    def _to_scalar(q):
        # coerce np.array / np.ndarray / pandas scalar to Python float or np.nan
        if q is None:
            return np.nan
        q_arr = np.asarray(q, dtype=float).reshape(-1)
        return q_arr[0] if q_arr.size else np.nan

    q_by_xbin = {}
    p = float(percentile) / 100.0

    for i in range(1, len(x_bins)):
        mask = (x_idx == i)
        if not mask.any():
            q_by_xbin[i] = np.nan
            continue

        vals = df.loc[mask, y_col].to_numpy()
        if use_weight:
            w = df.loc[mask, weight_col].to_numpy()
            q = DescrStatsW(data=vals, weights=w).quantile(probs=p, return_pandas=False)
        else:
            q = np.nanquantile(vals, p)

        q_by_xbin[i] = _to_scalar(q)

    # prepend NaN for the left edge so indexing aligns with bins
    y_vals = [np.nan] + [q_by_xbin.get(i, np.nan) for i in range(1, len(x_bins))]
    
    return np.asarray(y_vals, dtype=float)

def plot_binned_percentile(ax,
                           _df,
                           x_col, y_col,
                           x_bins, y_bins,
                           percentile=50,
                           color='cyan',
                           linestyle='-',
                           linewidth=2,
                           scatter=False,
                           scatter_size=1,
                           use_weight = True, 
                           weight_col = "weight",
                           bin_non_linear=False,   # <--- NEW
                           no_binning = False,
                           _plot = True,
                           **kwargs):
    """
    Plot a percentile line over a binned heatmap.
    If bin_non_linear=True, y is mapped to bin coordinates using the actual (possibly non-uniform) y_bins.
    Otherwise, a linear rescale (uniform bins) is applied as before.
    """
    # 1) Get percentile y at each x-bin (in DATA units)
    y_vals = _binned_percentile_yvals(_df, x_col, y_col, x_bins, 
                                      percentile, 
                                      use_weight = use_weight, 
                                      weight_col = weight_col)
    
    if no_binning:
        ax.scatter(x_bins, y_vals, color=color, s=scatter_size)
    else:
        # 2) Map y to heatmap coordinates
        if bin_non_linear:
            y_rescaled = map_values_to_bincoords(y_vals, y_bins, fractional=True)
        else:
            y_rescaled = rescale_to_bins(y_vals, y_bins)
    
        # 3) x mapping (keep your original)
        x_rescaled = (x_bins - x_bins[0]) / (np.diff(x_bins)[1]) - 0.5
    
        # 4) draw
        if _plot:
            ax.plot(x_rescaled, y_rescaled, color=color, linestyle=linestyle, linewidth=linewidth, **kwargs)
        if scatter:
            ax.scatter(x_rescaled, y_rescaled, color=color, s=scatter_size)


def panels_missing_users_detected_contacts(axes, 
                                           df_merged, 
                                           l):
    '''
    visualize detected contacts vs. missing users
    for a given range under different sparsity approaches
    '''
    _LS = config.List_ss_rename
    x_metric= 'missing_users_perc'

    for ax, s in zip(axes, _LS):
        
        df_s = df_merged[df_merged['sparsity'] == s]
        dict_level_df_s = subset_df_feature(df_s, 'sparsity_level').copy()
    
        #setting the colorbar for the hour
        start_hour = 8
        df_sl = dict_level_df_s[str(l)].copy()
        df_sl['hourofday'] += (24 - start_hour)
        df_sl['hourofday'] %= 24
    
        scatter_df(ax, 
                   df_sl, 
                   x = x_metric, 
                   y = 'count_contacts',
                   y_rename = '', 
                   x_rename = '',
                   s = .1,
                   #c = DICT_colors_ss[s])
                   c = df_sl['hourofday'],
                   cmap = 'Blues', 
                   colorbar=False)
    
        bin_users = np.arange(.0,.9,.05) 
        bin_contacts = np.linspace(10,1e3, 20)

        plot_binned_percentile(ax,
                               df_sl,
                               x_metric, 
                               'count_contacts', 
                               bin_users, 
                               bin_contacts, 
                               percentile=50,
                               color='black',
                               linestyle = '-',
                               linewidth = 2,
                               scatter=True,
                               scatter_size = 5,
                               use_weight = False, 
                               weight_col = "weight",
                               bin_non_linear=True,   # <--- NEW
                               _plot = False, 
                               no_binning = True)
    
        
        ax.set_yscale('log')
        
        #ax.set_xlim(0.2,0.75)
        ax.set_ylim(2,1.2e3)
    
        R2 = rsq(df_sl, 
                 x_metric, 
                 'count_contacts')
        
        ax.text(0.98, 0.02,                      # (x,y) bottom-right
                fr"$\rho = {R2:.2f}$",         # latex-style rho^2
                transform=ax.transAxes,
                ha="right", va="bottom",
                fontsize=10,
            )
        ax.set_title(s)
    
        if s != _LS[-1]:
            ax.set_xlabel('')
            #remove_axis_ticktext(ax, axis= 'x')

    axes[0].set_ylabel('detected contacts')
    axes[1].set_xlabel('missing users (%)')

#########################################
#### GAP DISTRIBUTION AND ENTROPIES #####
#########################################

def panel_gap_distribution(ax, 
                           df_gap_count,
                           level= '40-50',
                           fill_alpha = 0.18,
                           hours_select = range(1,13), 
                           ax_ticks = True):
    '''
    visualize the gap duration distribution for a sample of sequences withi a range of missing hours
    df_gap_count: df of gap durations
    level: level of missing hours (from 0-10 to 90-100)
    '''

    
    
    #subset the gap count dataframe by the level of missing hours                    
    df_gap_count_level = df_gap_count[df_gap_count['missing_hours'] == level]
    
    df_gcl = df_gap_count_level.pivot(index = 'gap_duration_hours', 
                                      columns = 'sparsity', 
                                      values = 'count').fillna(0)
    
    #normalize column counts to obtain the probability for each gap duration
    df_gcl = df_gcl.div(df_gcl.sum(axis=0), axis=1)
    
    
    
    for s in ['Data driven', 'Random uniform']:
        y = df_gcl.loc[hours_select, s].values
        x = np.arange(len(y))  # 0..11
        
        # area
        ax.fill_between(
            x, 0, y,
            color = config.DICT_colors_ss[s],
            alpha = fill_alpha,
            zorder = 0
        )
    
        # line
        ax.plot(
            x, y,
            color=config.DICT_colors_ss[s])
        
        # points
        ax.scatter(
            x, y,
            color=config.DICT_colors_ss[s],
            s=10)

    if ax_ticks: 
        yticks = ax.get_yticks()
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{y*100:.0f}" for y in yticks])
        ax.set_ylim(0,.8)
        ax.set_ylim(1e-5,1.1)
        ax.set_yscale('log')   # for y-axis
        
        # choose the major tick positions
        major_ticks = [0, 5, 11]#, 17, 23]
        # set axis limits that include all major ticks
        ax.set_xlim(-.25, 11)
        # major ticks and labels
        ax.set_xticks(major_ticks)
        ax.set_xticklabels([str(t+1) for t in major_ticks])
        # minor ticks (optional): every hour
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
        # style
        ax.tick_params(axis="x", which="major", length=6, width=1)
        ax.tick_params(axis="x", which="minor", length=3, width=0.8)


def plot_kde(ax, x, color, label=None, fill_alpha=0.35, ls='-', lw=1.5, z=1):
    '''
    plot gaussian kernel density estimation for a given array
    '''
    from scipy.stats import gaussian_kde
    x = np.asarray(x.dropna())
    kde = gaussian_kde(x)
    xx = np.linspace(x.min(), x.max(), 500)
    yy = kde(xx)
    if fill_alpha > 0:
        ax.fill_between(xx, yy, 0, color=color, alpha=fill_alpha, zorder=z)
    ax.plot(xx, yy, color=color, lw=lw, ls=ls, zorder=z+1, label=label)
    return xx, yy

def panel_sequence_entropies(ax,
                             df_sequence_entropies,
                             level_missing_hours = '40-50',
                             alpha = 0.35):

    df_sequence_entropies_level = df_sequence_entropies[df_sequence_entropies['missing_hours'] == level_missing_hours]

    for s in ['Data driven', 'Random uniform']:

        plot_kde(ax,
                 df_sequence_entropies_level[s],
                 color= config.DICT_colors_ss[s],
                 fill_alpha=alpha)


################################    
##### CALIBRATION OUTCOMES #####
################################



def visual_grid_R0_global(ax, 
                          df_GRID_sub, 
                          x = 'beta', y = 'gamma',
                          cmap = 'Greys', 
                          R0_column = 'R0_global_mean',
                          levels = 20, 
                          R0_min =0, 
                          R0_max =7,
                          cbar = True, 
                          cbar_title = 'mean R0',
                          contour_lines=None,  # pass a list of R0 values for contour lines
                          contour_line_color='k',
                          contour_line_style='--', 
                          manual_clabel = False, 
                          alpha = 1, 
                          inset_kw_arg = None, 
                          symmetric_cbar = False,
                          sym_linthresh = 1e-3):

    from matplotlib.colors import Normalize, SymLogNorm
    from matplotlib.cm import ScalarMappable
    
    # ---- choose normalization ----
    if symmetric_cbar:
        vmax_sym = max(abs(R0_min), abs(R0_max))
        norm = SymLogNorm(linthresh=sym_linthresh, vmin=-vmax_sym, vmax=vmax_sym, base=10)
        max_pow = int(np.floor(np.log10(vmax_sym)))
        pos_ticks = [10**k for k in range(0, max_pow + 1)]
        neg_ticks = [-t for t in reversed(pos_ticks)]
        ticks = neg_ticks + [0] + pos_ticks
    else:
        ticks = np.arange(R0_min, R0_max + 1, 1)
        norm = Normalize(vmin=R0_min, vmax=R0_max)
        
    # Filled contour plot
    tcf = ax.tricontourf(
        df_GRID_sub[x],
        df_GRID_sub[y],
        df_GRID_sub[R0_column],
        levels=levels,
        cmap=cmap, 
        norm = norm,
        vmin=0, 
        vmax=R0_max + 1,
        alpha=alpha)
    
    # Add contour lines if requested
    if contour_lines is not None:
        contours = ax.tricontour(
            df_GRID_sub[x],
            df_GRID_sub[y],
            df_GRID_sub[R0_column],
            levels=contour_lines,
            colors=contour_line_color,
            linestyles=contour_line_style,
            linewidths=1, 
            alpha = alpha)
        
        offset_x = 0
        offset_y = 0
        if manual_clabel is not False:
            labels = ax.clabel(contours, inline=True, fontsize=10, 
                               manual = manual_clabel)
            for txt in labels:
                x, y = txt.get_position()
                txt.set_position((x + offset_x, y + offset_y))  # offset_x > 0 moves to the rightc
                #txt.set_color('white')
    
    if cbar:
        inset_kw = dict(
            width="40%",
            height="4%",
            loc='lower center',
            bbox_to_anchor=(0.1, 0.7, 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=1
        )
        if inset_kw_arg is not None:
            inset_kw.update(inset_kw_arg)
        cax = inset_axes(ax, **inset_kw)

        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        # Create horizontal colorbar using dummy ScalarMappable
        cbar = plt.colorbar(sm, cax=cax, orientation='horizontal')

            
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{t:g}" for t in ticks])
    
        cbar.ax.xaxis.set_label_position('top')
        cbar.ax.xaxis.set_ticks_position('bottom')
        cbar.set_label(
            cbar_title,
            size=12,
            labelpad=5,
            bbox=dict(facecolor='white', edgecolor='none', pad=1)
        )
    
        for tick in cbar.ax.xaxis.get_ticklabels():
            tick.set_color('white')

def visual_GRID_R0(ax, 
                  df_GRID, 
                  x = 'gamma',
                  y = 'beta',
                  R0_column = 'R0_global_mean',
                  beta_range = None,
                  gamma_range = None, 
                  grid_levels = 4, 
                  contour_lines = None, 
                  manual_clabel = False,
                  cmap = 'gist_yarg',
                  cbar = False, 
                  cbar_title = '',
                  scatter_size =1, 
                  alpha =1, 
                  R0_min = 1,
                  R0_max = 7,
                  inset_kw_arg = None, 
                  symmetric_cbar = False): 

    if beta_range is not None:
        df_GRID = df_GRID[df_GRID['beta'].between(beta_range[0], beta_range[1])]
    if gamma_range is not None:
        df_GRID = df_GRID[df_GRID['gamma'].between(gamma_range[0], gamma_range[1])]

    visual_grid_R0_global(ax, 
                          df_GRID, 
                          R0_column = R0_column,
                          levels = grid_levels,
                          R0_min = R0_min,
                          R0_max = R0_max,
                          contour_lines = contour_lines,
                          cbar = cbar,
                          cbar_title = cbar_title,
                          x = x, 
                          y = y, 
                          manual_clabel = manual_clabel,
                          cmap = cmap, 
                          alpha =alpha, 
                          inset_kw_arg = inset_kw_arg,
                          symmetric_cbar = symmetric_cbar)
    
    scatter_df(ax, 
            df_GRID, 
            x, 
            y, 
            s = scatter_size,
            c = df_GRID[R0_column].values, 
            cmap = plt.cm.gist_yarg, 
            vmin =0,
            vmax=4,
            colorbar = False,
            colorbar_label = '$\overline{R}_0$')#fontsize_cbar = 20)

def visual_pars_med_ci(ax,
                       df_pars_stats, 
                       x = 'beta',
                       y = 'gamma',
                       capsize=4,
                       elinewidth=1.2,
                       markeredgewidth=0.80):
    '''
    visualize median and 95%CI 
    of pars (x,y) for different levels of sparsity    
    '''

    df = df_pars_stats.copy()
    df = df.loc[[str(l) for l in config.Levels]]
       
    #Median values
    X = df[f'{x}_med'].values
    Y = df[f'{y}_med'].values

    #Confidence interval values
    X_wl = df[f'{x}_whislo'].values
    Y_wl = df[f'{y}_whislo'].values
    X_wh = df[f'{x}_whishi'].values
    Y_wh = df[f'{y}_whishi'].values
    
    for i, col in enumerate(config.COLORS_LEVEL):
        ax.errorbar(X[i], Y[i],
                    xerr=[[X[i] - X_wl[i]], [X_wh[i] - X[i]]],
                    yerr=[[Y[i] - Y_wl[i]], [Y_wh[i] - Y[i]]],
                    fmt= 'o',
                    color=col,
                    ecolor=col,
                    capsize= capsize,
                    elinewidth= elinewidth,
                    markeredgewidth= markeredgewidth)

def visual_gt_par(ax, 
                  groundtruth_pars, 
                  x = 'beta', 
                  y = 'gamma', 
                  s = 30):
    '''
    visualization of the ground truth parameters 
    '''
    ax.scatter(
        groundtruth_pars[x],
        groundtruth_pars[y],
        facecolors=config.COLOR_GT,
        s=s,
        marker='D',
        edgecolors=config.COLOR_GT,
        zorder=999)

def visual_fitted_params(ax, 
                         df_fpc_stats, 
                         df_R0_grid_estimates, 
                         beta_grid_range, 
                         gamma_grid_range,
                         R0_min, 
                         R0_max,
                         groundtruth_pars,
                         grid_levels,
                         List_mc): 

    '''
    df_fpc : stats of fitted parameters
    df_R0_grid_estimates: grid of esitimated R0 value
    (beta_grid_range, gamma_grid_range): select grid for the range
    (R0_min, R0_max): min, max R0 for the colorbar 
    groundtruth_pars: dictionary containing the ground truth simulation parameters
    grid_levels : number of plotted levels over the contour plot of the R0 heatmap
    List_mc : list of ax coordinates for labeling of the contour lines (has same len of grid_levels)
    '''
    #visualize median and confidence interval of the estimated parameters
    visual_pars_med_ci(ax,
                       df_fpc_stats, 
                       x = 'beta',
                       y = 'gamma',
                       capsize=4,
                       elinewidth=1.2,
                       markeredgewidth=0.80)
    
    #visualize the underlying grid of R0 values 
    visual_GRID_R0(ax, 
                  df_R0_grid_estimates, 
                  y = 'gamma',
                  x = 'beta',
                  cmap = 'Blues',
                  R0_column = 'R0_mean',
                  beta_range = beta_grid_range,
                  gamma_range = gamma_grid_range, 
                  R0_min = R0_min,
                  R0_max= R0_max,
                  grid_levels = grid_levels, 
                  contour_lines = grid_levels, 
                  manual_clabel = List_mc,
                  cbar = False, 
                  #cbar_title = 'ground truth $\overline{R}_0$',
                  scatter_size=.001)

    #visualize the ground truth parameters 
    visual_gt_par(ax, 
                  groundtruth_pars, 
                  x = 'beta', 
                  y = 'gamma', 
                  s = 30)


################################    
##### OUTCOMES FROM CUEBIQ #####
################################


def process_cov_share_data(cov_share_sw, 
                           sws = 30):
    '''
    process the data of coverage share from cuebiq for a given sliding window renaming consistently the variables
    sws: sliding window size (possible values are 7, 14, 21, 30, 50)
    '''

    cov_share_sw['DATE'] = pd.to_datetime(cov_share_sw['DATE']) 
    
    cols_old = ['DATE', 
                '0.9–1.0', '0.8–0.9', '0.7–0.8', '0.6–0.7', '0.5–0.6','0.4–0.5', '0.3–0.4', '0.2–0.3', '0.1–0.2', '0.0–0.1', 
                'WINDOW_DAYS']
    
    cols_new = ['DATE', 
                '0-10', '10-20', '20-30', '30-40', '40-50','50-60', '60-70', '70-80', '80-90', '90-100', 
                'WINDOW_DAYS']
    
    dict_rename_cols = dict(zip(cols_old, cols_new))
    cov_share_sw.columns = [dict_rename_cols[c] for c in cov_share_sw.columns]
    
    sw_sizes  = cov_share_sw['WINDOW_DAYS'].unique()
    sws_share = cov_share_sw[cov_share_sw['WINDOW_DAYS'] == sws].set_index('DATE').drop('WINDOW_DAYS', 
                                                                                        axis = 1)
    
    return sws_share

def viz_coverage(ax,
                 cov_share_sw, 
                 sws = 7, 
                 date_range = None, 
                 date_step =10):

    sws_share = cov_share_sw[cov_share_sw['WINDOW_DAYS'] == sws].set_index('DATE').drop('WINDOW_DAYS', axis = 1)
    df = sws_share.copy()
    if date_range is not None:
        df = df.loc[date_range]
    
    cmap = cm.get_cmap("coolwarm")     # blue→red
    colors = [cmap(i) for i in np.linspace(0, 1, df.shape[1])]
    
    df.plot(kind='bar',
            stacked=True,
            ax=ax,
            width=1.1,
            color=colors)
        
    df.index = pd.to_datetime(df.index)
    
    ax.set_xticks(range(0, len(df), date_step))
    ax.set_xticklabels(df.index[::date_step].strftime("%m-%d"), rotation=90)
    ax.set_title(f'{sws} days')

def convert_mmdd_to_ddmon(ax):
    """Convert xticklabels from 'MM-DD' to 'DD Mon' format."""
    labels = ax.get_xticklabels()
    new_labels = []
    
    for lbl in labels:
        txt = lbl.get_text()
        if "-" in txt:
            try:
                m, d = txt.split("-")
                date = dt.datetime(2000, int(m), int(d))   # dummy year
                new_labels.append(date.strftime("%d %b"))
            except:
                new_labels.append(txt)
        else:
            new_labels.append(txt)

    ax.set_xticklabels(new_labels)





























