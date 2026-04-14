import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from matplotlib.colors import LogNorm
import xarray as xr
from pathlib import Path

# ===== CONFIGURATION PARAMETERS =====
BACKGROUND_COLOR = 'black'
FOREGROUND_COLOR = 'white'
LAND_COLOR = '#99AD8D'
OCEAN_COLOR = "#242829" # '#8795CC'
COASTLINE_COLOR = 'black'
BORDERS_COLOR = 'black'
STATES_COLOR = 'black'
GRIDLINES_COLOR = 'white'
USE_LOG_SCALE = True
FONT_SIZE = 20
# ====================================


def plot_L3_s5p_pca(
    dataset: xr.Dataset,
    variable: str,
    title: str = None,
    use_log_scale: bool = False,
    figsize: tuple = (16, 10),
    cmap: str = 'plasma',
    add_basemap: bool = True,
    vmin: float = None,
    vmax: float = None,
    save_fig_dir: str = None,
    prefix: str = "",
):
    """
    Plot gridded L3 S5P PCA data from an xarray Dataset.
    
    Parameters:
    -----------
    dataset : xr.Dataset
        Gridded S5P PCA dataset with latitude and longitude coordinates
    variable : str
        Variable to plot (e.g., 'score_CO_1_mean', 'cloud_fraction_mean', 'score_CO_2_count')
    title : str, optional
        Custom title for the plot. If None, auto-generated based on variable name
    use_log_scale : bool, optional
        Whether to use logarithmic scale for the colorbar (default: False)
    figsize : tuple, optional
        Figure size (width, height) in inches
    cmap : str, optional
        Colormap name (default: 'YlOrRd')
    add_basemap : bool, optional
        Whether to add geographic features (land, ocean, coastlines, etc.)
    vmin : float, optional
        Minimum value for the colormap scale. If None, auto-determined from data
    vmax : float, optional
        Maximum value for the colormap scale. If None, auto-determined from data
    save_fig_dir : str, optional
        Directory path to save the figure. If None, figure is not saved.
        Filename will be auto-generated based on variable name.
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    
    variable = prefix + variable # add prefix to variable name if provided
    
    # Validate variable exists in dataset
    if variable not in dataset.data_vars:
        raise ValueError(
            f"Variable '{variable}' not found in dataset. "
            f"Available variables: {list(dataset.data_vars)}"
        )
    
    # Extract data
    data = dataset[variable]
    lats = dataset['latitude'].values
    lons = dataset['longitude'].values
    
    # Create figure and axes with projection
    fig = plt.figure(figsize=figsize, facecolor=BACKGROUND_COLOR)
    ax = plt.axes(projection=ccrs.PlateCarree(), facecolor=BACKGROUND_COLOR)
    
    # Add geographic features if requested
    if add_basemap:
        ax.add_feature(cfeature.LAND, facecolor=LAND_COLOR, alpha=0.8)
        ax.add_feature(cfeature.OCEAN, facecolor=OCEAN_COLOR, alpha=0.8)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor=COASTLINE_COLOR)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=':', edgecolor=BORDERS_COLOR)
        ax.add_feature(cfeature.STATES, linewidth=0.3, edgecolor=STATES_COLOR)
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color=GRIDLINES_COLOR, alpha=0.3, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'color': FOREGROUND_COLOR, 'size': 10}
    gl.ylabel_style = {'color': FOREGROUND_COLOR, 'size': 10}
    
    # Determine if we should use log scale based on variable type and data range
    is_count_variable = 'count' in variable.lower()
    
    # Prepare data for plotting
    plot_data = data.values.copy()
    
    # For count variables, set zero values to NaN so they appear transparent
    if is_count_variable:
        plot_data = plot_data.astype(float)
        plot_data[plot_data <= 0] = np.nan
    
    # Handle NaN values - they will be transparent in the plot
    valid_data = plot_data[~np.isnan(plot_data)]
    
    if len(valid_data) == 0:
        print(f"Warning: No valid data to plot for variable '{variable}'")
        return fig, ax
    
    # Determine normalization
    if use_log_scale and not is_count_variable:
        # For log scale, use only positive values
        positive_data = valid_data[valid_data > 0]
        if len(positive_data) > 0:
            # Use provided vmin/vmax or auto-determine from data
            if vmin is None:
                vmin = max(positive_data.min(), 0.01)
            if vmax is None:
                vmax = positive_data.max()
            norm = LogNorm(vmin=vmin, vmax=vmax)
        else:
            print(f"Warning: No positive values for log scale. Using linear scale.")
            norm = None
    else:
        # Linear scale or count variable
        if vmin is not None or vmax is not None:
            # Manual vmin/vmax specified, but no log scale
            norm = None  # Will pass vmin/vmax directly to pcolormesh
        else:
            norm = None
    
    # Plot the gridded data
    if norm is not None:
        im = ax.pcolormesh(
            lons, lats, plot_data,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            norm=norm,
            shading='auto'
        )
    else:
        # Linear scale - apply vmin/vmax directly if provided
        im = ax.pcolormesh(
            lons, lats, plot_data,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading='auto'
        )
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.05, shrink=0.7)
    
    # Determine colorbar label and title
    if title is None:
        # Auto-generate title from variable name
        title = variable.replace('_', ' ').title()
    
    # Set colorbar label
    if 'mean' in variable.lower():
        scale_note = ' [log scale]' if (use_log_scale and not is_count_variable) else ''
        cbar.set_label(f'{variable}{scale_note}', fontsize=FONT_SIZE, color=FOREGROUND_COLOR)
    elif 'std' in variable.lower():
        cbar.set_label(f'{variable} (std dev)', fontsize=FONT_SIZE, color=FOREGROUND_COLOR)
    elif 'count' in variable.lower():
        cbar.set_label(f'{variable} (number of observations)', fontsize=FONT_SIZE, color=FOREGROUND_COLOR)
    elif 'min' in variable.lower():
        cbar.set_label(f'{variable} (minimum)', fontsize=FONT_SIZE, color=FOREGROUND_COLOR)
    elif 'max' in variable.lower():
        cbar.set_label(f'{variable} (maximum)', fontsize=FONT_SIZE, color=FOREGROUND_COLOR)
    else:
        cbar.set_label(variable, fontsize=FONT_SIZE, color=FOREGROUND_COLOR)
    
    cbar.ax.tick_params(colors=FOREGROUND_COLOR)
    
    # Add statistics to title
    n_valid_pixels = np.sum(~np.isnan(plot_data))
    n_nonzero_pixels = np.sum((~np.isnan(plot_data)) & (plot_data > 0))
    
    title_with_stats = f'{title}\n({n_nonzero_pixels} non-zero pixels, {n_valid_pixels} valid pixels)'
    plt.title(title_with_stats, fontsize=FONT_SIZE * 1.25, fontweight='bold', color=FOREGROUND_COLOR, pad=20)
    
    plt.tight_layout()
    
    # Save figure if directory is provided
    if save_fig_dir is not None:
        save_dir = Path(save_fig_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        # Extract date from source_attrs if available
        date_str = 'unknown'
        if 'source_attrs' in dataset.attrs:
            # Try to parse date from source_attrs string
            import re
            attrs_str = dataset.attrs['source_attrs']
            # Look for date pattern in input filename (e.g., S5P_OFFL_L1B_RA_BD7_20210801T...)
            match = re.search(r'(\d{8})T\d{6}', attrs_str)
            if match:
                date_str = match.group(1)
        filename = f"S5P_PCA_L3_{date_str}_{variable}.png"
        filepath = save_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor=BACKGROUND_COLOR)
        print(f"Figure saved to: {filepath}")
    
    return fig, ax


