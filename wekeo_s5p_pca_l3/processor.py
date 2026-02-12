import xarray as xr
import numpy as np

class thresholds:
    score_CO_1 = 0.405
    score_CO_2 = 0.516
    score_CO_3 = 0.627 # TODO; NOTE: randomly chosen, needs to be determined properly
    
   
variables_to_reproject = [
    "cloud_fraction",
    "score_CO_1",
    "score_CO_2",
    "score_CO_3",
    "diag_CO_1",
    "diag_CO_2",
    "diag_CO_3",
    "nb_detect",
]


def preprocess_invalid_data_nan(ds, fill_value=-9999):
    """
    Replace NaN values with fill_value for all variables of interest.
    """
    
    # Handle missing values for all variables of interest
    for var in variables_to_reproject:
        if var in ds:
            # Replace NaN values with fill_value
            ds[var] = ds[var].where(~np.isnan(ds[var]), fill_value)
    
    # Set to NaN where processing_flag has 16 (south_atlantic_anomaly)
    ds = ds.where(
        (ds.processing_flag.astype(np.uint8) != 1),
        other=np.nan
    )
    
    return ds
    

def preprocess_detection(ds, min_detection=1):
    """
    Add a new variable "nb_detect" to the dataset, which counts the number of scores above their respective thresholds for each pixel.
    Then filter out pixels that do not meet the minimum detection requirement.
    """
    ds["nb_detect"] = xr.DataArray(
        dims=('time', 'scanline', 'ground_pixel'), 
        data=np.zeros(
            (ds.time.size, ds.scanline.size, ds.ground_pixel.size), 
            dtype=np.uint8
        ) # NOTE: nb_detect E [0,1,2,3]
    )
    
    # Increment nb_detect for each score above its respective threshold
    ds["nb_detect"] += (ds.score_CO_1 > thresholds.score_CO_1).astype(np.uint8)
    ds["nb_detect"] += (ds.score_CO_2 > thresholds.score_CO_2).astype(np.uint8)
    ds["nb_detect"] += (ds.score_CO_3 > thresholds.score_CO_3).astype(np.uint8)
    
    # Filter out detections that do not meet the minimum detection requirement
    ds = ds.where(ds["nb_detect"] >= min_detection)
    
    return ds
    

def preprocess(ds, min_detection=1, fill_value=-9999):
    """
    Preprocess the dataset by handling invalid data and filtering detections based on a minimum detection threshold.
    """
    ds = preprocess_invalid_data_nan(ds, fill_value)
    ds = preprocess_detection(ds, min_detection)
    
    return ds


def accumulate_to_grid(
    dataset: xr.Dataset,
    width: int,
    lat_name: str = "latitude",
    lon_name: str = "longitude",
    min_count: int = 1,
) -> xr.Dataset:
    """
    Accumulate S5P PCA detection data into a 2D geographic grid
    and compute statistics per grid cell for multiple data fields.
    
    Parameters
    ----------
    dataset : xr.Dataset
        Input dataset containing the detection data with dimensions (time, scanline, ground_pixel)
    width : int
        Width of the output grid (longitude bins)
    lat_name : str, optional
        Name of the latitude variable in the dataset (default: "latitude")
    lon_name : str, optional
        Name of the longitude variable in the dataset (default: "longitude")
    min_count : int, optional
        Minimum number of observations required per grid cell to compute statistics
    
    Returns
    -------
    xr.Dataset
        Dataset with dimensions (latitude, longitude) containing for each field:
        - {field}_mean: mean value per grid cell
        - {field}_std: standard deviation per grid cell
        - {field}_min: minimum value per grid cell
        - {field}_max: maximum value per grid cell
        - {field}_count: number of observations per grid cell
    """
    
    height = width // 2
    assert width == 2 * height, "Expected width to be 2*height for lat/lon grid"
    
    variables = variables_to_reproject.copy()
    assert len(variables) > 0, "Must provide at least one variable to accumulate"
    
    # Extract lat/lon arrays from dataset and flatten
    lat = dataset[lat_name].values.flatten()
    lon = dataset[lon_name].values.flatten()
    
    # Create coordinate arrays
    lat_coords = np.linspace(-90, 90, height, endpoint=True)   # latitude is not circular
    lon_coords = np.linspace(-180, 180, width, endpoint=False) # longitude is circular
    
    # Convert lat/lon to grid indices (same for all fields)
    lat_idx = np.uint32(
        np.round((lat + 90.0) * ((height - 1) / 180.0))
    )
    
    lon_idx = np.uint32(
        np.round((lon + 180.0) * (width / 360.0)) % width
    )
    
    # Build the dataset by accumulating each field
    data_vars = {}
    
    for variable in variables:
        
        if variable not in dataset:
            continue
            
        # Extract data and flatten
        data = dataset[variable].values.flatten()
        
        # Filter out NaNs and invalid values
        filt = ~np.isnan(data) & ~np.isnan(lat) & ~np.isnan(lon)
        
        data_valid = data[filt]
        lat_idx_valid = lat_idx[filt]
        lon_idx_valid = lon_idx[filt]
        
        # Initialize accumulators
        sum_grid = np.zeros((height, width), dtype=np.float64)
        sum_sq_grid = np.zeros((height, width), dtype=np.float64)
        count_grid = np.zeros((height, width), dtype=np.int32)
        
        # Accumulate using np.add.at (fast)
        np.add.at(sum_grid, (lat_idx_valid, lon_idx_valid), data_valid)
        np.add.at(sum_sq_grid, (lat_idx_valid, lon_idx_valid), data_valid**2)
        np.add.at(count_grid, (lat_idx_valid, lon_idx_valid), 1)

        # Initialize min/max grids with appropriate values
        min_grid = np.full((height, width), np.inf, dtype=np.float32)
        max_grid = np.full((height, width), -np.inf, dtype=np.float32)

        # Accumulate min/max values directly
        np.minimum.at(min_grid, (lat_idx_valid, lon_idx_valid), data_valid)
        np.maximum.at(max_grid, (lat_idx_valid, lon_idx_valid), data_valid)
        
        # Compute mean and std
        mask = count_grid > 0
        mean_grid = np.full((height, width), np.nan, dtype=np.float32)
        std_grid = np.full((height, width), np.nan, dtype=np.float32)
        mean_grid[mask] = (sum_grid[mask] / count_grid[mask]).astype(np.float32)
        
        # std = sqrt(E[X^2] - E[X]^2)
        mean_sq = sum_sq_grid[mask] / count_grid[mask]
        std_grid[mask] = np.sqrt(np.maximum(0, mean_sq - mean_grid[mask]**2)).astype(np.float32)
        
        # Filter cells with insufficient data
        insufficient_data_mask = count_grid < min_count
        mean_grid[insufficient_data_mask] = np.nan
        std_grid[insufficient_data_mask] = np.nan
        min_grid[insufficient_data_mask] = np.nan
        max_grid[insufficient_data_mask] = np.nan
        count_grid[insufficient_data_mask] = -1
        
        # Remove infs from min/max grids
        min_grid[np.isinf(min_grid)] = np.nan
        max_grid[np.isinf(max_grid)] = np.nan
        
        # Add to data_vars dictionary
        data_vars[f'{variable}_mean'] = (('latitude', 'longitude'), mean_grid)
        data_vars[f'{variable}_count'] = (('latitude', 'longitude'), count_grid)
        
        # Unlike FRP only the mean and count are strictly necessary
        # data_vars[f'{variable}_std'] = (('latitude', 'longitude'), std_grid)
        # data_vars[f'{variable}_min'] = (('latitude', 'longitude'), min_grid)
        # data_vars[f'{variable}_max'] = (('latitude', 'longitude'), max_grid)
    
    # Create xarray Dataset
    result_ds = xr.Dataset(
        data_vars,
        coords={
            'latitude': lat_coords,
            'longitude': lon_coords,
        },
        attrs={
            'description': 'Accumulated S5P PCA detection data on an equirectangular grid',
            'grid_width': width,
            'grid_height': height,
            'source_attrs': str(dataset.attrs),
        }
    )
    
    return result_ds
    
    

def postprocess(ds: xr.Dataset):
    """
    Postprocess the accumulated dataset fields
    """

    # variables to remove after the computation of the final fields
    # tmp_vars = [s for s in list(ds.data_vars) if s.endswith('_count')]

    ds["mean_score_CO"] = (ds["score_CO_1_mean"] + ds["score_CO_2_mean"] + ds["score_CO_3_mean"]) / 3.0
    ds["mean_diag_CO"]  = (ds["diag_CO_1_mean"] + ds["diag_CO_2_mean"] + ds["diag_CO_3_mean"]) / 3.0
    ds["mean_cloud_fraction"] = ds["cloud_fraction_mean"]
    
    ds["nb_samples"] = ds["score_CO_1_count"]  # all count fields should be the same, so we can just take one of them
    ds["mean_nb_detect"] = ds["nb_detect_mean"]     # rename
    
    # ds.drop_vars(tmp_vars, errors='ignore')
    
    return ds