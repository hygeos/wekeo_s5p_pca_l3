import xarray as xr
import numpy as np

class GlobalAccumulator2D:
    
    def __init__(self, width: int, height: int):
        
        # store dimensions
        self.width  = width
        self.height = height
        self.dims = (height, width)  # Note: switched to (height, width) for standard array indexing
        
        assert self.width == 2 * self.height
        
        self.lat = np.linspace( -90,  90, height, endpoint=True) # latitude is not circular:    -90 !=  90 -> include endpoint
        self.lon = np.linspace(-180, 180, width, endpoint=False) # longitude is circular:      -180 == 180 -> exclude endpoint
        
        # Use 2D arrays instead of flattened for simplicity (very minor impact on performance)
        self.acc = np.zeros((self.height, self.width), dtype=np.float64)
        self.cnt = np.zeros((self.height, self.width), dtype=np.uint32)
    
    
    def add(self, data: np.ndarray, lat: np.ndarray, lon: np.ndarray):
        self._add_numpy(data, lat, lon)
    
    
    def get_mean_data_array(self):
        
        mask = np.isnan(self.acc) | (self.cnt <= 0)           # invalid case to compute mean
        mean = np.full(self.acc.shape, np.nan)                # fill with nans
        mean[~mask] = self.acc[~mask] / self.cnt[~mask]       # compute mean where possible
        
        # No need to reshape since we're already 2D
        return xr.DataArray(
            data=mean.astype(np.float32), 
            dims=("latitude", "longitude"), 
            coords={"latitude": self.lat, "longitude": self.lon}
        )
    
    def get_cnt_data_array(self):
        # No need to reshape since we're already 2D
        return xr.DataArray(
            data=self.cnt, 
            dims=("latitude", "longitude"), 
            coords={"latitude": self.lat, "longitude": self.lon}
        )
    

    def _add_numpy(self, data: np.ndarray, lat: np.ndarray, lon: np.ndarray):
        """
        Take a list of paths to ncdf files, accumulate them into a lvl3 grid
        """
        
        width, height = self.width, self.height
        
        # convert to indexes, use nearest integer np.uint32
        lat_idx = np.uint32(
            np.round((lat + 90.0) * ((height - 1) / 180.0))
        )          # height - 1 is the maximum index
        
        lon_idx = np.uint32(np.round(
            ((lon + 180.0) * (width / 360.0)) % width
        ))  # modulo because the dimension is circular
        
        # apply filter
        filt = ~np.isnan(data)
        data_valid = data[filt]
        lat_idx_valid = lat_idx[filt]
        lon_idx_valid = lon_idx[filt]
        
        # Ensure indices are within bounds
        lat_idx_valid = np.clip(lat_idx_valid, 0, height - 1)
        lon_idx_valid = np.clip(lon_idx_valid, 0, width - 1)
        
        # Use pure 2D fancy indexing with np.add.at
        np.add.at(self.acc, (lat_idx_valid, lon_idx_valid), data_valid)
        np.add.at(self.cnt, (lat_idx_valid, lon_idx_valid), 1)
        
        
    def merge(self, other: 'GlobalAccumulator2D'):
        """
        Merge another accumulator into this one
        """
        assert self.width == other.width and self.height == other.height
        self.acc += other.acc
        self.cnt += other.cnt