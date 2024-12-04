#!/usr/bin/env python
from dataclasses import dataclass, field
from typing import List, Tuple
from netCDF4 import Dataset, num2date, chartostring
from datetime import datetime, timedelta

from functools import wraps

from mapies.mapies import MAPIES
from mapies.util.func_tools import timeit, get_file_list, time_converter, inflate_array
import time
import os
import numpy as np
import pandas as pd
import xarray as xr



class TROPOMI(MAPIES):
    """
import pandas as pd
    Class for VIIRS specific data
    """
    
    def __init__(self, start_date, end_date,**kwargs):
        """
        Inherited init class with new variables
        """
        super().__init__(start_date, end_date)
        
        self.time_orig = np.datetime64("1993-01-01T00:00:00")
        self.datatype = "tropomi"
        self.dest = kwargs.get("dest")
        self.indir = kwargs.get("indir")
        #self.quality_flag_limit

        # Add quality value filter number


        if isinstance(self.start_date, str):
            self.start_date = time_converter(self.start_date)

        if isinstance(self.end_date, str):
            self.end_date = time_converter(self.end_date)


        self.read_config(**kwargs)


    @timeit
    def preprocess_vars(self):
        """
        Preprocessing of the dataset
        """
        super().preprocess_vars()


        # no2 column values
        obs_dims = self.ds[self.obs_var].dims
        obs_shape = self.ds[self.obs_var].shape
        obs_attrs = self.ds[self.obs_var].attrs


        # Duplicate values in the time array to get the same shape as the flattened variable array
        self.time_values = inflate_array(self.time_values, self.time_shape, obs_shape)
        self.time_values_index = inflate_array(self.time_values_index, self.time_shape, obs_shape)
        # Lon and lat values
        lon_dims = self.ds[self.lon_var].dims
        lon_shape = self.ds[self.lon_var].shape
        lon_attrs = self.ds[self.lon_var].attrs
        self.lon_values = self.ds[self.lon_var].values.flatten()

        # Reindex lon column values
        self.lon_values = self.reindex(
            self.time_values_index, 
            self.lon_values,
            )

        print(self.time_values)


        lat_dims = self.ds[self.lat_var].dims
        lat_shape = self.ds[self.lat_var].shape
        lat_attrs = self.ds[self.lat_var].attrs
        self.lat_values = self.ds[self.lat_var].values.flatten()

        # Reindex aod column values
        self.lat_values = self.reindex(
            self.time_values_index, 
            self.lat_values,
            )

        # TODO: Check if this works well, if the inflated values are along the right dimensions
        self.obs = self.ds[self.obs_var].values.flatten()

        print(np.nanmax(self.obs))
        print(self.obs.shape)
        """
        # Reindex no2 column values
        self.obs = self.reindex(
            self.time_values_index,
            self.obs, 
            )
        """
        print(self.lon_values)
        print(self.lat_values)
        print(self.obs)
        print(self.lon_values.shape)
        print(self.lat_values.shape)
        print(self.obs.shape)
        print(np.nanmax(self.obs))


    @timeit
    def read_nc(self):
        """
        Read netcdf files with xarray
        """
        file_patterns = f'{self.indir}/S5P_OFFL_L2__NO2____20230101*'
        files = get_file_list(file_patterns)
        print(file_patterns)
        print(files)
        
        
        # Open dataset with xarray and dask
        # Tropomi requires you choose the group
        # If we want to read anything other than a single tropomi dataset we will need to do it in parallel
        ds_list = []
        index = []
        for file in files:
            ds = xr.open_mfdataset(files, group="PRODUCT")
            ds_list.append(ds)
            index.append(ds["time"].values)
        self.ds = xr.concat(ds_list, dim=pd.Index(index))
        print(self.ds)


    @timeit
    def to_plumes(self):
        """
        Restructing the data to be passed to calculate the plumes 
        """

        coords = coords = dict(time=time_values,)

        ds = self.to_xarray(coords=coords, data_vars=data_vars)

