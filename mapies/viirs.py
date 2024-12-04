#!/usr/bin/env python
from datetime import datetime, timedelta
from functools import wraps

import logging
import os
import time
import yaml

import numpy as np
import pandas as pd
import xarray as xr

from mapies.mapies import MAPIES
from mapies.util.func_tools import *
#timeit, get_file_list, time_converter, frequency_int, error_estimation, time_domain_selection, geo_to_rot
from mapies.grids.monarch import RotatedGrid
from pathlib import Path



# TODO: still need to do a load of filtering huuuhhh
# TODO: Create Yaml config file for time, lat, lon, uncertainty const etc.


# Uncertainty constants should be able to be passed as number, array or dict

class VIIRS(MAPIES):
    """
    Class for VIIRS specific data
    """

    def __init__(self, start_date, end_date, **kwargs):
        """
        Inherited init class with new variables for VIIRS
        """
        super().__init__(start_date, end_date)
        
        self.time_orig = np.datetime64("1993-01-01T00:00:00")
        self.datatype="viirs"
        frequency = kwargs.get("frequency")
        self.dest = kwargs.get("dest")
        self.indir = kwargs.get("indir")

        self.dates_slice = pd.date_range(
            self.start_date, 
            self.end_date, 
            freq=frequency
            ).strftime('%Y%m%d%H%M')
        if frequency:
            self.frequency = frequency_int(frequency)

        if isinstance(self.start_date, str):
            self.start_date = time_converter(self.start_date)

        if isinstance(self.end_date, str):
            self.end_date = time_converter(self.end_date)

        self.read_config(**kwargs)

    def read_config(self, **kwargs):
        super().read_config()
        da_dict = self.config[self.datatype]["da"]
        self.obsid_da = da_dict["obsid"] #int

        unc_const = kwargs.get("unc_const")
        if unc_const:
            self.unc_const = unc_const
        else:
            self.unc_const = da_dict["uncertainty_constants"] #float, int, list, dict
        

        grid_repr = kwargs.get("grid_repr")
        if grid_repr:
            self.grid_repr = grid_repr
        else:
            self.grid_repr = da_dict["grid_repr"] #str

    @timeit
    def preprocess_vars(self):
        """
        Preprocessing of the dataset
        """
        super().preprocess_vars()

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

        lat_dims = self.ds[self.lat_var].dims
        lat_shape = self.ds[self.lat_var].shape
        lat_attrs = self.ds[self.lat_var].attrs
        self.lat_values = self.ds[self.lat_var].values.flatten()

        # Reindex aod column values
        self.lat_values = self.reindex(
            self.time_values_index, 
            self.lat_values,
            )


        # AOD column values, default "Aerosol_Optical_Thickness_550_Land_Ocean_Best_Estimate"
        aod_dims = self.ds[self.obs_var].dims
        aod_shape = self.ds[self.obs_var].shape
        aod_attrs = self.ds[self.obs_var].attrs
        self.obs = self.ds[self.obs_var].values.flatten()

        # Reindex aod column values
        self.obs = self.reindex(
            self.time_values_index, 
            self.obs,
            )


    @timeit
    def rotate(self):
        """
        Perform Rotation of Grid representation
        """
        # Calculate rotated lon and lat values 
        
        # Calculate the grid representation
        r = RotatedGrid(centre_lon=20, centre_lat=35, dlon=.1, dlat=.1, west=-51, south=-35)
        # Aggregate the the observations to the grid representation
        lon, lat, rlon, rlat, obs = r.aggregate(self.lon_values, self.lat_values, self.obs)
        self.lon_values = lon
        self.lat_values = lat
        self.obs = obs

    @timeit
    def read_nc(self):
        """
        Read netcdf files with xarray
        """
        # Take in start and end date and convert to julian dates
        file_patterns = []
        for date in self.dates_slice:
            date = datetime.strptime(date, '%Y%m%d%H%M').strftime('%Y%j')
            #filepaths = f'{self.indir}/{date[0:4]}/{date[4:]}/AERDB_L2_VIIRS_NOAA20.A{date}*'
            filepaths = f'{self.indir}/*/AERDB_L2_VIIRS_NOAA20.A{date}*'
            print(filepaths)
            file_patterns.append(filepaths)

        # TODO:  maybe change this to NETCDF4
        files = get_file_list(file_patterns)

        # A bit of preprocessing to start before reading in the files
        def preprocess(ds):
            ds = ds.expand_dims(dim={"index": [ds.attrs["product_name"]]})
            return ds

        # Open dataset with xarray and dask
        self.ds = xr.open_mfdataset(files, preprocess=preprocess)

    @timeit
    def to_da(self):
        """
        Function that returns all the needed variables for the DA
        """

        # TODO: Move this
        r = RotatedGrid(centre_lon=20, centre_lat=35, dlon=.1, dlat=.1, west=-51, south=-35)
        # Calculate error
        self.obserr = error_estimation(self.datatype, self.obs, self.unc_const)

        outfiles = []

        # Cut to the frequencies that we want
        # Look at start and end date - Find date ranges in between with that frequency 
        # Then find the boundaries
        # We take the start date as the midpoint of the lower boundary
        for date in self.dates_slice:
            l_border = (datetime.strptime(date, "%Y%m%d%H%M") - timedelta(hours=self.frequency/2)).strftime('%Y%m%d%H%M')
            r_border = (datetime.strptime(date, "%Y%m%d%H%M") + timedelta(hours=self.frequency/2)).strftime('%Y%m%d%H%M')
            # Included by Guillaume in his DA function so I believe he needs this to be true
            filename = Path(self.dest).joinpath(f'obs{date}.nc')
            # Cut down the time interval again then reindex obs, obserr, lat and lon
            # Returning also arrays for type, lev, id and n with the same index as the time var
            time_values, frequency_index = time_domain_selection(
                self.time_values, 
                l_border, 
                r_border,
                )


            # Reindex
            obs = self.reindex(frequency_index, self.obs)
            lon = self.reindex(frequency_index, self.lon_values)
            lat = self.reindex(frequency_index, self.lat_values)
            obserr = self.reindex(frequency_index, self.obserr)
            # Run aggregation with grid representation
            lon, lat, obs, obserr = r.aggregate(lon, lat, obs, obserr)
            rlon, rlat = geo_to_rot(lon, lat, centre_lon=20, centre_lat=35)


            # Create new arrays with same length as obs
            obsid = np.full(shape=obs.shape, fill_value=self.obsid_da, dtype=self.int_type)
            obstype = np.full(shape=obs.shape, fill_value=30, dtype=self.int_type)
            obslev = np.full(shape=obs.shape, fill_value=-99999, dtype=self.int_type)
            obsn = np.full(shape=obs.shape, fill_value=1, dtype=self.int_type)
            time_values = np.full(shape=obs.shape, fill_value=date, dtype=str)

            # Coords equals index

            assert obs.shape == obserr.shape
            assert lon.shape == time_values.shape
            coords = dict(index=("index", np.indices(obs.shape)[0, :]))
            data_vars = dict(
                time=(["index",], time_values),
                lon=(["index",], lon), 
                lat=(["index",], lat),
                rlon=(["index",], rlon), 
                rlat=(["index",], rlat), #Also return rlon/rlat values if available
                obs=(["index",], obs),
                obserr=(["index", ], obserr),
                obsid=(["index", ], obsid),
                obstype=(["index", ], obstype),
                obslev=(["index", ], obslev),
                obsn=(["index", ], obsn),
            )

            ds = self.to_xarray(coords=coords, data_vars=data_vars)

            ds.to_netcdf(filename, encoding={})
            outfiles.append(filename)

        return outfiles

    # Plots
    def plot_2D_obs(self, outdir="./", **kwargs):
        """
        Plotting the observations specific to VIIRS
        """
        if self.grid_repr == "rotated":
            self.rotate()
        super().plot_2D_obs()
