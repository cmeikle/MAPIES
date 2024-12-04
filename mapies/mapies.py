#!/usr/bin/env python
# MAPIES base

from functools import wraps
from datetime import datetime, timedelta
import sys
import os
import yaml
from mapies.util.func_tools import timeit, time_domain_selection, frequency_int, error_estimation
import time
import logging
import numpy as np
import pandas as pd
import xarray as xr
#import matplotlib
#matplotlib.use("TkAgg") # Use tinker to perform plt.show() as it is GUI supported 
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs


class MAPIES:
    """
    Base class for only base functions used in every Mapies run
    
    For example Reading and writing functions
    """
    def __init__(self, start_date, end_date, **kwargs):
        """
        INITIATE the MAPIES class with the variables used in every run
        """
        self.np_typetime = "datetime64[s]"
        self.int_type = 'int64'
        self.time_orig = np.datetime64("1900-01-01T00:00:00")
        self.start_date = start_date
        self.end_date = end_date


    @timeit
    def read_config(self, **kwargs):
        """
        Read yaml config file
        """
        module_dir = os.path.dirname(__file__)


        # Let the user read in their own config
        config_file = kwargs.get("config_file")
        if config_file:
            try:
                self.config = yaml.safe_load(open(config_file))
            except FileNotFoundError:
                logging.error("This is not a config file")
        else:
            self.config = yaml.safe_load(open(os.path.join(module_dir, "config/satellite_config.yaml")))


        variable_dict = self.config[self.datatype]["variables"]
        self.time_var = variable_dict["time_variable"]
        self.lon_var = variable_dict["lon_variable"]
        self.lat_var = variable_dict["lat_variable"]

        # If obs_var passed then run data analysis on that variable if not pull default
        obs_var = kwargs.get("obs_var")
        if obs_var:
            self.obs_var = obs_var
        else:
            self.obs_var = variable_dict["obs_variable"]



    def preprocess_vars(self):
        """
        Preprocessing of the dataset
        """
        # Get all info about time columns
        self.time_dims = self.ds[self.time_var].dims
        self.time_shape = self.ds[self.time_var].shape
        self.time_attrs = self.ds[self.time_var].attrs # Useful if we need to convert to datetime
        

        # Get time values flatten and convert to datetime values
        self.time_values = self.ds[self.time_var].values
        self.time_values = self.time_values.flatten()

        
        if self.time_values.dtype == "timedelta64[ns]":
            logging.info("Adding time origin to time values as the time variable is in timedelta")
            self.time_values = np.add(self.time_orig, self.time_values)
            self.time_values = pd.to_datetime(self.time_values)
        else:
            self.time_values = pd.to_datetime(self.time_values)
        
        
        # Time domain selection
        if self.start_date and self.end_date:
            self.time_values, self.time_values_index = time_domain_selection(self.time_values, self.start_date, self.end_date)


        # TODO: We need to do some reindexing of the longitude and latitude variables too so that if we need them for regridding later we have them reindexed



    @timeit
    def plot_2D_obs(self, obs=None, **kwargs):
        """
        Plotting the observations
        """
        
        # TODO: Make these arguments
        figsize = (15,10)
        #markersize = 2.5
        markersize = 100

        # Set Basemap projection
        proj = ccrs.PlateCarree()

        # Create the plot and add features, TODO make this adjustable in arguments
        fig, ax = plt.subplots(subplot_kw={"projection": proj}, figsize=figsize)
        ax.gridlines()
        #ax.add_feature(cartopy.feature.BORDERS, linestyle=':', alpha=1)
        #ax.add_feature(cartopy.feature.OCEAN,facecolor=("lightblue"))
        #ax.add_feature(cartopy.feature.LAND)
        ax.coastlines(resolution='10m')

        

        print("Plotting observations")
        

        x, y = self.lon_values, self.lat_values

        im = ax.scatter(x,y,markersize,c=self.obs, transform=proj)
        
        fig.colorbar(im, ax=ax)
        
        ax.set_title(f'Observation 2D plot of {self.datatype.upper()} data from {self.start_date} to {self.end_date}')
        print("Saving Figure")
        plt.savefig(f"{self.dest}{self.datatype}_2D_obs.png", format="png")
        plt.close(fig)


    @staticmethod
    def reindex(
        dependent_var_index,
        independent_var_values,
        ):
        """"
        Recutting the data whenever a selction of the data has been made along one of the dimenisons
        
        Based off of how it has been done in Providentia
        """
        # Maybe add a checker try/except or assert
        independent_var_values = independent_var_values[dependent_var_index]

        # return reindexed values
        return independent_var_values

    
    
    @staticmethod
    def to_xarray(coords:dict, data_vars:dict, **kwargs):
        """
        Method to convert numpy arrays to xarray opject
        """
        attrs = kwargs.get("attrs")
        if attrs is None:
            attrs=dict()

        ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
        return ds

