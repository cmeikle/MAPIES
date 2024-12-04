#!/usr/bin/env python
from datetime import datetime, timedelta
from functools import wraps

import logging
import os
import time
import yaml
import calendar

import numpy as np
import pandas as pd
import xarray as xr
import dask.array as da

from mapies.mapies import MAPIES
from mapies.util.func_tools import *
from mapies.util.cloudsat_config import VARIABLE_DICT
#timeit, get_file_list, time_converter, frequency_int, error_estimation, time_domain_selection, geo_to_rot
from mapies.grids.monarch import RegularGrid
from pathlib import Path

from pyhdf.SD import SD, SDC
from pyhdf.HDF import HDF
import pyhdf.VS




# Uncertainty constants should be able to be passed as number, array or dict

class CLOUDSAT(MAPIES):
    """
    Script for reprocessing of Cloudsat data to a MAPIES format
    """

    def __init__(self, start_date, end_date, **kwargs):
        """
        Inherited init class with new variables for Cloudsat
        """
        super().__init__(start_date, end_date)
        self.datatype="cloudsat"
        self.dest = kwargs.get("dest")
        self.indir = kwargs.get("indir")
        
        if isinstance(self.start_date, str):
            self.start_date = time_converter(self.start_date)

        if isinstance(self.end_date, str):
            self.end_date = time_converter(self.end_date)
        self.read_config(**kwargs)


    def read_multiple_files(self):
        # Create a function to read in multiple files in regex and also for the julian days
        # cut down for month using start and end date
        output_data= {}
        monthly_dates_dict = monthly_dates(self.start_date, self.end_date, output_date_format="%Y%m%d")
        for month, dates in monthly_dates_dict.items():
            print(f"Running analysis for this month {month}")
            monthly_data = {}
            for date in dates:
                print(date)
                data = {}
                file_pattern = f'{self.indir}/{date}*'

                filenames = get_file_list(file_pattern)
                print(filenames)
                # Return new dictionary with TAI start
                for filename in filenames:
                    da_dict = self.read_hdf(filename)
                    da_dict = self.get_geoprof_time(da_dict)
                    epoch = tai_base = calendar.timegm((1993, 1, 1, 0, 0, 0)) + da_dict["data_vars"]['TAI_start']["data"]
                    print(f"Reading in this file {filename}")
                    # save each file data in a different time variable each time
                    data[epoch] = da_dict

                monthly_data[date] = self.preprocess_vars(data)

            days_in_data = len(dates)

            output_data[month] = self.daily_process_vars(monthly_data, days_in_data)
        



        print(output_data)
        self.obs = output_data["202001"]['RO_liq_water_path']
        self.output_nc(output_data)

        
        

    # I need to connect this to main MAPIES and how we deal with hdf5 files not just netcdf
    @staticmethod
    def read_hdf(filename):
        """
        Open the cloudsat data

        Make this more general for all MAPIES
        """
        da_dict = {
            "coords":{
                "nray":{"dims":"nray", "data":np.arange(0, 36946)},
                "nbin":{"dims":"nbin", "data":np.arange(0, 125)}
            },
            "dims":["nbin", "nray"],
            "data_vars":{}
        }

        # First read SD (scientific datasets)
        sd = SD(filename)
        for dname, info in VARIABLE_DICT.items():

            if dname not in sd.datasets().keys(): 
                continue
            
            sds = sd.select(dname)
            # get (masked) data
            data = np.array(sds[:]).squeeze()
            data = array_calcs(data, info)
            # Converting to dask array
            data = da.from_array(data, chunks=500)

    
            da_dict["data_vars"][dname] = {"data":data, "dims":info["dims"]}
            # Close this dataset
            sds.endaccess()
        # Close file
        sd.end()
        
        # ...now read VDATA...
        hdf = HDF(filename)
        vs = hdf.vstart()
        for vname, info in VARIABLE_DICT.items():
            
            if vname not in [v[0] for v in vs.vdatainfo()]: 
                continue
            
            # attach vdata
            vd = vs.attach(vname)

            # read data
            data = np.array(vd[:]).squeeze()
            data = array_calcs(data, info)
            #data = da.from_array(data, chunks=500)

            if info["dims"] == "scalar":
                da_dict["data_vars"][vname] = {"data":data, "dims":()}
            else:
                da_dict["data_vars"][vname] = {"data":data, "dims":info["dims"]}
                
            vd.detach()
        # clean up
        vs.end()
        # HDF files do not always close cleanly, so close manually
        hdf.close()

        return da_dict
        
    @staticmethod
    def get_geoprof_time(da_dict):
        """
        Read time stamp and return seconds since Epoch
        """

        # read data
        profile_sec = da_dict["data_vars"]['Profile_time']["data"]
        start_tai = da_dict["data_vars"]['TAI_start']["data"]
        # TAI time for each profile
        time_tai = (profile_sec + start_tai)  # seconds since 00:00:00 Jan 1
        tai_base = calendar.timegm((1993, 1, 1, 0, 0, 0))
        # get epoch time as a datetime
        epoch = tai_base + time_tai
        # get array of datetime objects
        data = np.array([datetime.utcfromtimestamp(t) for t in epoch])
        
        da_dict["data_vars"]["time"] = {"data":data, "dims":da_dict["data_vars"]['Profile_time']["dims"]}

        return da_dict

    def daily_process_vars(self, orig_data, days_in_data):
        """
        """
        data_dict = {}
        for time, data in orig_data.items():
            for var, info in data.items():
                if var in [self.lon_var, self.lat_var, self.time_var]:
                    continue

                
                # Sum the aggregated variables
                if var in data_dict:
                    a = data_dict[var]
                    b = info
                    data_dict[var] = np.nansum(np.dstack((a,b)), 2).flatten()
                else:
                    data_dict[var] = info

        
        # Divide by number of days in month, to get the mean monthly value
        mean_data_dict = {var: np.divide(info, days_in_data) for var, info in data_dict.items()}
        
        return mean_data_dict



    def preprocess_vars(self, orig_data):
        """
        Preprocessing of the dataset.

        This involves creating a regular 5 by 5 grid 
        """

        r = RegularGrid(centre_lon=0, centre_lat=0, nlat= 256, nlon=512, west=-180, south=-90)
        # I need to save the lon and lat values from the grid repr
        grid_lon_values = r.gdf["lon_cen"]
        grid_lat_values = r.gdf["lat_cen"]
        data_dict = {}
        for time, data in orig_data.items():
            lon_values = data["data_vars"][self.lon_var]["data"].flatten()
            lat_values = data["data_vars"][self.lat_var]["data"].flatten()
            
            for var, info in data["data_vars"].items():
                print(info["data"].shape)
                if var in [self.lon_var, self.lat_var, self.time_var]:
                    continue
                
                # Aggregate the data in the grid and then add to the data dict
                obs = info["data"].flatten()

                # If I can 
                try:
                    lon, lat, obs = r.aggregate(lon_values, lat_values, obs)
                except ValueError:
                    continue
                # Save obs len
                nobs = len(obs)
                print(np.nanmin(obs))

                # TODO: I need to also conserve some attributes, number of obs etc.
                # As well as lat lon time variable values
                # Add the new lon and lat at the end
                # Sum the aggregated variables
                if var in data_dict:
                    a = data_dict[var]
                    b = obs
                    data_dict[var] = np.nansum(np.dstack((a,b)), 2).flatten()
                else:
                    data_dict[var] = obs
                

                
        # final pass, append lat and lon values

        data_dict[self.lon_var] = np.unique(grid_lon_values)
        data_dict[self.lat_var] = np.unique(grid_lat_values)
        # Not adding time var as we are calculating a monthly mean
        self.lon_values = grid_lon_values
        self.lat_values = grid_lat_values

        # To get mean also divide by number of grid_cells
        num_grid_cells = len(np.unique(grid_lon_values))*len(np.unique(grid_lat_values))
        print(data_dict)
        mean_data_dict = {var: np.divide(info, num_grid_cells) for var, info in data_dict.items()}

        return data_dict
        

    def output_nc(self, output_data):
        """
        Output the data to netcdf,
        Need to have the option to ouptut all data for day or regridded for month
        """
        # We already have the output dir variable
        # We can run through the data dictionary 
        # And output each variable to separate directories
                    # I need to reshape the obs
        print(self.lon_values)
        print(self.lat_values)
        lon_values = np.unique(self.lon_values)
        lat_values = np.unique(self.lat_values)
        
        for month, monthly_data in output_data.items():
            # Create an xarray dataset for each variable and then output it to created dirctory
            for var, data in monthly_data.items():
    
                data_for_var = np.reshape(data, (len(lon_values), len(lat_values)))
                ds = xr.Dataset(
                    data_vars={var:(["lon", "lat"], data_for_var)},
                    coords=dict(
                        lon=("lon", lon_values),
                        lat=("lat", lat_values),
                    ),
                    attrs=dict(description="Weather related data."),
                )
                print(month)
                print(ds)
                outdir = f"{self.dest}/{var}/"
                if not os.path.exists(outdir):
                    os.makedirs(outdir)
                output_path = f"{outdir}/{month}.nc"
                print(output_path)

                ds.to_netcdf(output_path)



    # Plots
    def plot_2D_obs(self, outdir="./", **kwargs):
        """
        Plotting the observations specific to VIIRS
        """
        super().plot_2D_obs()
    

