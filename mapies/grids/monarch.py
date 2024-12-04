#!/usr/bin/env python
from dataclasses import dataclass
from pandas import DataFrame
import xarray as xr
import pandas as pd
import numpy as np
from typing import Tuple, Dict
from numpy import cos, sin, arctan, pi, nan
from functools import partial, wraps
import geopandas as gpd
from geopandas import GeoDataFrame
from shapely.geometry import Polygon, Point
from mapies.util.func_tools import *
import logging



@dataclass
class Grid:
    centre_lat: float
    centre_lon: float
    dlon: float = None
    dlat: float = None
    nlon: float = None
    nlat: float = None
    west: float = None
    south: float = None


    def calculate_grid_coords(self):
        """
        This method does two things:
        1. It calculates the centre values of the grid
        2. Before Calculating the boundary points of the regular grid 

        This can then be used in many grid representations, for example in a rotated grid used in the ihouse Monarch model
        """
        if (self.nlon == None) and (self.nlat == None):
            self.nlat = int((abs(self.south) / self.dlat) * 2 + 1)
            self.nlon = int((abs(self.west) / self.dlon) * 2 + 1)
        elif (self.dlon == None) and (self.dlat == None):
            self.dlon = (abs(self.south) * 2 / self.nlon)
            self.dlat = (abs(self.west) * 2 / self.nlat)
        else:
            print("Enter resolution or number of grid cells")
        print(self.dlat)
        print(self.nlat)
        self.center_latitudes = np.linspace(self.south, self.south + (self.dlat * (self.nlat - 1)), self.nlat, dtype=float) 
        self.center_longitudes = np.linspace(self.west, self.west + (self.dlon * (self.nlon - 1)), self.nlon, dtype=float)
        
        counter = 0
        self.coords = np.empty((self.center_latitudes.shape[0] * self.center_longitudes.shape[0], 2))
        for i in range(self.center_longitudes.shape[0]):
            for j in range(self.center_latitudes.shape[0]):
                self.coords[counter, 0] = self.center_longitudes[i]
                self.coords[counter, 1] = self.center_latitudes[j]
                counter += 1


        self.b_lats = self.create_bounds(self.coords[:, 1], self.dlat, number_vertices=4, inverse=True)
        self.b_lons = self.create_bounds(self.coords[:, 0], self.dlon, number_vertices=4)

        return True


    @staticmethod
    def create_bounds(coordinates, inc, number_vertices=2, inverse=False):
        """
        Calculate the boundaries of each grid points
        """
        # Create new arrays moving the centers half increment less and more.
        coords_left = coordinates - inc / 2
        coords_right = coordinates + inc / 2

        # Defining the number of corners needed. 2 to regular grids and 4 for irregular ones.
        if number_vertices == 2:
            # Create an array of N arrays of 2 elements to store the floor and the ceil values for each cell
            bound_coords = np.dstack((coords_left, coords_right))
            bound_coords = bound_coords.reshape((len(coordinates), number_vertices))
        elif number_vertices == 4:
            # Create an array of N arrays of 4 elements to store the corner values for each cell
            # It can be stored in clockwise starting form the left-top element, or in inverse mode.
            if inverse:
                bound_coords = np.dstack((coords_left, coords_left, coords_right, coords_right))
            else:
                bound_coords = np.dstack((coords_left, coords_right, coords_right, coords_left))
        else:
            print('The number of vertices of the boundaries must be 2 or 4.')
        return bound_coords


    def building_grid(self):
        """
        Here we build the grid using Shapely polygons. And attribute it to a geopandas dataframe

        This is also where we decide whether to use a regular grid for global or a rotated grid for regional

        """


        # May need to add centre lats and centre lons to this dataframe as well

        # Re-shape
        aux_b_lats = self.b_lats.squeeze()
        aux_b_lons = self.b_lons.squeeze()
        

        geometry = []
        # Create one dataframe with 8 columns, 4 points with two coordinates each one
        for i in range(aux_b_lons.shape[0]):
            geometry.append(Polygon([(aux_b_lons[i, 0], aux_b_lats[i, 0]),
                                     (aux_b_lons[i, 1], aux_b_lats[i, 1]),
                                     (aux_b_lons[i, 2], aux_b_lats[i, 2]),
                                     (aux_b_lons[i, 3], aux_b_lats[i, 3]),
                                     (aux_b_lons[i, 0], aux_b_lats[i, 0])]))

        self.gdf = GeoDataFrame(index=range(aux_b_lons.shape[0]), geometry=geometry, crs='epsg:4326')
        self.gdf["grid_cell"] = ["grid_cell_{}".format(i+1) for i in range(len(self.gdf))]
        
        # Append the center lat and lon values to this dataframe
        self.gdf["lon_cen"] = self.coords[:, 0]
        self.gdf["lat_cen"] = self.coords[:, 1]

        
        return True


@dataclass
class RegularGrid(Grid):
    """
    Regridding if the Grid we want is regular, but also want to aggregate the observations to a regular grid.

    Useful for a global grid representation
    """ 
    projection: str = "regular"

    def __post_init__(self):
        self.calculate_grid_coords()
        self.building_grid()
        
    def aggregate(self, lon, lat, obs, obserr=None):
        """
        Aggregate
        
        Parameters:
        lon, lat, obs values Optional obserr
        
        Returns:
        lon, lat, obs values
        """

        if obserr is not None:
            df = pd.DataFrame({"obs":obs, "lon":lon, "lat":lat, "obserr": obserr})
        else:
            df = pd.DataFrame({"obs":obs, "lon":lon, "lat":lat})

        df['coords'] = list(zip(df['lon'],df['lat']))
        df['coords'] = df['coords'].apply(Point)

        # Create a geodataframe of the rotated obseravtional lon/ lat points
        points = gpd.GeoDataFrame(df, geometry='coords', crs='epsg:4326')
        
        
        # Geopandas sjoin calculates if these shapely points are within the shapely grid polygons
        # I tried using predicate="within" but it was procuding 0 values, possibly due to the nan values, sticking with default
        try:
            gdf = self.gdf.sjoin(points, how="left")
        except TypeError:
            print("Empty dataframe messing up some versions of geopandas")
            gdf = pd.DataFrame(columns=[
                'obstype', 'obslon', 'obslat', 'obsn', 'obsid', 'obslev', 'obs',
                'obserr', 'time', 'lon', 'lat', 'coords', 'index_right',
                'lon_cen', 'lat_cen', 'grid_cell'],
                )
        
        # Need to also return the original lat lon
        # Calculate the mean of all points within each grid cell
        gdf = gdf[["grid_cell", "lon_cen", "lat_cen", "obs", "lon", "lat"]].groupby("grid_cell").mean().reset_index()
        
        # Return the data we are interested in UPDATE this
        if obserr is not None:
            return gdf["lon"].values, gdf["lat"].values, gdf["obs"].values, gdf["obserr"].values
        else:
            return gdf["lon_cen"].values, gdf["lat_cen"].values,  gdf["obs"].values




@dataclass
class RotatedGrid(Grid):
    projection: str = "rotated"
        
    def __post_init__(self):
        self.calculate_grid_coords()
        self.building_grid()
        
        
    def aggregate(self, lon, lat, obs, obserr=None):
        """
        Aggregate
        
        Parameters:
        rlon, rlat, obs values Optional obserr
        
        Returns:
        rlon, rlat, obs values
        """
        # Rotate the observations
        rlon, rlat = geo_to_rot(lon, lat, centre_lon=self.centre_lon, centre_lat=self.centre_lat)
        if obserr is not None:
            df = pd.DataFrame({"obs":obs, "lon":lon, "lat":lat, "rlon":rlon, "rlat":rlat, "obserr": obserr})
        else:
            df = pd.DataFrame({"obs":obs, "lon":lon, "lat":lat, "rlon":rlon, "rlat":rlat})

        df['coords'] = list(zip(df['rlon'],df['rlat']))
        df['coords'] = df['coords'].apply(Point)
        

        # Create a geodataframe of the rotated obseravtional lon/ lat points
        points = gpd.GeoDataFrame(df, geometry='coords', crs='epsg:4326')
        
        
        # Geopandas sjoin calculates if these shapely points are within the shapely grid pologons
        try:
            gdf = self.gdf.sjoin(points, how="left")
        except TypeError:
            print("Empty dataframe messing up some versions of geopandas")
            gdf = pd.DataFrame(columns=[
                'obstype', 'obslon', 'obslat', 'obsn', 'obsid', 'obslev', 'obs',
                'obserr', 'time', 'lon', 'lat', 'coords', 'index_right',
                'lon_cen', 'lat_cen', 'grid_cell'],
                )
        
        # Need to also return the original lat lon
        # Calculate the mean of all points within each grid cell
        gdf = gdf[["grid_cell", "lon_cen", "lat_cen", "obs", "lon", "lat", "rlon", "rlat"]].groupby("grid_cell").mean().reset_index()
        
        
        # Return the data we are interested in 
        if obserr is not None:
            return gdf["lon"].values, gdf["lat"].values, gdf["rlon"].values, gdf["rlat"].values, gdf["obs"].values, gdf["obserr"].values
        else:
            return gdf["lon"].values, gdf["lat"].values, gdf["rlon"].values, gdf["rlat"].values, gdf["obs"].values
        
        
