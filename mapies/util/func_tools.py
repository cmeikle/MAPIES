import time
import logging
import numpy as np
import pandas as pd
import xarray as xr

from functools import wraps
from numpy import cos, sin, arctan, pi, nan
from glob import glob
from typing import List
from datetime import datetime



#Timing decorator
def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper


# Function from Guillaume's script could be useful
# Only supported in python3.10
def get_file_list(patterns) -> List[str]:
    """
    :param patterns: one or several glob patterns (or exact file names) pointing to the location of files on disk.
    :return: the matching list of files
    """

    # 1st make sure we have a list of patterns
    if isinstance(patterns, str):
        patterns = [patterns]

    # Then decode them one by one
    files = []
    for pattern in patterns:
        logging.debug(f"Getting file list matching {pattern}")
        files.extend(glob(pattern))

    return files

def exception_factory(exception, message):
    return exception(message)


def time_converter(date:str) -> pd.Timestamp:
    """
    :param date: date that you wish convert to standardised format
    """
    # Remove any special characters
    for ch in ['\\','`','*','_','{','}','[',']','(',')','%','>','#','+','-','.','!','$','\'', ':', ' ']:
        if ch in date:
            date = date.replace(ch,"")
            logging.info(f"Special character {ch} was replaced in the date")
    
    if len(date) == 14: # Probable str format %Y%m%d%H%M%S
        date = pd.to_datetime(date, format="%Y%m%d%H%M%S")
    elif len(date) == 12: # Probable str format %Y%m%d%H%M
        date = pd.to_datetime(date, format="%Y%m%d%H%M")
    elif len(date) == 10: # Probable str format %Y%m%d%H
        date = pd.to_datetime(date, format="%Y%m%d%H")
    elif len(date) == 8: # Probable str format %Y%m%d
        date = pd.to_datetime(date, format="%Y%m%d")
    elif len(date) == 7: # Probable str format %Y%J
        date =pd.to_datetime(date, format="%Y%j")
    else:
        raise exception_factory(ValueError, "Invalid date format") 

    return date


def monthly_dates(start_date, end_date, output_date_format="%Y%m%d") -> dict:
    """
    :params:
    - start_date: Start date of the period
    - end_date: End date of the period
    - date_format: Date format passed to datetime object. Examples %Y%m%d, %Y%j
    """

    month_start_dates = pd.date_range(start_date, end_date, freq="MS")
    month_end_dates = pd.date_range(start_date, end_date, freq="M")

    print(month_start_dates)
    print(month_end_dates)
    #assert len(month_start_dates) == len(month_end_dates)
    monthly_dates = {}
    for i, date in enumerate(month_start_dates):
        month_string = date.strftime("%Y%m")
        # Output daily values for each month in the dict, i.e. month: [days]
        monthly_dates[month_string] = pd.date_range(date, month_end_dates[i]).strftime(output_date_format)
        
        
    return monthly_dates



# Time Domain Selection function based off of xarray Will this work with dask too, a question for later
def time_domain_selection(
        time_values, 
        start_date, 
        end_date, 
        closed = None) -> np.array:
    """
    :param ds: one or several xarray datasets to cut down to correct size.
    :param start_date: the start date that we want to start with for our time domain selection
    :param end_date: the end date that we want to end with for our time domain seleciton
    :param time_column: the column for the time variables
    :param closed: like the pandas old param on the pandas date_range method (None, left, right), default:None
    :return: a cut down numpy array
    """

    # But the format of the time column may change aaaaaaaghh

    # For these if it is a string we need to know what the format is (maybe pandas figures that out)
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)

    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)

    
    # For the time column we can either centralise it in the seperate Instrument objects or pass it here,
    # I will start by passing it here

    def is_between_times(array):
        
        if closed == "left":
            return (array > start_date) & (array <= end_date)
        elif closed == "right":
            return (array >= start_date) & (array < end_date)
        else:
            return (array >= start_date) & (array <= end_date)

    time_values_index = np.where(is_between_times(time_values))[0] # np.where returns tuple
    
    time_values = time_values[time_values_index]

    return time_values, time_values_index


def inflate_array(var_array, shape1, shape2):
    """
    A function to inflate an array by repeating it along one dimension
    This is useful if you have a flattened array and need to repeat it due to the original array having less 
    dimensions that the target array
    """
    dif1 = np.setdiff1d(shape1, shape2)
    dif2 = np.setdiff1d(shape2, shape1)

    repeats = np.concatenate((dif1, dif2))

    return np.repeat(var_array, repeats)


def frequency_int(time_interval):
    """
    Convert frequency to an integer value, used in the DA
    """
    if time_interval == "H":
        frequency = 1
    else:
        frequency = int(time_interval[0])
    return frequency


def error_estimation(datatype:str, obs:np.typing.NDArray, unc_const:list) -> np.typing.NDArray:
    """
    Error estimation function
    """
    if datatype == "viirs":
        obserr = np.sqrt((unc_const[0]*obs+unc_const[1])**2 + unc_const[2]**2)
    elif datatype == "in-situ":
        # TODO: Need to work out how to do this for each different element
        #obserr = np.maximum(unc_const[e][1], obs*unc_const[e][0])
        pass
    return obserr




def geo_to_rot(lons, lats, centre_lon: float, centre_lat: float):
    """
    Rotating coordinates from cartesian lat/lon to rotated rlon/rlat
    """
    distance_lons = np.radians(lons - centre_lon)
    lons = np.radians(lons)
    lats = np.radians(lats)
    centre_lon = np.radians(centre_lon)
    centre_lat = np.radians(centre_lat)
    
    x = cos(centre_lat) * sin(lats) - sin(centre_lat) * cos(lats) * cos(distance_lons)
    y = cos(lats) * sin(distance_lons)
    z = cos(centre_lat) * cos(lats) * cos(distance_lons) + sin(centre_lat) * sin(lats)
    # Arctan2 used 
    # Explanation of the difference https://geo.libretexts.org/Courses/University_of_California_Davis/GEL_056%3A_Introduction_to_Geophysics/Geophysics_is_everywhere_in_geology.../zz%3A_Back_Matter/Arctan_vs_Arctan2
    rlon = np.arctan2(y, z)
    rlat = np.arcsin(x)
    #rlon[x < 0] += pi
    # Convert back to degrees
    rlon = np.degrees(rlon)
    rlat = np.degrees(rlat)
    return rlon, rlat



def array_calcs(data, info):
    """
    Array calculations for the values of each variable
    
    E.g. adding offset and scale factor and removing missing value

    params : 
    data - numpy array of data variables
    """
    # Fill value
    if info["missing_value"] != "":
        if info["missing_oper"] == "==":
            condition = data == info["missing_value"]
        elif info["missing_oper"] == ">=":
            condition = data >= info["missing_value"]
        elif info["missing_oper"] == "<=":
            condition = data <= info["missing_value"]
            
        data = np.where(condition, np.nan, data)
            
    # Add offset and scale factor
    data = data / info["scale_factor"] + info["offset"]
    
    return data