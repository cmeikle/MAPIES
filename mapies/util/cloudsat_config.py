#!/bin/env/python

VARIABLE_DICT = {
    "Profile_time":{"type":"FLOAT32", "dims":"nray", "units":"seconds", "valid_range":[0.0, 6000.0], "missing_value":"", "missing_oper":"", "scale_factor":1.0, "offset":0.0},
    "UTC_start":{"type":"FLOAT32", "dims":"scalar", "units":"seconds", "valid_range":[0.0, 86400.0], "missing_value":"", "missing_oper":"", "scale_factor":1.0, "offset":0.0},
    "TAI_start":{"type":"FLOAT64", "dims":"scalar", "units":"seconds", "valid_range":[0.0, 600000000.0], "missing_value":"", "missing_oper":"", "scale_factor":1.0, "offset":0.0},
    "Latitude":{"type":"FLOAT32", "dims":"nray", "units":"degrees", "valid_range":[-90.0, 90.0], "missing_value":"", "missing_oper":"", "scale_factor":1.0, "offset":0.0},
    "Longitude":{"type":"FLOAT32", "dims":"nray", "units":"degrees", "valid_range":[-180.0, 180.0], "missing_value":"", "missing_oper":"", "scale_factor":1.0, "offset":0.0},
    "Height":{"type":"INT16", "dims":["nray", "nbin"], "units":109, "valid_range":[-5000, 30000], "missing_value":-9999, "missing_oper":"==", "scale_factor":1.0, "offset":0.0},
    "Range_to_intercept":{"type":"FLOAT32", "dims":"nray", "units":"km", "valid_range":[600.0, 800.0], "missing_value":"", "missing_oper":"", "scale_factor":1.0, "offset":0.0},
    "DEM_elevation":{"type":"INT16", "dims":"nray", "units":109, "valid_range":[-9999, 8850], "missing_value":-9999, "missing_oper":"==", "scale_factor":1.0, "offset":0.0},
    "Vertical_binsize":{"type":"FLOAT32", "dims":"scalar", "units":109, "valid_range":"", "missing_value":-9999, "missing_oper":"==", "scale_factor":1.0, "offset":0.0},
    "Pitch_offset":{"type":"FLOAT32", "dims":"scalar", "units":"degrees", "valid_range":[-90.0, 90.0], "missing_value":"", "missing_oper":"", "scale_factor":1.0, "offset":0.0},
    "Roll_offset":{"type":"FLOAT32", "dims":"scalar", "units":"degrees", "valid_range":[-90.0, 90.0], "missing_value":"", "missing_oper":"", "scale_factor":1.0, "offset":0.0},
    "RO_liq_effective_radius":{"type":"INT16", "dims":["nray", "nbin"], "units":"um", "valid_range":[0, 10000], "missing_value":-3333, "missing_oper":"<=", "scale_factor":10.0, "offset":0.0},
    "RO_liq_effective_radius_uncertainty":{"type":"UINT8", "dims":["nray", "nbin"], "units":37, "valid_range":[0, 250], "missing_value":253, "missing_oper":">=", "scale_factor":1.0, "offset":0.0},
    "RO_ice_effective_radius":{"type":"INT16", "dims":["nray", "nbin"], "units":"um", "valid_range":[0, 30000], "missing_value":-3333, "missing_oper":"<=", "scale_factor":10.0, "offset":0.0},
    "RO_ice_effective_radius_uncertainty":{"type":"UINT8", "dims":["nray", "nbin"], "units":37, "valid_range":[0, 250], "missing_value":253, "missing_oper":">=", "scale_factor":1.0, "offset":0.0},
    "RO_liq_number_conc":{"type":"INT16", "dims":["nray", "nbin"], "units":"cm^{-3}", "valid_range":[0, 30000], "missing_value":-3333, "missing_oper":"<=", "scale_factor":10.0, "offset":0.0},
    "RO_liq_num_conc_uncertainty":{"type":"UINT8", "dims":["nray", "nbin"], "units":37, "valid_range":[0, 250], "missing_value":253, "missing_oper":">=", "scale_factor":1.0, "offset":0.0},
    "RO_ice_number_conc":{"type":"INT16", "dims":["nray", "nbin"], "units":"L^{-1}", "valid_range":[0, 30000], "missing_value":-3333, "missing_oper":"<=", "scale_factor":10.0, "offset":0.0},
    "RO_ice_num_conc_uncertainty":{"type":"UINT8", "dims":["nray", "nbin"], "units":37, "valid_range":[0, 250], "missing_value":253, "missing_oper":">=", "scale_factor":1.0, "offset":0.0},
    "RO_liq_water_content":{"type":"FLOAT32", "dims":["nray", "nbin"], "units":"mg m^{-3}", "valid_range":[0.0, 15000.0], "missing_value":-3333, "missing_oper":"<=", "scale_factor":1.0, "offset":0.0},
    "RO_liq_water_content_uncertainty":{"type":"UINT8", "dims":["nray", "nbin"], "units":37, "valid_range":[0, 250], "missing_value":253, "missing_oper":">=", "scale_factor":1.0, "offset":0.0},
    "RO_ice_water_content":{"type":"FLOAT32", "dims":["nray", "nbin"], "units":"mg m^{-3}", "valid_range":[0.0, 15000.0], "missing_value":-3333, "missing_oper":"<=", "scale_factor":1.0, "offset":0.0},
    "RO_ice_water_content_uncertainty":{"type":"UINT8", "dims":["nray", "nbin"], "units":37, "valid_range":[0, 250], "missing_value":253, "missing_oper":">=", "scale_factor":1.0, "offset":0.0},
    "RO_ice_phase_fraction":{"type":"INT16", "dims":["nray", "nbin"], "units":"-", "valid_range":[0, 1000], "missing_value":-3333, "missing_oper":"<=", "scale_factor":1000.0, "offset":0.0},
    "RO_radar_uncertainty":{"type":"INT16","dims":["nray", "nbin"], "units":"dBZ", "valid_range":[0, 10000], "missing_value":-7777, "missing_oper":"<=", "scale_factor":100.0, "offset":0.0},
    "RO_liq_water_path":{"type":"FLOAT32", "dims":"nray", "units":"g m^{-2}", "valid_range":[0.0, 15000.0], "missing_value":-3333, "missing_oper":"<=", "scale_factor":1.0, "offset":0.0},
    "RO_liq_water_path_uncertainty":{"type":"UINT8", "dims":"nray", "units":37, "valid_range":[0, 250], "missing_value":253, "missing_oper":">=", "scale_factor":1.0, "offset":0.0},
    "RO_ice_water_path":{"type":"FLOAT32", "dims":"nray", "units":"g m^{-2}", "valid_range":[0.0, 15000.0], "missing_value":-3333, "missing_oper":"<=", "scale_factor":1.0, "offset":0.0},
    "RO_ice_water_path_uncertainty":{"type":"UINT8", "dims":"nray", "units":37, "valid_range":[0, 250], "missing_value":253, "missing_oper":">=", "scale_factor":1.0, "offset":0.0},
    "RO_liq_distrib_width_param":{"type":"INT16", "dims":["nray", "nbin"], "units":"-", "valid_range":[0, 5000], "missing_value":-3333, "missing_oper":"<=", "scale_factor":1000.0, "offset":0.0},
    "RO_liq_distrib_width_param_uncertainty":{"type":"UINT8", "dims":["nray", "nbin"], "units":37, "valid_range":[0, 250], "missing_value":253, "missing_oper":">=", "scale_factor":1.0, "offset":0.0},
    "RO_ice_distrib_width_param":{"type":"INT16", "dims":["nray", "nbin"], "units":"-", "valid_range":[0, 5000], "missing_value":-3333, "missing_oper":"<=", "scale_factor":1000.0, "offset":0.0},
    "RO_ice_distrib_width_param_uncertainty":{"type":"UINT8", "dims":["nray", "nbin"], "units":37, "valid_range":[0, 250], "missing_value":253, "missing_oper":">=", "scale_factor":1.0, "offset":0.0},
    "Data_quality":{"type":"UINT8", "dims":"nray", "units":"-", "valid_range":[0, -1], "missing_value":"", "missing_oper":"", "scale_factor":1.0, "offset":0.0},
    "Data_status":{"type":"UINT8", "dims":"nray", "units":"-", "valid_range":[0, -1], "missing_value":"", "missing_oper":"", "scale_factor":1.0, "offset":0.0},
    "Data_targetID":{"type":"UINT8", "dims":"nray", "units":"-", "valid_range":[0, -1], "missing_value":"", "missing_oper":"", "scale_factor":1.0, "offset":0.0},
    "RayStatus_validity":{"type":"UINT8", "dims":"nray", "units":"-", "valid_range":[0, -1], "missing_value":"", "missing_oper":"", "scale_factor":1.0, "offset":0.0},
    "Navigation_land_sea_flag":{"type":"UINT8", "dims":"nray", "units":"-", "valid_range":[1, 4], "missing_value":"", "missing_oper":"", "scale_factor":1.0, "offset":0.0},
    "RO_CWC_status":{"type":"INT16", "dims":"nray", "units":"-", "valid_range":[-32768, 32767], "missing_value":"", "missing_oper":"", "scale_factor":1.0, "offset":0.0},
    "N_cloudy_bins":{"type":"INT16", "dims":"nray", "units":"-", "valid_range":[0, 125], "missing_value":77, "missing_oper":"<=", "scale_factor":1.0, "offset":0.0},
    "Cloud_mask_threshold":{"type":"INT8", "dims":"scalar", "units":"-", "valid_range":[0, 40], "missing_value":99, "missing_oper":">=", "scale_factor":1.0, "offset":0.0},
    "N_ray_granule":{"type":"INT32", "dims":"scalar", "units":"", "valid_range":"", "missing_value":"", "missing_oper":"", "scale_factor":1.0, "offset":0.0},
    "N_indeterminate_scenario":{"type":"INT32", "dims":"scalar", "units":"", "valid_range":"", "missing_value":"", "missing_oper":"", "scale_factor":1.0, "offset":0.0},
    "PCT_indeterminate_scenario":{"type":"INT8", "dims":"scalar", "units":"", "valid_range":"", "missing_value":"", "missing_oper":"", "scale_factor":1.0, "offset":0.0},
    "N_invalid_CLDCLASS":{"type":"INT32", "dims":"scalar", "units":"", "valid_range":"", "missing_value":"", "missing_oper":"", "scale_factor":1.0, "offset":0.0},
    "PCT_invalid_CLDCLASS":{"type":"INT8", "dims":"scalar", "units":"", "valid_range":"", "missing_value":"", "missing_oper":"", "scale_factor":1.0, "offset":0.0},
    "N_low_confidence_CLDCLASS":{"type":"INT32", "dims":"scalar", "units":"", "valid_range":"", "missing_value":"", "missing_oper":"", "scale_factor":1.0, "offset":0.0},
    "PCT_low_confidence_CLDCLASS":{"type":"INT8", "dims":"scalar", "units":"", "valid_range":"", "missing_value":"", "missing_oper":"", "scale_factor":1.0, "offset":0.0},
    "N_invalid_temperature":{"type":"INT32", "dims":"scalar", "units":"", "valid_range":"", "missing_value":"", "missing_oper":"", "scale_factor":1.0, "offset":0.0},
    "PCT_invalid_temperature":{"type":"INT8", "dims":"scalar", "units":"", "valid_range":"", "missing_value":"", "missing_oper":"", "scale_factor":1.0, "offset":0.0},
    "N_clear_CPR":{"type":"INT32", "dims":"scalar", "units":"", "valid_range":"", "missing_value":"", "missing_oper":"", "scale_factor":1.0, "offset":0.0},
    "PCT_clear_CPR":{"type":"INT8", "dims":"scalar", "units":"", "valid_range":"", "missing_value":"", "missing_oper":"", "scale_factor":1.0, "offset":0.0},
    "N_unphys_missing_reflectivity":{"type":"INT32", "dims":"scalar", "units":"", "valid_range":"", "missing_value":"", "missing_oper":"", "scale_factor":1.0, "offset":0.0},
    "PCT_unphys_missing_reflectivity":{"type":"INT8", "dims":"scalar", "units":"", "valid_range":"", "missing_value":"", "missing_oper":"", "scale_factor":1.0, "offset":0.0},
    "N_drizzle_or_precipitation":{"type":"INT32", "dims":"scalar", "units":"", "valid_range":"", "missing_value":"", "missing_oper":"", "scale_factor":1.0, "offset":0.0},
    "PCT_drizzle_or_precipitation":{"type":"INT8", "dims":"scalar", "units":"", "valid_range":"", "missing_value":"", "missing_oper":"", "scale_factor":1.0, "offset":0.0},
    "Temp_min_mixph_K":{"type":"FLOAT32", "dims":"scalar", "units":75, "valid_range":[2.33e+02, 2.73e+02], "missing_value":"", "missing_oper":"", "scale_factor":1.0, "offset":0.0},
    "Temp_max_mixph_K":{"type":"FLOAT32", "dims":"scalar", "units":75, "valid_range":[2.33e+02, 2.73e+02], "missing_value":"", "missing_oper":"", "scale_factor":1.0, "offset":0.0},
}