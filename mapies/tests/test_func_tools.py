#!/usr/bin/env python

import pytest
import pandas as pd
import numpy as np
from util.func_tools import time_converter, time_domain_selection, geo_to_rot



@pytest.mark.parametrize(
    "test_input,expected", 
    [
        ("2024079", pd.to_datetime("2024079", format="%Y%j")), 
        ("2024-06-07", pd.to_datetime("20240607", format="%Y%m%d")), 
        ("2024-06-(%$07", pd.to_datetime("20240607", format="%Y%m%d")),
        ("202406071411", pd.to_datetime("202406071411", format="%Y%m%d%H%M")),
    ]
)
def test_time_converter(test_input, expected):
    assert time_converter(test_input) == expected

def test_time_converter_error():
    with pytest.raises(ValueError):
        time_converter("20240607141")

@pytest.mark.parametrize(
    "test_input,expected",
    [
        ((0, 0, 20, 35), (-23.95680324, -32.61460715)),
        ((10, 10, 20, 35), (-10.82847335, -24.45866947)),
        ((-10, -10, 20, 35), (-39.42034458, -39.15567129)),
        ((-100, -50, 20, 35), (-141.61239392, -26.30586515))
    ]
)
def test_geo_to_rot(test_input, expected):
    assert (round(geo_to_rot(*test_input)[0], 8), round(geo_to_rot(*test_input)[1], 8))  == expected

"""
@pytest.mark.parametrize(
    "test_input,expected", 
    [
        ([], np.array([])),
    ]
)
def test_time_domain_selection(test_input, expected):
    assert time_domain_selection(test_input) == expected
"""