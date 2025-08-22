#!/usr/bin/env python3
"""
Earth Engine Mock Service for Development and Testing

This module provides mock Earth Engine functionality for development and testing
when actual Earth Engine authentication is not available. It simulates EE APIs
with realistic Uzbekistan satellite data patterns.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
from typing import Dict, List, Tuple, Optional, Any

class MockEEImage:
    """Mock Earth Engine Image class"""
    def __init__(self, data=None, bands=None):
        self.data = data or {}
        self.bands = bands or []
    
    def select(self, band_names):
        if isinstance(band_names, str):
            band_names = [band_names]
        return MockEEImage(self.data, band_names)
    
    def addBands(self, bands):
        new_bands = self.bands + (bands.bands if hasattr(bands, 'bands') else [])
        return MockEEImage(self.data, new_bands)
    
    def clip(self, geometry):
        return self
    
    def mean(self):
        return self
    
    def reduce(self, reducer):
        return self
    
    def multiply(self, factor):
        return self
    
    def add(self, value):
        return self
    
    def subtract(self, other):
        return self
    
    def normalizedDifference(self, bands):
        return MockEEImage(bands=bands)
    
    def expression(self, expr, variables):
        return MockEEImage()
    
    def gte(self, threshold):
        return self
    
    def gt(self, threshold):
        return self
    
    def eq(self, value):
        return self
    
    def rename(self, name):
        return MockEEImage(bands=[name])

class MockEEImageCollection:
    """Mock Earth Engine ImageCollection class"""
    def __init__(self, images=None):
        self.images = images or []
    
    def filterBounds(self, geometry):
        return self
    
    def filterDate(self, start, end):
        return self
    
    def filter(self, filter_obj):
        return self
    
    def map(self, func):
        return self
    
    def select(self, band_names):
        return self
    
    def mean(self):
        return MockEEImage()
    
    def sum(self):
        return MockEEImage()
    
    def first(self):
        return MockEEImage()
    
    def merge(self, other):
        return self

class MockEEGeometry:
    """Mock Earth Engine Geometry class"""
    def __init__(self, coords=None):
        self.coords = coords
    
    @staticmethod
    def Rectangle(coords):
        return MockEEGeometry(coords)

class MockEEFeatureCollection:
    """Mock Earth Engine FeatureCollection class"""
    def __init__(self, collection_id=None):
        self.collection_id = collection_id
    
    def filter(self, filter_obj):
        return self
    
    def geometry(self):
        return MockEEGeometry()

class MockEEFilter:
    """Mock Earth Engine Filter class"""
    @staticmethod
    def eq(property_name, value):
        return {'property': property_name, 'value': value}
    
    @staticmethod
    def lt(property_name, value):
        return {'property': property_name, 'value': value, 'op': 'lt'}

class MockEEReducer:
    """Mock Earth Engine Reducer class"""
    @staticmethod
    def stdDev():
        return 'stdDev'
    
    @staticmethod
    def linearFit():
        return 'linearFit'

class MockEEDate:
    """Mock Earth Engine Date class"""
    @staticmethod
    def fromYMD(year, month, day):
        return datetime(year, month, day)

class MockEEList:
    """Mock Earth Engine List class"""
    @staticmethod
    def sequence(start, end):
        return list(range(start, end + 1))

class MockEE:
    """Mock Earth Engine module"""
    
    Image = MockEEImage
    ImageCollection = MockEEImageCollection
    Geometry = MockEEGeometry
    FeatureCollection = MockEEFeatureCollection
    Filter = MockEEFilter
    Reducer = MockEEReducer
    Date = MockEEDate
    List = MockEEList
    
    @staticmethod
    def Initialize(*args, **kwargs):
        print("🔧 Mock Earth Engine initialized")
        pass
    
    @staticmethod
    def Authenticate():
        print("🔧 Mock Earth Engine authentication")
        pass

def generate_mock_uzbekistan_satellite_data() -> pd.DataFrame:
    """
    Generate realistic mock satellite data for Uzbekistan biodiversity analysis
    
    Returns:
        DataFrame with mock satellite observations
    """
    np.random.seed(42)  # For reproducibility
    
    # Uzbekistan geographic bounds
    min_lat, max_lat = 37.184, 45.573
    min_lon, max_lon = 55.998, 73.137
    
    # Generate sampling points across Uzbekistan
    n_points = 500
    
    # Create realistic spatial distribution
    lats = np.random.uniform(min_lat, max_lat, n_points)
    lons = np.random.uniform(min_lon, max_lon, n_points)
    
    # Assign regions based on approximate boundaries
    regions = []
    for lat, lon in zip(lats, lons):
        if lat > 43.0 and lon < 62.0:
            regions.append('Karakalpakstan')
        elif lat > 40.8 and lon > 68.5:
            regions.append('Tashkent')
        elif 39.2 < lat < 40.2 and 66.5 < lon < 67.5:
            regions.append('Samarkand')
        elif 39.5 < lat < 40.0 and 64.0 < lon < 65.0:
            regions.append('Bukhara')
        elif lat > 40.5 and lon > 70.5:
            regions.append('Namangan')
        elif lat < 39.0 and 67.0 < lon < 68.0:
            regions.append('Surkhandarya')
        elif 38.5 < lat < 39.2 and 65.5 < lon < 66.5:
            regions.append('Kashkadarya')
        elif 39.5 < lat < 40.5 and 65.0 < lon < 66.0:
            regions.append('Navoi')
        elif 39.8 < lat < 40.5 and 67.5 < lon < 68.5:
            regions.append('Jizzakh')
        elif 40.0 < lat < 40.8 and 68.0 < lon < 69.0:
            regions.append('Syrdarya')
        elif lat > 40.5 and 72.0 < lon < 73.0:
            regions.append('Andijan')
        elif lat > 40.0 and 71.5 < lon < 72.5:
            regions.append('Fergana')
        elif lat > 41.0 and lon < 61.0:
            regions.append('Khorezm')
        else:
            # Default assignment for edge cases
            regions.append(np.random.choice(['Karakalpakstan', 'Navoi', 'Kashkadarya']))
    
    # Generate time series data (2015-2024)
    dates = pd.date_range('2015-01-01', '2024-12-31', freq='16D')  # Landsat revisit
    
    data_records = []
    
    for i, (lat, lon, region) in enumerate(zip(lats, lons, regions)):
        # Regional characteristics for realistic patterns
        region_params = {
            'Karakalpakstan': {'base_ndvi': 0.15, 'variability': 0.1, 'trend': -0.002},  # Aral Sea region
            'Tashkent': {'base_ndvi': 0.35, 'variability': 0.15, 'trend': -0.001},      # Urban/agricultural
            'Samarkand': {'base_ndvi': 0.45, 'variability': 0.12, 'trend': 0.001},     # Agricultural
            'Bukhara': {'base_ndvi': 0.25, 'variability': 0.08, 'trend': -0.001},      # Oasis
            'Namangan': {'base_ndvi': 0.55, 'variability': 0.18, 'trend': 0.002},      # Fergana Valley
            'Surkhandarya': {'base_ndvi': 0.4, 'variability': 0.15, 'trend': 0.001},   # Southern
            'Kashkadarya': {'base_ndvi': 0.35, 'variability': 0.12, 'trend': 0.0},     # Central
            'Navoi': {'base_ndvi': 0.2, 'variability': 0.08, 'trend': -0.003},         # Mining/desert
            'Jizzakh': {'base_ndvi': 0.4, 'variability': 0.14, 'trend': 0.001},        # Agricultural
            'Syrdarya': {'base_ndvi': 0.45, 'variability': 0.16, 'trend': 0.002},      # River valley
            'Andijan': {'base_ndvi': 0.5, 'variability': 0.17, 'trend': 0.0},          # Fergana Valley
            'Fergana': {'base_ndvi': 0.48, 'variability': 0.16, 'trend': 0.001},       # Fergana Valley
            'Khorezm': {'base_ndvi': 0.35, 'variability': 0.13, 'trend': 0.0}          # Ancient irrigation
        }
        
        params = region_params.get(region, {'base_ndvi': 0.3, 'variability': 0.1, 'trend': 0.0})
        
        # Sample a subset of dates for this location
        location_dates = np.random.choice(dates, size=min(50, len(dates)), replace=False)
        
        for date in location_dates:
            # Calculate realistic indices
            year_progress = (date - dates[0]).days / 365.25
            day_of_year = date.timetuple().tm_yday if hasattr(date, 'timetuple') else pd.Timestamp(date).dayofyear
            seasonal_effect = 0.2 * np.sin(2 * np.pi * day_of_year / 365.25)
            trend_effect = params['trend'] * year_progress
            noise = np.random.normal(0, params['variability'])
            
            # NDVI with realistic bounds
            ndvi = np.clip(
                params['base_ndvi'] + seasonal_effect + trend_effect + noise,
                -0.1, 1.0
            )
            
            # EVI (Enhanced Vegetation Index) - correlated but different
            evi = np.clip(ndvi * 0.8 + np.random.normal(0, 0.05), -0.1, 1.0)
            
            # SAVI (Soil Adjusted Vegetation Index)
            savi = np.clip(ndvi * 1.2 + np.random.normal(0, 0.03), -0.1, 1.5)
            
            # NDWI (Normalized Difference Water Index)
            # Higher in irrigated areas and near water
            water_proximity = np.random.exponential(0.3)  # Distance effect
            ndwi = np.clip(0.1 + 0.3 * np.exp(-water_proximity) + np.random.normal(0, 0.1), -1.0, 1.0)
            
            # NDBI (Normalized Difference Built-up Index)
            # Higher in urban areas
            urban_factor = 1.0 if region == 'Tashkent' else 0.3
            ndbi = np.clip(urban_factor * 0.2 + np.random.normal(0, 0.05), -1.0, 1.0)
            
            # Land surface temperature (Kelvin)
            day_of_year = date.timetuple().tm_yday if hasattr(date, 'timetuple') else pd.Timestamp(date).dayofyear
            lst = 273.15 + 15 + 20 * np.sin(2 * np.pi * day_of_year / 365.25) + np.random.normal(0, 3)
            
            # Cloud cover
            cloud_cover = np.random.exponential(10)  # Low cloud cover bias
            
            data_records.append({
                'latitude': lat,
                'longitude': lon,
                'region': region,
                'date': date,
                'ndvi': ndvi,
                'evi': evi,
                'savi': savi,
                'ndwi': ndwi,
                'ndbi': ndbi,
                'lst': lst,
                'cloud_cover': min(cloud_cover, 100),
                'pixel_id': f"px_{i}_{pd.Timestamp(date).strftime('%Y%m%d')}"
            })
    
    return pd.DataFrame(data_records)

def generate_mock_land_cover_data() -> pd.DataFrame:
    """
    Generate mock land cover change data for Uzbekistan
    
    Returns:
        DataFrame with land cover change information
    """
    np.random.seed(42)
    
    # Define land cover classes (MODIS-style)
    lc_classes = {
        0: 'Water',
        1: 'Evergreen Needleleaf Forests',
        2: 'Evergreen Broadleaf Forests', 
        3: 'Deciduous Needleleaf Forests',
        4: 'Deciduous Broadleaf Forests',
        5: 'Mixed Forests',
        6: 'Closed Shrublands',
        7: 'Open Shrublands',
        8: 'Woody Savannas',
        9: 'Savannas',
        10: 'Grasslands',
        11: 'Permanent Wetlands',
        12: 'Croplands',
        13: 'Urban and Built-up Lands',
        14: 'Cropland/Natural Vegetation Mosaics',
        15: 'Snow and Ice',
        16: 'Barren'
    }
    
    # Typical Uzbekistan land cover distribution
    uzbekistan_lc_weights = {
        0: 0.02,   # Water (Aral Sea, rivers)
        6: 0.05,   # Closed Shrublands
        7: 0.15,   # Open Shrublands  
        10: 0.08,  # Grasslands
        12: 0.25,  # Croplands (major agriculture)
        13: 0.03,  # Urban and Built-up
        14: 0.12,  # Cropland/Natural Vegetation Mosaics
        16: 0.30   # Barren (deserts)
    }
    
    n_pixels = 1000
    years = [2015, 2020, 2023]
    
    records = []
    
    for pixel_id in range(n_pixels):
        # Random location in Uzbekistan
        lat = np.random.uniform(37.184, 45.573)
        lon = np.random.uniform(55.998, 73.137)
        
        # Initial land cover (2015)
        lc_2015 = np.random.choice(
            list(uzbekistan_lc_weights.keys()),
            p=list(uzbekistan_lc_weights.values())
        )
        
        # Simulate changes over time
        current_lc = lc_2015
        
        for year in years:
            # Probability of change (higher for certain classes)
            change_prob = {
                0: 0.05,   # Water - can dry up (Aral Sea)
                6: 0.10,   # Shrublands - moderate change
                7: 0.15,   # Open shrublands - higher change
                10: 0.12,  # Grasslands - moderate change
                12: 0.08,  # Croplands - stable
                13: 0.02,  # Urban - very stable
                14: 0.20,  # Mosaics - higher change
                16: 0.10   # Barren - moderate change
            }
            
            if np.random.random() < change_prob.get(current_lc, 0.1):
                # Land cover change occurred
                if current_lc == 0:  # Water drying
                    current_lc = 16  # Becomes barren
                elif current_lc in [6, 7, 10]:  # Natural vegetation
                    current_lc = np.random.choice([12, 14, 16])  # Agriculture or barren
                elif current_lc == 14:  # Mosaic
                    current_lc = np.random.choice([12, 16])  # Pure agriculture or barren
                elif current_lc == 16:  # Barren
                    current_lc = np.random.choice([7, 12])  # Shrubland or agriculture
            
            records.append({
                'pixel_id': pixel_id,
                'latitude': lat,
                'longitude': lon,
                'year': year,
                'land_cover': current_lc,
                'land_cover_name': lc_classes[current_lc]
            })
    
    return pd.DataFrame(records)

def generate_mock_disturbance_data() -> pd.DataFrame:
    """
    Generate mock disturbance event data for Uzbekistan
    
    Returns:
        DataFrame with disturbance events
    """
    np.random.seed(42)
    
    disturbance_types = [
        'Fire', 'Drought', 'Flood', 'Agricultural_Expansion', 
        'Urban_Expansion', 'Mining', 'Infrastructure', 'Grazing'
    ]
    
    # Realistic disturbance frequencies for Uzbekistan
    disturbance_weights = [0.05, 0.25, 0.10, 0.20, 0.08, 0.12, 0.10, 0.10]
    
    n_events = 200
    years = range(2015, 2025)
    
    records = []
    
    for event_id in range(n_events):
        # Random location
        lat = np.random.uniform(37.184, 45.573)
        lon = np.random.uniform(55.998, 73.137)
        
        # Random year and month
        year = np.random.choice(years)
        month = np.random.randint(1, 13)
        day = np.random.randint(1, 29)
        date = datetime(year, month, day)
        
        # Disturbance type
        dist_type = np.random.choice(disturbance_types, p=disturbance_weights)
        
        # Severity (0-1 scale)
        if dist_type == 'Drought':
            severity = np.random.beta(2, 2)  # Moderate severity typical
        elif dist_type == 'Fire':
            severity = np.random.exponential(0.3)  # Usually small fires
        else:
            severity = np.random.uniform(0.1, 0.8)
        
        severity = min(severity, 1.0)
        
        # Affected area (hectares)
        if dist_type in ['Urban_Expansion', 'Agricultural_Expansion']:
            area = np.random.lognormal(3, 1.5)  # Larger areas
        else:
            area = np.random.lognormal(2, 1)    # Smaller areas
        
        records.append({
            'event_id': event_id,
            'latitude': lat,
            'longitude': lon,
            'date': date,
            'year': year,
            'month': month,
            'disturbance_type': dist_type,
            'severity': severity,
            'affected_area_ha': area,
            'detected_by': 'satellite'
        })
    
    return pd.DataFrame(records)

# Create mock Earth Engine module for import
class MockEarthEngine:
    """Main mock Earth Engine module"""
    def __init__(self):
        self.ee = MockEE()
        self.satellite_data = generate_mock_uzbekistan_satellite_data()
        self.land_cover_data = generate_mock_land_cover_data()
        self.disturbance_data = generate_mock_disturbance_data()
    
    def get_satellite_data(self, region=None, start_date=None, end_date=None):
        """Get mock satellite data filtered by parameters"""
        data = self.satellite_data.copy()
        
        if region:
            data = data[data['region'] == region]
        
        if start_date:
            data = data[data['date'] >= pd.to_datetime(start_date)]
        
        if end_date:
            data = data[data['date'] <= pd.to_datetime(end_date)]
        
        return data
    
    def get_land_cover_data(self, year=None):
        """Get mock land cover data"""
        data = self.land_cover_data.copy()
        
        if year:
            data = data[data['year'] == year]
        
        return data
    
    def get_disturbance_data(self, dist_type=None, year=None):
        """Get mock disturbance data"""
        data = self.disturbance_data.copy()
        
        if dist_type:
            data = data[data['disturbance_type'] == dist_type]
        
        if year:
            data = data[data['year'] == year]
        
        return data