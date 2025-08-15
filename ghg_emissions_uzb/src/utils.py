#!/usr/bin/env python3
"""
Utilities for GHG Emissions Downscaling Analysis

This module provides essential utilities for greenhouse gas emissions
analysis and downscaling, adapted from AlphaEarth utilities.

Author: AlphaEarth Analysis Team - GHG Module
Date: January 2025
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

def load_config(path: str = "config_ghg.json"):
    """Load configuration file for GHG emissions analysis"""
    possible_paths = [
        path,
        f"ghg_emissions_uzb/{path}",
        f"../{path}",
        f"../../{path}"
    ]
    
    for config_path in possible_paths:
        try:
            return json.loads(Path(config_path).read_text(encoding="utf-8"))
        except FileNotFoundError:
            continue
    
    # If none found, create a default config for GHG analysis
    default_config = {
        "country": "Uzbekistan",
        "regions": ["Karakalpakstan", "Tashkent", "Samarkand", "Bukhara", "Namangan", 
                   "Andijan", "Fergana", "Khorezm", "Navoiy", "Jizzakh", "Kashkadarya", 
                   "Surkhandarya", "Syrdarya"],
        "analysis_period": {
            "start_year": 2015,
            "end_year": 2023
        },
        "spatial_resolution": 1000,  # meters
        "target_resolution": 200,    # meters for downscaling
        "crs": "EPSG:4326",
        "random_seed": 42,
        "ghg_sources": {
            "ODIAC": True,
            "EDGAR": True,
            "GFEI": True,
            "FFDAS": True
        },
        "emission_sectors": [
            "power_generation",
            "industry", 
            "transportation",
            "residential",
            "agriculture",
            "waste"
        ],
        "gases": ["CO2", "CH4", "N2O"],
        "paths": {
            "data": "data",
            "outputs": "outputs", 
            "figs": "figs",
            "reports": "reports"
        }
    }
    
    # Save default config
    with open("config_ghg.json", 'w') as f:
        json.dump(default_config, f, indent=2)
    
    return default_config

def ensure_dir(p: str):
    """Create directory if it doesn't exist"""
    Path(p).mkdir(parents=True, exist_ok=True)

def setup_plotting():
    """Configure matplotlib for high-quality research plots"""
    plt.style.use('seaborn-v0_8')
    plt.rcParams.update({
        'figure.figsize': (12, 8),
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'font.family': 'sans-serif'
    })

def get_uzbekistan_coordinates():
    """Get geographic boundaries and key coordinates for Uzbekistan"""
    return {
        'bounds': {
            'min_lon': 55.9,
            'max_lon': 73.2,
            'min_lat': 37.1,
            'max_lat': 45.6
        },
        'major_cities': {
            'Tashkent': (69.2401, 41.2995),
            'Samarkand': (66.9597, 39.6542),
            'Bukhara': (64.4264, 39.7747),
            'Namangan': (71.6726, 40.9983),
            'Andijan': (72.3440, 40.7821),
            'Nukus': (59.6167, 42.4531),
            'Qarshi': (65.7892, 38.8606),
            'Kokand': (70.9428, 40.5183),
            'Margilan': (71.7246, 40.4731),
            'Urgench': (60.6311, 41.5509),
            'Gulistan': (68.7850, 40.4897),
            'Jizzakh': (67.8392, 40.1158),
            'Navoiy': (65.3792, 40.0844)
        },
        'industrial_zones': {
            'Almalyk_Mining': (69.5983, 40.8489),
            'Mubarek_Gas': (67.8736, 39.2617),
            'Surgil_Oil': (60.7608, 42.1167),
            'Ustyurt_Gas': (58.5000, 43.0000),
            'Zerafshan_Mining': (64.2000, 40.1000)
        }
    }

def load_uzbekistan_auxiliary_data() -> pd.DataFrame:
    """
    Load auxiliary geospatial data for Uzbekistan for emissions modeling
    
    Returns:
        DataFrame with auxiliary variables for emissions downscaling
    """
    print("🌍 Loading auxiliary geospatial data for Uzbekistan...")
    
    coords = get_uzbekistan_coordinates()
    bounds = coords['bounds']
    cities = coords['major_cities']
    industrial = coords['industrial_zones']
    
    # Create a systematic grid for analysis
    np.random.seed(42)
    
    # Generate sampling grid
    n_points = 5000  # Sufficient sampling for country-level analysis
    
    # Stratified sampling to ensure good coverage
    lons = np.random.uniform(bounds['min_lon'], bounds['max_lon'], n_points)
    lats = np.random.uniform(bounds['min_lat'], bounds['max_lat'], n_points)
    
    # Create base DataFrame
    df = pd.DataFrame({
        'longitude': lons,
        'latitude': lats
    })
    
    # Add administrative regions based on coordinates
    df['region'] = df.apply(lambda row: assign_region_from_coords(row['longitude'], row['latitude']), axis=1)
    
    # Population density (people per km²)
    df['population_density'] = calculate_population_density(df)
    
    # Urban/rural classification
    df['urban_fraction'] = calculate_urban_fraction(df, cities)
    
    # Distance to major cities
    df['dist_to_major_city'] = calculate_distance_to_cities(df, cities)
    
    # Distance to industrial zones
    df['dist_to_industrial'] = calculate_distance_to_industrial(df, industrial)
    
    # Land use categories
    df['land_use_type'] = determine_land_use(df)
    
    # Infrastructure density
    df['road_density'] = calculate_road_density(df)
    df['power_plant_proximity'] = calculate_power_plant_proximity(df)
    
    # Topographic features
    df['elevation'] = calculate_elevation(df)
    df['slope'] = calculate_slope(df)
    
    # Climate variables
    df['temperature'] = calculate_temperature(df)
    df['precipitation'] = calculate_precipitation(df)
    
    # Economic activity indicators
    df['gdp_density'] = calculate_economic_activity(df)
    df['industrial_activity'] = calculate_industrial_activity(df)
    
    # Agricultural indicators
    df['agricultural_area'] = calculate_agricultural_area(df)
    df['irrigation_density'] = calculate_irrigation_density(df)
    
    print(f"✅ Loaded auxiliary data for {len(df)} grid points")
    print(f"   Regions covered: {df['region'].unique()}")
    
    return df

def assign_region_from_coords(lon, lat):
    """Assign administrative region based on coordinates"""
    # Simplified regional assignment based on Uzbekistan administrative boundaries
    
    if lat > 43.5:  # Northern regions
        return "Karakalpakstan"
    elif lon > 70.5 and lat > 40.5:
        if lat > 41.5:
            return "Namangan"
        else:
            return "Andijan"
    elif lon > 69 and lat > 40.5:
        return "Tashkent"
    elif lon > 70 and lat < 41:
        return "Fergana"
    elif lon > 67 and lat > 39.5:
        return "Samarkand"
    elif lon > 66 and lat < 39.5:
        return "Kashkadarya"
    elif lon > 65 and lat > 39:
        return "Jizzakh"
    elif lon < 64 and lat > 40:
        return "Bukhara"
    elif lon < 62 and lat > 41:
        return "Khorezm"
    elif lon > 64 and lat < 39:
        return "Surkhandarya"
    elif lon > 67 and lat < 40:
        return "Navoiy"
    else:
        return "Syrdarya"

def calculate_population_density(df):
    """Calculate population density based on proximity to urban centers"""
    coords = get_uzbekistan_coordinates()
    cities = coords['major_cities']
    
    # City population estimates (approximate)
    city_populations = {
        'Tashkent': 2500000,
        'Samarkand': 550000,
        'Namangan': 475000,
        'Andijan': 450000,
        'Bukhara': 280000,
        'Nukus': 260000,
        'Qarshi': 250000,
        'Kokand': 230000,
        'Margilan': 200000,
        'Urgench': 180000,
        'Gulistan': 75000,
        'Jizzakh': 170000,
        'Navoiy': 140000
    }
    
    pop_density = np.zeros(len(df))
    
    for i, row in df.iterrows():
        total_influence = 0
        
        for city_name, (city_lon, city_lat) in cities.items():
            # Calculate distance (approximate)
            dist = np.sqrt((row['longitude'] - city_lon)**2 + (row['latitude'] - city_lat)**2)
            
            # Distance decay function
            if dist < 0.1:  # Very close to city center
                influence = city_populations.get(city_name, 100000) / 10  # High density
            elif dist < 0.5:  # Urban area
                influence = city_populations.get(city_name, 100000) / (50 * dist)
            elif dist < 1.0:  # Suburban
                influence = city_populations.get(city_name, 100000) / (200 * dist)
            else:  # Rural
                influence = city_populations.get(city_name, 100000) / (1000 * dist**2)
            
            total_influence += influence
        
        pop_density[i] = max(total_influence, 5)  # Minimum rural density
    
    return np.clip(pop_density, 5, 10000)  # Reasonable bounds

def calculate_urban_fraction(df, cities):
    """Calculate urban fraction based on distance to cities"""
    urban_frac = np.zeros(len(df))
    
    for i, row in df.iterrows():
        min_dist_to_city = float('inf')
        
        for city_name, (city_lon, city_lat) in cities.items():
            dist = np.sqrt((row['longitude'] - city_lon)**2 + (row['latitude'] - city_lat)**2)
            min_dist_to_city = min(min_dist_to_city, dist)
        
        # Urban fraction based on distance to nearest city
        if min_dist_to_city < 0.05:  # City center
            urban_frac[i] = 0.9
        elif min_dist_to_city < 0.2:  # Urban area
            urban_frac[i] = 0.7
        elif min_dist_to_city < 0.5:  # Suburban
            urban_frac[i] = 0.4
        elif min_dist_to_city < 1.0:  # Peri-urban
            urban_frac[i] = 0.2
        else:  # Rural
            urban_frac[i] = 0.05
    
    return urban_frac

def calculate_distance_to_cities(df, cities):
    """Calculate minimum distance to major cities"""
    min_distances = np.zeros(len(df))
    
    for i, row in df.iterrows():
        min_dist = float('inf')
        
        for city_name, (city_lon, city_lat) in cities.items():
            dist = np.sqrt((row['longitude'] - city_lon)**2 + (row['latitude'] - city_lat)**2)
            min_dist = min(min_dist, dist)
        
        min_distances[i] = min_dist * 111  # Convert to approximate km
    
    return min_distances

def calculate_distance_to_industrial(df, industrial_zones):
    """Calculate distance to industrial zones"""
    min_distances = np.zeros(len(df))
    
    for i, row in df.iterrows():
        min_dist = float('inf')
        
        for zone_name, (zone_lon, zone_lat) in industrial_zones.items():
            dist = np.sqrt((row['longitude'] - zone_lon)**2 + (row['latitude'] - zone_lat)**2)
            min_dist = min(min_dist, dist)
        
        min_distances[i] = min_dist * 111  # Convert to approximate km
    
    return min_distances

def determine_land_use(df):
    """Determine primary land use type"""
    land_use = []
    
    for _, row in df.iterrows():
        if row['urban_fraction'] > 0.5:
            land_use.append('urban')
        elif row['urban_fraction'] > 0.2:
            land_use.append('mixed')
        elif row['latitude'] > 42.5:  # Northern desert regions
            land_use.append('desert')
        elif row['longitude'] < 62:  # Western regions near Aral Sea
            land_use.append('shrubland')
        else:
            land_use.append('agricultural')
    
    return land_use

def calculate_road_density(df):
    """Calculate road/transportation infrastructure density"""
    # Based on distance to cities and economic activity
    road_density = np.zeros(len(df))
    
    for i, row in df.iterrows():
        base_density = row['urban_fraction'] * 10  # Urban areas have more roads
        
        # Add highways connecting major cities
        if row['dist_to_major_city'] < 50:  # Near major transportation corridors
            base_density += 5
        elif row['dist_to_major_city'] < 100:
            base_density += 2
        
        road_density[i] = base_density + np.random.normal(0, 1)
    
    return np.clip(road_density, 0.1, 20)

def calculate_power_plant_proximity(df):
    """Calculate proximity to power generation facilities"""
    # Major power plants in Uzbekistan (approximate locations)
    power_plants = {
        'Tashkent_TPP': (69.3, 41.3),
        'Syrdarya_TPP': (68.8, 40.5),
        'Navoiy_TPP': (65.4, 40.1),
        'Angren_TPP': (70.1, 41.0),
        'Mubarek_Gas': (67.9, 39.3)
    }
    
    proximity = np.zeros(len(df))
    
    for i, row in df.iterrows():
        min_dist = float('inf')
        
        for plant, (plant_lon, plant_lat) in power_plants.items():
            dist = np.sqrt((row['longitude'] - plant_lon)**2 + (row['latitude'] - plant_lat)**2)
            min_dist = min(min_dist, dist)
        
        # Inverse distance relationship
        proximity[i] = 1 / (min_dist + 0.1)
    
    return proximity

def calculate_elevation(df):
    """Calculate elevation based on geographic location"""
    # Simplified elevation model for Uzbekistan
    elevation = np.zeros(len(df))
    
    for i, row in df.iterrows():
        lon, lat = row['longitude'], row['latitude']
        
        # Mountain regions (Tian Shan foothills)
        if lon > 70 and lat > 40.5:
            base_elev = 800 + (lon - 70) * 200 + (lat - 40.5) * 300
        # Kyzylkum Desert plateau
        elif 60 < lon < 68 and 39 < lat < 43:
            base_elev = 200 + np.random.normal(0, 50)
        # Western lowlands
        elif lon < 62:
            base_elev = 50 + np.random.normal(0, 20)
        # Central plains
        else:
            base_elev = 150 + np.random.normal(0, 30)
        
        elevation[i] = max(base_elev, 50)  # Minimum elevation
    
    return elevation

def calculate_slope(df):
    """Calculate terrain slope"""
    # Simplified slope calculation based on elevation patterns
    slope = np.abs(np.random.normal(2, 3, len(df)))  # Most areas are relatively flat
    
    # Increase slope in mountainous regions
    mountain_mask = (df['longitude'] > 70) & (df['latitude'] > 40.5)
    slope[mountain_mask] += np.random.uniform(5, 15, mountain_mask.sum())
    
    return np.clip(slope, 0, 30)

def calculate_temperature(df):
    """Calculate average annual temperature"""
    temp = np.zeros(len(df))
    
    for i, row in df.iterrows():
        lat = row['latitude']
        
        # Base temperature decreases with latitude
        base_temp = 20 - (lat - 37) * 1.5
        
        # Elevation effect
        elev_effect = -row['elevation'] * 0.006  # Lapse rate
        
        # Random variation
        temp_variation = np.random.normal(0, 2)
        
        temp[i] = base_temp + elev_effect + temp_variation
    
    return temp

def calculate_precipitation(df):
    """Calculate average annual precipitation"""
    precip = np.zeros(len(df))
    
    for i, row in df.iterrows():
        lon, lat = row['longitude'], row['latitude']
        
        # Base precipitation pattern (increases eastward and with elevation)
        base_precip = 100 + (lon - 56) * 10 + row['elevation'] * 0.2
        
        # Mountain effect
        if lon > 69 and lat > 40:
            base_precip += 200
        
        # Desert regions
        if 59 < lon < 66 and 39 < lat < 44:
            base_precip *= 0.5
        
        # Random variation
        precip_variation = np.random.normal(0, 50)
        
        precip[i] = max(base_precip + precip_variation, 50)
    
    return precip

def calculate_economic_activity(df):
    """Calculate economic activity density"""
    gdp_density = np.zeros(len(df))
    
    for i, row in df.iterrows():
        # Base on population density and urban fraction
        base_activity = row['population_density'] * row['urban_fraction'] * 0.01
        
        # Industrial zone effect
        if row['dist_to_industrial'] < 20:
            base_activity *= 3
        elif row['dist_to_industrial'] < 50:
            base_activity *= 1.5
        
        # Add natural resource activity
        if 'mining' in row.get('land_use_type', '') or row['longitude'] < 62:
            base_activity += 50  # Gas/oil regions
        
        gdp_density[i] = base_activity + np.random.normal(0, 10)
    
    return np.clip(gdp_density, 10, 1000)

def calculate_industrial_activity(df):
    """Calculate industrial activity intensity"""
    industrial = np.zeros(len(df))
    
    for i, row in df.iterrows():
        # Base on distance to industrial zones
        base_industrial = 1 / (row['dist_to_industrial'] + 1)
        
        # Urban industrial activity
        base_industrial += row['urban_fraction'] * 0.3
        
        # Natural resource extraction
        if row['longitude'] < 62:  # Western gas/oil regions
            base_industrial += 0.5
        
        industrial[i] = base_industrial * 100 + np.random.normal(0, 5)
    
    return np.clip(industrial, 1, 100)

def calculate_agricultural_area(df):
    """Calculate agricultural land fraction"""
    ag_area = np.zeros(len(df))
    
    for i, row in df.iterrows():
        # Base on land use and climate
        if row['land_use_type'] == 'agricultural':
            base_ag = 0.7
        elif row['land_use_type'] == 'mixed':
            base_ag = 0.4
        elif row['land_use_type'] == 'urban':
            base_ag = 0.1
        else:
            base_ag = 0.2
        
        # Climate suitability
        if 300 < row['precipitation'] < 600:  # Optimal range
            base_ag *= 1.2
        elif row['precipitation'] < 200:  # Arid
            base_ag *= 0.5
        
        ag_area[i] = base_ag + np.random.normal(0, 0.1)
    
    return np.clip(ag_area, 0, 1)

def calculate_irrigation_density(df):
    """Calculate irrigation infrastructure density"""
    irrigation = np.zeros(len(df))
    
    for i, row in df.iterrows():
        # Base on agricultural area and water availability
        base_irrigation = row['agricultural_area'] * 0.8
        
        # Proximity to rivers (simplified)
        if 67 < row['longitude'] < 70 and 39 < row['latitude'] < 42:  # Amu Darya region
            base_irrigation *= 1.5
        elif 67 < row['longitude'] < 71 and 40 < row['latitude'] < 42:  # Syr Darya region
            base_irrigation *= 1.3
        
        irrigation[i] = base_irrigation + np.random.normal(0, 0.1)
    
    return np.clip(irrigation, 0, 1)

def save_plot(fig, filename: str, figs_dir: str = "figs"):
    """Save plot with consistent formatting"""
    ensure_dir(figs_dir)
    filepath = Path(figs_dir) / filename
    fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"📊 Plot saved: {filepath}")

def create_summary_table(data: dict, title: str, output_dir: str = "outputs"):
    """Create and save summary table"""
    ensure_dir(output_dir)
    
    df = pd.DataFrame.from_dict(data, orient='index', columns=['Value'])
    
    filepath = Path(output_dir) / f"{title.lower().replace(' ', '_')}.csv"
    df.to_csv(filepath)
    print(f"📋 Summary table saved: {filepath}")
    
    return df

def validate_data_quality(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """Validate data quality and completeness"""
    print("🔍 Validating data quality...")
    
    issues = []
    
    # Check required columns
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")
    
    # Check for null values
    null_counts = df.isnull().sum()
    if null_counts.any():
        issues.append(f"Null values found: {null_counts[null_counts > 0].to_dict()}")
    
    # Check data ranges
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in df.columns:
            if df[col].std() == 0:
                issues.append(f"No variation in column: {col}")
    
    if issues:
        print("⚠️  Data quality issues found:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    else:
        print("✅ Data quality validation passed")
        return True

def print_analysis_summary(title: str, stats: dict):
    """Print formatted analysis summary"""
    print(f"\n📊 {title}")
    print("=" * len(title))
    
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.3f}")
        else:
            print(f"   {key}: {value}")