#!/usr/bin/env python3
"""
Real Atmospheric Data GHG Analysis for Uzbekistan

This script uses real Sentinel-5P atmospheric concentration data
instead of simulated emissions inventories.

Uses only real satellite data:
- Sentinel-5P CO, NO2, CH4, SO2 concentrations
- MODIS land surface temperature
- MODIS vegetation indices
- MODIS land cover

Author: AlphaEarth Analysis Team
Date: August 15, 2025
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Google Earth Engine
try:
    import ee
    GEE_AVAILABLE = True
except ImportError:
    GEE_AVAILABLE = False
    print("❌ Google Earth Engine not available")
    sys.exit(1)

class RealAtmosphericDataAnalyzer:
    """Analyzes real atmospheric concentration data for Uzbekistan"""
    
    def __init__(self):
        self.project_id = 'ee-sabitovty'
        self.uzbekistan_bounds = None  # Will be set after GEE initialization
        self.gee_initialized = False
        self.atmospheric_data = {}
        self.auxiliary_data = {}
        
    def initialize_gee(self):
        """Initialize Google Earth Engine with project"""
        try:
            ee.Initialize(project=self.project_id)
            print(f"✅ Google Earth Engine initialized with project: {self.project_id}")
            
            # Now create geometry after initialization
            self.uzbekistan_bounds = ee.Geometry.Rectangle([55.9, 37.2, 73.2, 45.6])
            
            self.gee_initialized = True
            return True
        except Exception as e:
            print(f"❌ GEE initialization failed: {e}")
            return False
    
    def load_real_atmospheric_data(self, year=2023):
        """Load real Sentinel-5P atmospheric concentration data"""
        
        if not self.gee_initialized:
            print("❌ GEE not initialized")
            return False
        
        print(f"\n🛰️  Loading Real Atmospheric Data for {year}")
        print("=" * 50)
        
        # Date range for the year
        start_date = f'{year}-01-01'
        end_date = f'{year}-12-31'
        
        # Define real atmospheric datasets
        datasets = {
            'CO': 'COPERNICUS/S5P/OFFL/L3_CO',
            'NO2': 'COPERNICUS/S5P/OFFL/L3_NO2', 
            'CH4': 'COPERNICUS/S5P/OFFL/L3_CH4',
            'SO2': 'COPERNICUS/S5P/NRTI/L3_SO2'
        }
        
        # Band names for each gas
        bands = {
            'CO': 'CO_column_number_density',
            'NO2': 'tropospheric_NO2_column_number_density',
            'CH4': 'CH4_column_volume_mixing_ratio_dry_air',
            'SO2': 'SO2_column_number_density'
        }
        
        for gas, dataset in datasets.items():
            try:
                print(f"📡 Loading {gas} concentrations from {dataset}...")
                
                # Load and filter the collection
                collection = ee.ImageCollection(dataset) \
                    .filterDate(start_date, end_date) \
                    .filterBounds(self.uzbekistan_bounds) \
                    .select(bands[gas])
                
                # Get collection size
                size = collection.size().getInfo()
                print(f"   Found {size} images for {gas}")
                
                if size > 0:
                    # Calculate mean concentration for the year
                    mean_concentration = collection.mean()
                    
                    # Sample points across Uzbekistan
                    sample_points = self.generate_sampling_grid()
                    
                    # Extract values at sample points
                    sampled_data = mean_concentration.sampleRegions(
                        collection=sample_points,
                        scale=1000,  # 1km resolution
                        projection='EPSG:4326'
                    )
                    
                    # Convert to DataFrame
                    features = sampled_data.getInfo()['features']
                    data_list = []
                    
                    for feature in features:
                        props = feature['properties']
                        coords = feature['geometry']['coordinates']
                        
                        data_point = {
                            'longitude': coords[0],
                            'latitude': coords[1],
                            f'{gas}_concentration': props.get(bands[gas], None),
                            'year': year
                        }
                        data_list.append(data_point)
                    
                    df = pd.DataFrame(data_list)
                    # Remove null values
                    df = df.dropna(subset=[f'{gas}_concentration'])
                    
                    self.atmospheric_data[gas] = df
                    print(f"✅ Loaded {len(df)} valid {gas} concentration measurements")
                    
                else:
                    print(f"⚠️  No {gas} data available for {year}")
                    
            except Exception as e:
                print(f"❌ Error loading {gas} data: {e}")
        
        return len(self.atmospheric_data) > 0
    
    def generate_sampling_grid(self, resolution=0.01):
        """Generate regular sampling grid over Uzbekistan"""
        
        # Uzbekistan bounding box
        min_lon, min_lat, max_lon, max_lat = 55.9, 37.2, 73.2, 45.6
        
        # Create regular grid
        lons = np.arange(min_lon, max_lon, resolution)
        lats = np.arange(min_lat, max_lat, resolution)
        
        # Create point features
        points = []
        for lon in lons:
            for lat in lats:
                point = ee.Feature(ee.Geometry.Point([lon, lat]), {
                    'longitude': lon,
                    'latitude': lat
                })
                points.append(point)
        
        return ee.FeatureCollection(points)
    
    def load_auxiliary_real_data(self, year=2023):
        """Load real auxiliary geospatial data"""
        
        print(f"\n🗺️  Loading Real Auxiliary Data for {year}")
        print("=" * 40)
        
        # Date range
        start_date = f'{year}-01-01'
        end_date = f'{year}-12-31'
        
        try:
            # MODIS Land Surface Temperature
            print("🌡️  Loading MODIS Land Surface Temperature...")
            lst_collection = ee.ImageCollection('MODIS/061/MOD11A1') \
                .filterDate(start_date, end_date) \
                .filterBounds(self.uzbekistan_bounds) \
                .select('LST_Day_1km')
            
            lst_mean = lst_collection.mean().multiply(0.02).subtract(273.15)  # Convert to Celsius
            
            # MODIS Vegetation Indices
            print("🌱 Loading MODIS Vegetation Indices...")
            ndvi_collection = ee.ImageCollection('MODIS/061/MOD13A1') \
                .filterDate(start_date, end_date) \
                .filterBounds(self.uzbekistan_bounds) \
                .select('NDVI')
            
            ndvi_mean = ndvi_collection.mean().multiply(0.0001)  # Scale factor
            
            # MODIS Land Cover
            print("🏞️  Loading MODIS Land Cover...")
            land_cover = ee.Image('MODIS/061/MCD12Q1/2022_01_01').select('LC_Type1')
            
            # Combine auxiliary layers
            auxiliary_image = ee.Image.cat([
                lst_mean.rename('temperature'),
                ndvi_mean.rename('ndvi'),
                land_cover.rename('land_cover')
            ])
            
            # Sample at the same points as atmospheric data
            sample_points = self.generate_sampling_grid()
            
            sampled_aux = auxiliary_image.sampleRegions(
                collection=sample_points,
                scale=1000,
                projection='EPSG:4326'
            )
            
            # Convert to DataFrame
            features = sampled_aux.getInfo()['features']
            aux_data = []
            
            for feature in features:
                props = feature['properties']
                coords = feature['geometry']['coordinates']
                
                data_point = {
                    'longitude': coords[0],
                    'latitude': coords[1],
                    'temperature': props.get('temperature', None),
                    'ndvi': props.get('ndvi', None),
                    'land_cover': props.get('land_cover', None),
                    'year': year
                }
                aux_data.append(data_point)
            
            aux_df = pd.DataFrame(aux_data)
            aux_df = aux_df.dropna()
            
            self.auxiliary_data = aux_df
            print(f"✅ Loaded {len(aux_df)} auxiliary data points")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading auxiliary data: {e}")
            return False
    
    def combine_datasets(self):
        """Combine atmospheric and auxiliary data"""
        
        print(f"\n🔗 Combining Real Datasets")
        print("=" * 30)
        
        if not self.atmospheric_data or len(self.auxiliary_data) == 0:
            print("❌ No data to combine")
            return None
        
        # Start with auxiliary data
        combined_df = self.auxiliary_data.copy()
        
        # Add atmospheric concentrations
        for gas, gas_df in self.atmospheric_data.items():
            # Merge on coordinates (with tolerance for floating point)
            combined_df = pd.merge(
                combined_df,
                gas_df[['longitude', 'latitude', f'{gas}_concentration']],
                on=['longitude', 'latitude'],
                how='left'
            )
        
        # Remove rows with missing atmospheric data
        concentration_cols = [f'{gas}_concentration' for gas in self.atmospheric_data.keys()]
        combined_df = combined_df.dropna(subset=concentration_cols)
        
        print(f"✅ Combined dataset: {len(combined_df)} points")
        print(f"   Variables: {list(combined_df.columns)}")
        
        return combined_df
    
    def analyze_patterns(self, combined_df):
        """Analyze atmospheric concentration patterns"""
        
        print(f"\n📊 Analyzing Atmospheric Patterns")
        print("=" * 35)
        
        # Basic statistics
        print("\n📈 Concentration Statistics:")
        concentration_cols = [col for col in combined_df.columns if '_concentration' in col]
        
        for col in concentration_cols:
            gas = col.replace('_concentration', '')
            values = combined_df[col].dropna()
            
            print(f"\n{gas.upper()} Concentrations:")
            print(f"  Count: {len(values)}")
            print(f"  Mean: {values.mean():.2e}")
            print(f"  Std: {values.std():.2e}")
            print(f"  Min: {values.min():.2e}")
            print(f"  Max: {values.max():.2e}")
        
        # Land cover analysis
        if 'land_cover' in combined_df.columns:
            print(f"\n🏞️  Land Cover Distribution:")
            lc_counts = combined_df['land_cover'].value_counts()
            for lc_type, count in lc_counts.head(5).items():
                print(f"  Type {lc_type}: {count} points")
        
        # Temperature analysis
        if 'temperature' in combined_df.columns:
            temp_stats = combined_df['temperature'].describe()
            print(f"\n🌡️  Temperature Statistics:")
            print(f"  Mean: {temp_stats['mean']:.1f}°C")
            print(f"  Range: {temp_stats['min']:.1f}°C to {temp_stats['max']:.1f}°C")
        
        return combined_df
    
    def save_results(self, combined_df, output_dir='outputs'):
        """Save analysis results"""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save main dataset
        data_file = output_path / 'real_atmospheric_data_uzbekistan.csv'
        combined_df.to_csv(data_file, index=False)
        print(f"💾 Saved data to: {data_file}")
        
        # Save summary report
        report_file = output_path / 'atmospheric_analysis_summary.txt'
        with open(report_file, 'w') as f:
            f.write("REAL ATMOSPHERIC DATA ANALYSIS - UZBEKISTAN\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Data Points: {len(combined_df)}\n")
            f.write(f"Variables: {len(combined_df.columns)}\n\n")
            
            # Add concentration statistics
            concentration_cols = [col for col in combined_df.columns if '_concentration' in col]
            f.write("ATMOSPHERIC CONCENTRATIONS:\n")
            f.write("-" * 30 + "\n")
            
            for col in concentration_cols:
                gas = col.replace('_concentration', '')
                values = combined_df[col].dropna()
                f.write(f"\n{gas.upper()}:\n")
                f.write(f"  Data Points: {len(values)}\n")
                f.write(f"  Mean: {values.mean():.2e}\n")
                f.write(f"  Std Dev: {values.std():.2e}\n")
                f.write(f"  Range: {values.min():.2e} - {values.max():.2e}\n")
        
        print(f"📄 Saved report to: {report_file}")
        
        return data_file

def main():
    """Main analysis function"""
    
    print("🌍 REAL ATMOSPHERIC DATA ANALYSIS - UZBEKISTAN")
    print("=" * 55)
    print("Using only real satellite measurements:")
    print("• Sentinel-5P atmospheric concentrations")
    print("• MODIS land surface data")
    print("• No simulated or mock data")
    print("=" * 55)
    
    # Initialize analyzer
    analyzer = RealAtmosphericDataAnalyzer()
    
    # Initialize Google Earth Engine
    if not analyzer.initialize_gee():
        print("❌ Cannot proceed without GEE authentication")
        return False
    
    # Load real atmospheric data for 2023
    if not analyzer.load_real_atmospheric_data(year=2023):
        print("❌ Failed to load atmospheric data")
        return False
    
    # Load auxiliary data
    if not analyzer.load_auxiliary_real_data(year=2023):
        print("❌ Failed to load auxiliary data")
        return False
    
    # Combine datasets
    combined_data = analyzer.combine_datasets()
    if combined_data is None:
        print("❌ Failed to combine datasets")
        return False
    
    # Analyze patterns
    analyzed_data = analyzer.analyze_patterns(combined_data)
    
    # Save results
    output_file = analyzer.save_results(analyzed_data)
    
    print(f"\n🎉 ANALYSIS COMPLETE!")
    print(f"✅ Real atmospheric data analysis finished")
    print(f"📊 {len(analyzed_data)} data points processed")
    print(f"💾 Results saved to: {output_file}")
    print(f"\n📁 Check the 'outputs' directory for:")
    print(f"   • real_atmospheric_data_uzbekistan.csv")
    print(f"   • atmospheric_analysis_summary.txt")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Analysis failed")
        sys.exit(1)
