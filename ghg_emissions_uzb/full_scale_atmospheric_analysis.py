#!/usr/bin/env python3
"""
Full-Scale Real Atmospheric Data Analysis for Uzbekistan with Progress Tracking

This script performs comprehensive atmospheric analysis across Uzbekistan
with real-time progress monitoring and efficient batch processing.

Author: AlphaEarth Analysis Team
Date: August 15, 2025
"""

import ee
import pandas as pd
import numpy as np
from pathlib import Path
import json
import time
import sys
from datetime import datetime, timedelta

class ProgressTracker:
    """Simple progress tracking class"""
    
    def __init__(self, total_steps, description="Processing"):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.start_time = time.time()
        
    def update(self, step_description=""):
        """Update progress and display"""
        self.current_step += 1
        elapsed = time.time() - self.start_time
        
        # Calculate progress percentage
        progress = (self.current_step / self.total_steps) * 100
        
        # Estimate time remaining
        if self.current_step > 0:
            time_per_step = elapsed / self.current_step
            remaining_steps = self.total_steps - self.current_step
            eta = remaining_steps * time_per_step
            eta_str = f"{eta:.0f}s"
        else:
            eta_str = "Unknown"
        
        # Create progress bar
        bar_length = 40
        filled_length = int(bar_length * progress / 100)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        # Display progress
        print(f"\r🔄 {self.description} [{bar}] {progress:.1f}% ({self.current_step}/{self.total_steps}) ETA: {eta_str} - {step_description}", end='', flush=True)
        
        if self.current_step >= self.total_steps:
            print(f"\n✅ {self.description} completed in {elapsed:.1f}s")

class FullScaleAtmosphericAnalyzer:
    """Full-scale atmospheric data analyzer with progress tracking"""
    
    def __init__(self):
        self.project_id = 'ee-sabitovty'
        self.uzbekistan_bounds = None
        self.gee_initialized = False
        self.results = {}
        
    def initialize_gee(self):
        """Initialize Google Earth Engine"""
        print("🔧 Initializing Google Earth Engine...")
        try:
            ee.Initialize(project=self.project_id)
            self.uzbekistan_bounds = ee.Geometry.Rectangle([55.9, 37.2, 73.2, 45.6])
            self.gee_initialized = True
            print(f"✅ GEE initialized with project: {self.project_id}")
            return True
        except Exception as e:
            print(f"❌ GEE initialization failed: {e}")
            return False
    
    def create_sampling_grid(self, resolution=0.05):
        """Create systematic sampling grid across Uzbekistan"""
        print(f"📐 Creating sampling grid (resolution: {resolution}°)...")
        
        # Uzbekistan bounds
        min_lon, min_lat, max_lon, max_lat = 55.9, 37.2, 73.2, 45.6
        
        # Create grid points
        lons = np.arange(min_lon, max_lon, resolution)
        lats = np.arange(min_lat, max_lat, resolution)
        
        grid_points = []
        point_id = 0
        
        for i, lon in enumerate(lons):
            for j, lat in enumerate(lats):
                grid_points.append({
                    'id': point_id,
                    'longitude': lon,
                    'latitude': lat,
                    'grid_x': i,
                    'grid_y': j
                })
                point_id += 1
        
        print(f"✅ Created {len(grid_points)} sampling points")
        return grid_points
    
    def process_atmospheric_data_batch(self, gas_info, sample_points, batch_size=50):
        """Process atmospheric data in batches with progress tracking"""
        
        gas, dataset, band = gas_info
        print(f"\n🛰️  Processing {gas} atmospheric data...")
        
        # Create batches
        batches = [sample_points[i:i + batch_size] for i in range(0, len(sample_points), batch_size)]
        total_batches = len(batches)
        
        # Initialize progress tracker
        progress = ProgressTracker(total_batches, f"{gas} Data Extraction")
        
        all_results = []
        successful_extractions = 0
        
        # Get collection info first
        try:
            collection = ee.ImageCollection(dataset) \
                .filterDate('2024-01-01', '2024-12-31') \
                .filterBounds(self.uzbekistan_bounds) \
                .select(band)
            
            collection_size = collection.size().getInfo()
            print(f"📊 Found {collection_size} {gas} images for 2024")
            
            if collection_size == 0:
                print(f"⚠️  No {gas} data available for 2024")
                return pd.DataFrame()
            
            # Calculate mean image
            mean_image = collection.mean()
            
        except Exception as e:
            print(f"❌ Error accessing {gas} collection: {e}")
            return pd.DataFrame()
        
        # Process each batch
        for batch_idx, batch_points in enumerate(batches):
            try:
                # Create point features for this batch
                features = []
                for point in batch_points:
                    feature = ee.Feature(
                        ee.Geometry.Point([point['longitude'], point['latitude']]),
                        point
                    )
                    features.append(feature)
                
                batch_collection = ee.FeatureCollection(features)
                
                # Sample the mean image
                sampled = mean_image.sampleRegions(
                    collection=batch_collection,
                    scale=1000,  # 1km resolution
                    projection='EPSG:4326'
                )
                
                # Extract results
                batch_features = sampled.getInfo()['features']
                
                batch_results = []
                for feature in batch_features:
                    props = feature['properties']
                    if band in props and props[band] is not None:
                        result = {
                            'id': props['id'],
                            'longitude': props['longitude'],
                            'latitude': props['latitude'],
                            'grid_x': props['grid_x'],
                            'grid_y': props['grid_y'],
                            f'{gas}_concentration': props[band]
                        }
                        batch_results.append(result)
                
                all_results.extend(batch_results)
                successful_extractions += len(batch_results)
                
                # Update progress
                progress.update(f"Batch {batch_idx + 1}/{total_batches} - {len(batch_results)} points extracted")
                
                # Add small delay to avoid rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                progress.update(f"Batch {batch_idx + 1}/{total_batches} - Error: {str(e)[:50]}")
                continue
        
        # Convert to DataFrame
        if all_results:
            df = pd.DataFrame(all_results)
            print(f"\n✅ {gas} extraction complete: {len(df)} points with valid data")
            return df
        else:
            print(f"\n⚠️  No valid {gas} data extracted")
            return pd.DataFrame()
    
    def load_auxiliary_data_batch(self, sample_points, batch_size=100):
        """Load auxiliary data (temperature, vegetation, land cover) in batches"""
        
        print(f"\n🗺️  Processing auxiliary geospatial data...")
        
        # Create batches
        batches = [sample_points[i:i + batch_size] for i in range(0, len(sample_points), batch_size)]
        total_batches = len(batches)
        
        # Initialize progress tracker
        progress = ProgressTracker(total_batches, "Auxiliary Data Extraction")
        
        all_results = []
        
        try:
            # Prepare auxiliary datasets
            print("📊 Preparing auxiliary datasets...")
            
            # MODIS Land Surface Temperature (2024 mean)
            lst_collection = ee.ImageCollection('MODIS/061/MOD11A1') \
                .filterDate('2024-01-01', '2024-12-31') \
                .filterBounds(self.uzbekistan_bounds) \
                .select('LST_Day_1km')
            
            lst_mean = lst_collection.mean().multiply(0.02).subtract(273.15)  # Convert to Celsius
            
            # MODIS Vegetation Index (2024 mean)
            ndvi_collection = ee.ImageCollection('MODIS/061/MOD13A1') \
                .filterDate('2024-01-01', '2024-12-31') \
                .filterBounds(self.uzbekistan_bounds) \
                .select('NDVI')
            
            ndvi_mean = ndvi_collection.mean().multiply(0.0001)
            
            # MODIS Land Cover (latest available)
            land_cover = ee.Image('MODIS/061/MCD12Q1/2022_01_01').select('LC_Type1')
            
            # Elevation (SRTM)
            elevation = ee.Image('USGS/SRTMGL1_003')
            
            # Combine all auxiliary layers
            auxiliary_image = ee.Image.cat([
                lst_mean.rename('temperature'),
                ndvi_mean.rename('ndvi'),
                land_cover.rename('land_cover'),
                elevation.rename('elevation')
            ])
            
            # Process each batch
            for batch_idx, batch_points in enumerate(batches):
                try:
                    # Create point features
                    features = []
                    for point in batch_points:
                        feature = ee.Feature(
                            ee.Geometry.Point([point['longitude'], point['latitude']]),
                            point
                        )
                        features.append(feature)
                    
                    batch_collection = ee.FeatureCollection(features)
                    
                    # Sample auxiliary data
                    sampled = auxiliary_image.sampleRegions(
                        collection=batch_collection,
                        scale=1000,
                        projection='EPSG:4326'
                    )
                    
                    # Extract results
                    batch_features = sampled.getInfo()['features']
                    
                    for feature in batch_features:
                        props = feature['properties']
                        result = {
                            'id': props['id'],
                            'longitude': props['longitude'],
                            'latitude': props['latitude'],
                            'grid_x': props['grid_x'],
                            'grid_y': props['grid_y'],
                            'temperature': props.get('temperature'),
                            'ndvi': props.get('ndvi'),
                            'land_cover': props.get('land_cover'),
                            'elevation': props.get('elevation')
                        }
                        all_results.append(result)
                    
                    # Update progress
                    progress.update(f"Batch {batch_idx + 1}/{total_batches} - {len(batch_features)} points processed")
                    
                    # Small delay
                    time.sleep(0.05)
                    
                except Exception as e:
                    progress.update(f"Batch {batch_idx + 1}/{total_batches} - Error: {str(e)[:50]}")
                    continue
            
            if all_results:
                df = pd.DataFrame(all_results)
                # Remove rows with all null auxiliary data
                df = df.dropna(subset=['temperature', 'ndvi', 'land_cover', 'elevation'], how='all')
                print(f"\n✅ Auxiliary data extraction complete: {len(df)} points")
                return df
            else:
                print(f"\n⚠️  No auxiliary data extracted")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"\n❌ Error in auxiliary data processing: {e}")
            return pd.DataFrame()
    
    def run_full_analysis(self):
        """Run complete full-scale analysis"""
        
        print("🌍 FULL-SCALE REAL ATMOSPHERIC DATA ANALYSIS - UZBEKISTAN")
        print("=" * 70)
        print("🎯 Comprehensive analysis with progress tracking")
        print("📊 High-resolution sampling across entire country")
        print("🛰️  Real satellite data only - no simulations")
        print("=" * 70)
        
        # Initialize
        if not self.initialize_gee():
            return False
        
        # Create sampling grid
        sample_points = self.create_sampling_grid(resolution=0.05)  # ~5km resolution
        print(f"📍 Total sampling points: {len(sample_points)}")
        
        # Define atmospheric datasets to process
        atmospheric_datasets = [
            ('CO', 'COPERNICUS/S5P/OFFL/L3_CO', 'CO_column_number_density'),
            ('NO2', 'COPERNICUS/S5P/OFFL/L3_NO2', 'tropospheric_NO2_column_number_density'),
            ('CH4', 'COPERNICUS/S5P/OFFL/L3_CH4', 'CH4_column_volume_mixing_ratio_dry_air'),
            ('SO2', 'COPERNICUS/S5P/NRTI/L3_SO2', 'SO2_column_number_density')
        ]
        
        # Process each atmospheric dataset
        atmospheric_results = {}
        for gas_info in atmospheric_datasets:
            gas_df = self.process_atmospheric_data_batch(gas_info, sample_points, batch_size=25)
            if not gas_df.empty:
                atmospheric_results[gas_info[0]] = gas_df
        
        # Process auxiliary data
        auxiliary_df = self.load_auxiliary_data_batch(sample_points, batch_size=50)
        
        # Combine all results
        print(f"\n🔗 Combining all datasets...")
        if atmospheric_results and not auxiliary_df.empty:
            # Start with auxiliary data
            combined_df = auxiliary_df.copy()
            
            # Merge atmospheric data
            for gas, gas_df in atmospheric_results.items():
                combined_df = pd.merge(
                    combined_df,
                    gas_df[['id', f'{gas}_concentration']],
                    on='id',
                    how='left'
                )
            
            # Save results
            self.save_full_results(combined_df, atmospheric_results)
            
            print(f"\n🎉 FULL-SCALE ANALYSIS COMPLETE!")
            print(f"✅ Final dataset: {len(combined_df)} points")
            print(f"🛰️  Atmospheric gases: {len(atmospheric_results)}")
            print(f"🗺️  Auxiliary variables: 4")
            print(f"💾 Results saved to outputs/ directory")
            
            return True
        else:
            print(f"\n❌ Insufficient data for analysis")
            return False
    
    def save_full_results(self, combined_df, atmospheric_results):
        """Save comprehensive results"""
        
        output_dir = Path('outputs')
        output_dir.mkdir(exist_ok=True)
        
        # Main dataset
        main_file = output_dir / 'full_scale_atmospheric_analysis_uzbekistan.csv'
        combined_df.to_csv(main_file, index=False)
        
        # Individual gas datasets
        for gas, gas_df in atmospheric_results.items():
            gas_file = output_dir / f'{gas.lower()}_concentrations_uzbekistan.csv'
            gas_df.to_csv(gas_file, index=False)
        
        # Comprehensive summary
        summary = {
            'analysis_date': datetime.now().isoformat(),
            'total_points': len(combined_df),
            'spatial_resolution': '0.05 degrees (~5km)',
            'temporal_coverage': '2024',
            'atmospheric_gases': list(atmospheric_results.keys()),
            'auxiliary_variables': ['temperature', 'ndvi', 'land_cover', 'elevation'],
            'data_sources': {
                'atmospheric': 'Sentinel-5P OFFL/NRTI',
                'temperature': 'MODIS/061/MOD11A1',
                'vegetation': 'MODIS/061/MOD13A1',
                'land_cover': 'MODIS/061/MCD12Q1',
                'elevation': 'USGS/SRTMGL1_003'
            },
            'coverage_area': 'Uzbekistan (55.9°E-73.2°E, 37.2°N-45.6°N)',
            'data_quality': 'Real satellite measurements only'
        }
        
        summary_file = output_dir / 'full_analysis_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"💾 Saved comprehensive results:")
        print(f"   Main dataset: {main_file}")
        print(f"   Individual gases: {len(atmospheric_results)} files")
        print(f"   Summary: {summary_file}")

def main():
    """Main execution function"""
    
    analyzer = FullScaleAtmosphericAnalyzer()
    success = analyzer.run_full_analysis()
    
    if not success:
        print("\n❌ Full-scale analysis failed")
        return False
    
    return True

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
