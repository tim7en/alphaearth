#!/usr/bin/env python3
"""
Efficient Progressive Atmospheric Analysis with Auto-Resume

This script performs atmospheric analysis in small progressive chunks
with automatic resume capability and detailed progress tracking.

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
from datetime import datetime

class ProgressiveAnalyzer:
    """Progressive atmospheric data analyzer with resume capability"""
    
    def __init__(self):
        self.project_id = 'ee-sabitovty'
        self.uzbekistan_bounds = None
        self.gee_initialized = False
        self.progress_file = Path('outputs/analysis_progress.json')
        self.results_dir = Path('outputs/progressive_results')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def initialize_gee(self):
        """Initialize Google Earth Engine"""
        try:
            ee.Initialize(project=self.project_id)
            self.uzbekistan_bounds = ee.Geometry.Rectangle([55.9, 37.2, 73.2, 45.6])
            self.gee_initialized = True
            print(f"✅ GEE initialized with project: {self.project_id}")
            return True
        except Exception as e:
            print(f"❌ GEE initialization failed: {e}")
            return False
    
    def create_region_grid(self):
        """Create manageable regional chunks"""
        print("📐 Creating regional analysis grid...")
        
        # Divide Uzbekistan into 4x3 regions for manageable processing
        min_lon, min_lat, max_lon, max_lat = 55.9, 37.2, 73.2, 45.6
        
        lon_step = (max_lon - min_lon) / 4  # 4 columns
        lat_step = (max_lat - min_lat) / 3  # 3 rows
        
        regions = []
        region_id = 0
        
        for i in range(4):
            for j in range(3):
                region = {
                    'id': region_id,
                    'name': f'Region_{i}_{j}',
                    'bounds': [
                        min_lon + i * lon_step,      # min_lon
                        min_lat + j * lat_step,      # min_lat  
                        min_lon + (i + 1) * lon_step, # max_lon
                        min_lat + (j + 1) * lat_step  # max_lat
                    ]
                }
                regions.append(region)
                region_id += 1
        
        print(f"✅ Created {len(regions)} analysis regions")
        return regions
    
    def load_progress(self):
        """Load existing progress if available"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    progress = json.load(f)
                print(f"📋 Loaded existing progress: {progress['completed_regions']}/{progress['total_regions']} regions completed")
                return progress
            except:
                pass
        
        return {'completed_regions': 0, 'total_regions': 0, 'completed_region_ids': []}
    
    def save_progress(self, progress):
        """Save current progress"""
        with open(self.progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
    
    def process_single_region(self, region, gas_datasets):
        """Process atmospheric data for a single region"""
        
        region_name = region['name']
        bounds = region['bounds']
        
        print(f"\n🎯 Processing {region_name}")
        print(f"   Bounds: {bounds[0]:.2f}°E to {bounds[2]:.2f}°E, {bounds[1]:.2f}°N to {bounds[3]:.2f}°N")
        
        # Create region geometry
        region_geom = ee.Geometry.Rectangle(bounds)
        
        # Create sampling points for this region (higher resolution)
        resolution = 0.02  # ~2km resolution
        lons = np.arange(bounds[0], bounds[2], resolution)
        lats = np.arange(bounds[1], bounds[3], resolution)
        
        sample_points = []
        for lon in lons:
            for lat in lats:
                sample_points.append({
                    'longitude': lon,
                    'latitude': lat
                })
        
        print(f"   📍 {len(sample_points)} sampling points")
        
        # Process each gas for this region
        region_results = {}
        
        for gas, dataset, band in gas_datasets:
            print(f"   🛰️  Processing {gas}...", end='', flush=True)
            
            try:
                # Get atmospheric data
                collection = ee.ImageCollection(dataset) \
                    .filterDate('2024-01-01', '2024-12-31') \
                    .filterBounds(region_geom) \
                    .select(band)
                
                size = collection.size().getInfo()
                
                if size > 0:
                    mean_image = collection.mean()
                    
                    # Create point features (process in smaller batches)
                    batch_size = 20
                    all_results = []
                    
                    for i in range(0, len(sample_points), batch_size):
                        batch = sample_points[i:i + batch_size]
                        
                        features = []
                        for point in batch:
                            feature = ee.Feature(
                                ee.Geometry.Point([point['longitude'], point['latitude']]),
                                point
                            )
                            features.append(feature)
                        
                        batch_collection = ee.FeatureCollection(features)
                        
                        # Sample data
                        sampled = mean_image.sampleRegions(
                            collection=batch_collection,
                            scale=1000,
                            projection='EPSG:4326'
                        )
                        
                        # Extract results
                        batch_features = sampled.getInfo()['features']
                        
                        for feature in batch_features:
                            props = feature['properties']
                            if band in props and props[band] is not None:
                                result = {
                                    'longitude': props['longitude'],
                                    'latitude': props['latitude'],
                                    f'{gas}_concentration': props[band],
                                    'region': region_name
                                }
                                all_results.append(result)
                        
                        # Progress indicator
                        print('.', end='', flush=True)
                        time.sleep(0.05)  # Small delay
                    
                    if all_results:
                        region_results[gas] = pd.DataFrame(all_results)
                        print(f" ✅ {len(all_results)} points")
                    else:
                        print(f" ⚠️  No data")
                else:
                    print(f" ⚠️  No images")
                    
            except Exception as e:
                print(f" ❌ Error: {str(e)[:50]}")
        
        # Save region results
        if region_results:
            region_file = self.results_dir / f'{region_name}_atmospheric_data.json'
            
            # Convert DataFrames to dict for JSON serialization
            json_results = {}
            for gas, df in region_results.items():
                json_results[gas] = df.to_dict('records')
            
            with open(region_file, 'w') as f:
                json.dump(json_results, f, indent=2)
            
            print(f"   💾 Saved to {region_file}")
            return True
        else:
            print(f"   ⚠️  No data saved for {region_name}")
            return False
    
    def combine_region_results(self):
        """Combine all regional results into final dataset"""
        print(f"\n🔗 Combining results from all regions...")
        
        # Find all region result files
        result_files = list(self.results_dir.glob('Region_*_atmospheric_data.json'))
        
        if not result_files:
            print("❌ No region results found")
            return False
        
        print(f"📂 Found {len(result_files)} region files")
        
        # Combine all gas data
        combined_gases = {}
        
        for result_file in result_files:
            print(f"   📄 Processing {result_file.name}")
            
            with open(result_file, 'r') as f:
                region_data = json.load(f)
            
            for gas, data_records in region_data.items():
                if gas not in combined_gases:
                    combined_gases[gas] = []
                combined_gases[gas].extend(data_records)
        
        # Convert to DataFrames and save
        final_results = {}
        
        for gas, records in combined_gases.items():
            if records:
                df = pd.DataFrame(records)
                final_results[gas] = df
                
                # Save individual gas file
                gas_file = Path('outputs') / f'{gas.lower()}_concentrations_uzbekistan_full.csv'
                df.to_csv(gas_file, index=False)
                print(f"   💾 {gas}: {len(df)} points → {gas_file}")
        
        # Create master combined dataset
        if final_results:
            # Start with first gas
            gas_names = list(final_results.keys())
            master_df = final_results[gas_names[0]][['longitude', 'latitude', 'region']].copy()
            
            # Add concentration columns
            for gas in gas_names:
                concentration_data = final_results[gas][['longitude', 'latitude', f'{gas}_concentration']]
                master_df = pd.merge(
                    master_df,
                    concentration_data,
                    on=['longitude', 'latitude'],
                    how='outer'
                )
            
            # Save master file
            master_file = Path('outputs') / 'complete_atmospheric_analysis_uzbekistan.csv'
            master_df.to_csv(master_file, index=False)
            
            print(f"\n🎉 FINAL RESULTS:")
            print(f"   📊 Total data points: {len(master_df)}")
            print(f"   💨 Gases analyzed: {len(gas_names)}")
            print(f"   💾 Master dataset: {master_file}")
            
            # Create final summary
            summary = {
                'analysis_complete': True,
                'completion_date': datetime.now().isoformat(),
                'total_points': len(master_df),
                'gases_analyzed': gas_names,
                'spatial_resolution': '0.02 degrees (~2km)',
                'regions_processed': len(result_files),
                'data_quality': 'Real Sentinel-5P satellite measurements'
            }
            
            summary_file = Path('outputs') / 'complete_analysis_summary.json'
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            return True
        
        return False
    
    def run_progressive_analysis(self):
        """Run progressive analysis with resume capability"""
        
        print("🚀 PROGRESSIVE ATMOSPHERIC ANALYSIS - UZBEKISTAN")
        print("=" * 60)
        print("📊 High-resolution real data analysis")
        print("🔄 Automatic progress saving and resume")
        print("🛰️  Sentinel-5P atmospheric measurements")
        print("=" * 60)
        
        # Initialize
        if not self.initialize_gee():
            return False
        
        # Create regions
        regions = self.create_region_grid()
        
        # Load existing progress
        progress = self.load_progress()
        progress['total_regions'] = len(regions)
        
        # Define gas datasets
        gas_datasets = [
            ('CO', 'COPERNICUS/S5P/OFFL/L3_CO', 'CO_column_number_density'),
            ('NO2', 'COPERNICUS/S5P/OFFL/L3_NO2', 'tropospheric_NO2_column_number_density'),
            ('CH4', 'COPERNICUS/S5P/OFFL/L3_CH4', 'CH4_column_volume_mixing_ratio_dry_air')
        ]
        
        print(f"\n🎯 Processing Plan:")
        print(f"   Total regions: {len(regions)}")
        print(f"   Atmospheric gases: {len(gas_datasets)}")
        print(f"   Already completed: {progress['completed_regions']}")
        print(f"   Remaining: {len(regions) - progress['completed_regions']}")
        
        # Process remaining regions
        completed_this_session = 0
        
        for region in regions:
            if region['id'] in progress['completed_region_ids']:
                continue  # Skip already completed regions
            
            print(f"\n📈 Progress: {progress['completed_regions']}/{len(regions)} regions completed")
            
            try:
                success = self.process_single_region(region, gas_datasets)
                
                if success:
                    progress['completed_regions'] += 1
                    progress['completed_region_ids'].append(region['id'])
                    completed_this_session += 1
                    self.save_progress(progress)
                    
                    print(f"✅ {region['name']} completed ({completed_this_session} this session)")
                
            except KeyboardInterrupt:
                print(f"\n⏹️  Analysis paused. Progress saved.")
                print(f"   Completed this session: {completed_this_session}")
                print(f"   To resume, run the script again.")
                return False
            except Exception as e:
                print(f"❌ Error processing {region['name']}: {e}")
                continue
        
        # Combine all results
        print(f"\n🏁 All regions processed! Combining results...")
        success = self.combine_region_results()
        
        if success:
            # Clean up progress file
            if self.progress_file.exists():
                self.progress_file.unlink()
            
            print(f"\n🎉 PROGRESSIVE ANALYSIS COMPLETE!")
            print(f"✅ {len(regions)} regions processed")
            print(f"💾 Results saved to outputs/")
            
        return success

def main():
    """Main execution"""
    analyzer = ProgressiveAnalyzer()
    
    try:
        success = analyzer.run_progressive_analysis()
        return success
    except KeyboardInterrupt:
        print("\n⏹️  Analysis interrupted by user")
        return False

if __name__ == "__main__":
    main()
