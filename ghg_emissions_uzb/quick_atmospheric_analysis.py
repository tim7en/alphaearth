#!/usr/bin/env python3
"""
Quick Real Atmospheric Data Analysis for Uzbekistan

This script efficiently extracts real atmospheric data using optimized sampling.

Author: AlphaEarth Analysis Team
Date: August 15, 2025
"""

import ee
import pandas as pd
import numpy as np
from pathlib import Path
import json

def quick_atmospheric_analysis():
    """Quick analysis of real atmospheric data"""
    
    print("🚀 Quick Real Atmospheric Data Analysis - Uzbekistan")
    print("=" * 60)
    
    try:
        # Initialize GEE
        ee.Initialize(project='ee-sabitovty')
        print("✅ Google Earth Engine initialized")
        
        # Define Uzbekistan region (simplified)
        uzbekistan = ee.Geometry.Rectangle([55.9, 37.2, 73.2, 45.6])
        
        # Define key cities for targeted sampling
        key_cities = {
            'Tashkent': [69.2401, 41.2995],
            'Samarkand': [66.9597, 39.6270],
            'Namangan': [71.6726, 40.9983],
            'Bukhara': [64.4207, 39.7747],
            'Nukus': [59.6103, 42.4531]
        }
        
        print(f"\n📍 Sampling {len(key_cities)} major cities")
        
        # Create point collection for cities
        city_points = []
        for city, coords in key_cities.items():
            point = ee.Feature(
                ee.Geometry.Point(coords),
                {'city': city, 'longitude': coords[0], 'latitude': coords[1]}
            )
            city_points.append(point)
        
        city_collection = ee.FeatureCollection(city_points)
        
        # Test available datasets quickly
        print("\n🔍 Testing Data Availability:")
        
        datasets_to_test = {
            'CO': 'COPERNICUS/S5P/OFFL/L3_CO',
            'NO2': 'COPERNICUS/S5P/OFFL/L3_NO2',
            'CH4': 'COPERNICUS/S5P/OFFL/L3_CH4'
        }
        
        bands = {
            'CO': 'CO_column_number_density',
            'NO2': 'tropospheric_NO2_column_number_density', 
            'CH4': 'CH4_column_volume_mixing_ratio_dry_air'
        }
        
        results = {}
        
        for gas, dataset in datasets_to_test.items():
            try:
                print(f"📡 Testing {gas} from {dataset}...")
                
                # Get recent data (last 30 days to ensure availability)
                collection = ee.ImageCollection(dataset) \
                    .filterDate('2024-01-01', '2024-12-31') \
                    .filterBounds(uzbekistan) \
                    .select(bands[gas]) \
                    .limit(10)  # Limit to 10 images for speed
                
                size = collection.size().getInfo()
                print(f"   Found {size} images")
                
                if size > 0:
                    # Calculate mean for available data
                    mean_image = collection.mean()
                    
                    # Sample at city points only (much faster)
                    sampled = mean_image.sampleRegions(
                        collection=city_collection,
                        scale=5000,  # 5km resolution for speed
                        projection='EPSG:4326'
                    )
                    
                    # Get the results
                    features = sampled.getInfo()['features']
                    
                    city_data = []
                    for feature in features:
                        props = feature['properties']
                        if bands[gas] in props and props[bands[gas]] is not None:
                            city_data.append({
                                'city': props['city'],
                                'longitude': props['longitude'],
                                'latitude': props['latitude'],
                                f'{gas}_concentration': props[bands[gas]]
                            })
                    
                    if city_data:
                        results[gas] = pd.DataFrame(city_data)
                        print(f"   ✅ Extracted data for {len(city_data)} cities")
                    else:
                        print(f"   ⚠️  No valid data extracted")
                
            except Exception as e:
                print(f"   ❌ Error: {str(e)[:100]}...")
        
        # Combine results
        if results:
            print(f"\n🔗 Combining results from {len(results)} gases...")
            
            # Start with first dataset
            gas_names = list(results.keys())
            combined_df = results[gas_names[0]].copy()
            
            # Merge other gases
            for gas in gas_names[1:]:
                combined_df = pd.merge(
                    combined_df,
                    results[gas][['city', f'{gas}_concentration']],
                    on='city',
                    how='outer'
                )
            
            print(f"✅ Combined dataset: {len(combined_df)} cities")
            
            # Display results
            print(f"\n📊 ATMOSPHERIC CONCENTRATIONS BY CITY:")
            print("=" * 50)
            
            for _, row in combined_df.iterrows():
                print(f"\n🏙️  {row['city']} ({row['latitude']:.2f}°N, {row['longitude']:.2f}°E)")
                
                for gas in gas_names:
                    conc_col = f'{gas}_concentration'
                    if conc_col in row and pd.notna(row[conc_col]):
                        if gas == 'CH4':
                            print(f"   {gas}: {row[conc_col]:.1f} ppb")
                        else:
                            print(f"   {gas}: {row[conc_col]:.2e} mol/m²")
                    else:
                        print(f"   {gas}: No data")
            
            # Save results
            output_dir = Path('outputs')
            output_dir.mkdir(exist_ok=True)
            
            output_file = output_dir / 'real_atmospheric_data_cities.csv'
            combined_df.to_csv(output_file, index=False)
            
            # Create summary
            summary = {
                'analysis_date': '2025-08-15',
                'cities_analyzed': len(combined_df),
                'gases_detected': gas_names,
                'data_source': 'Sentinel-5P OFFL',
                'spatial_resolution': '5km',
                'temporal_coverage': '2024'
            }
            
            summary_file = output_dir / 'analysis_summary.json'
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"\n💾 Results saved:")
            print(f"   Data: {output_file}")
            print(f"   Summary: {summary_file}")
            
            print(f"\n🎉 QUICK ANALYSIS COMPLETE!")
            print(f"✅ Real atmospheric data successfully extracted")
            print(f"📊 {len(combined_df)} cities analyzed")
            print(f"💨 {len(gas_names)} gases detected")
            
            return True
            
        else:
            print("\n❌ No atmospheric data could be extracted")
            print("This may be due to:")
            print("  • Data access restrictions")
            print("  • Temporary data unavailability")
            print("  • Network connectivity issues")
            return False
            
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        return False

if __name__ == "__main__":
    success = quick_atmospheric_analysis()
    if not success:
        print("\n💡 Consider trying:")
        print("  • Different date ranges")
        print("  • Alternative atmospheric datasets") 
        print("  • Checking data access permissions")
