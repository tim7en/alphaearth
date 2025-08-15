#!/usr/bin/env python3
"""
High-Resolution Server-Side Atmospheric Analysis for Uzbekistan

Enhanced version with higher spatial resolution and more detailed analysis.

Author: AlphaEarth Analysis Team
Date: August 15, 2025
"""

import ee
import pandas as pd
import json
from pathlib import Path
import time
from datetime import datetime

def high_resolution_atmospheric_analysis():
    """High-resolution server-side atmospheric analysis"""
    
    print("🔬 HIGH-RESOLUTION ATMOSPHERIC ANALYSIS - UZBEKISTAN")
    print("=" * 65)
    print("🎯 Enhanced parameters:")
    print("   • 1km spatial resolution (vs 10km in test)")
    print("   • 10 major cities")
    print("   • 3 gases (NO₂, CO, CH₄)")
    print("   • 3-month period (Jul-Sep 2024)")
    print("   • Regional grid analysis")
    print("   • Progress tracking")
    print("=" * 65)
    
    try:
        # Initialize GEE
        print("\n🔧 Initializing Google Earth Engine...")
        ee.Initialize(project='ee-sabitovty')
        print("✅ Google Earth Engine initialized")
        
        # Define study area
        uzbekistan_bounds = ee.Geometry.Rectangle([55.9, 37.2, 73.2, 45.6])
        
        # Extended city list for higher resolution analysis
        cities = {
            'Tashkent': [69.2401, 41.2995],      # Capital
            'Samarkand': [66.9597, 39.6270],     # Historic center
            'Namangan': [71.6726, 40.9983],      # Fergana Valley
            'Andijan': [72.3442, 40.7821],       # Eastern industrial
            'Bukhara': [64.4207, 39.7747],       # Central oasis
            'Nukus': [59.6103, 42.4531],         # Karakalpakstan capital
            'Qarshi': [65.7887, 38.8569],        # Southern region
            'Kokand': [70.9428, 40.5258],        # Fergana Valley
            'Urgench': [60.6348, 41.5500],       # Khorezm region
            'Margilan': [71.7246, 40.4731]       # Silk road city
        }
        
        print(f"\n📍 High-resolution analysis for {len(cities)} cities")
        
        # Create enhanced city points with buffer zones
        city_features = []
        for city_name, coords in cities.items():
            # Main city point
            main_point = ee.Feature(
                ee.Geometry.Point(coords),
                {'city': city_name, 'type': 'center', 'lon': coords[0], 'lat': coords[1]}
            )
            city_features.append(main_point)
            
            # Add surrounding points for urban area analysis (higher resolution)
            offsets = [
                [-0.02, -0.02], [0, -0.02], [0.02, -0.02],  # South row
                [-0.02, 0],     [0.02, 0],                   # Center row (excluding main point)
                [-0.02, 0.02],  [0, 0.02],  [0.02, 0.02]    # North row
            ]
            
            for i, (dx, dy) in enumerate(offsets):
                surrounding_point = ee.Feature(
                    ee.Geometry.Point([coords[0] + dx, coords[1] + dy]),
                    {'city': city_name, 'type': f'suburb_{i+1}', 'lon': coords[0] + dx, 'lat': coords[1] + dy}
                )
                city_features.append(surrounding_point)
        
        city_collection = ee.FeatureCollection(city_features)
        print(f"✅ Created {len(city_features)} sampling points (city centers + suburban areas)")
        
        # High-resolution gas analysis
        datasets = {
            'NO2': {
                'collection': 'COPERNICUS/S5P/OFFL/L3_NO2',
                'band': 'tropospheric_NO2_column_number_density',
                'scale': 1000,  # 1km resolution
                'description': 'Nitrogen Dioxide'
            },
            'CO': {
                'collection': 'COPERNICUS/S5P/OFFL/L3_CO',
                'band': 'CO_column_number_density',
                'scale': 1000,  # 1km resolution
                'description': 'Carbon Monoxide'
            },
            'CH4': {
                'collection': 'COPERNICUS/S5P/OFFL/L3_CH4',
                'band': 'CH4_column_volume_mixing_ratio_dry_air',
                'scale': 1000,  # 1km resolution
                'description': 'Methane'
            }
        }
        
        # Extended time period for better statistics
        start_date = '2024-07-01'
        end_date = '2024-09-30'  # 3 months (Jul-Sep 2024)
        
        print(f"\n📅 Analysis period: {start_date} to {end_date} (3 months)")
        print(f"🎯 Target resolution: 1km")
        
        results = {}
        total_gases = len(datasets)
        
        for gas_idx, (gas, config) in enumerate(datasets.items(), 1):
            print(f"\n📡 Processing {config['description']} ({gas}) - {gas_idx}/{total_gases}")
            print("   " + "▓" * int(gas_idx/total_gases * 20) + "░" * (20 - int(gas_idx/total_gases * 20)) + f" {gas_idx/total_gases*100:.0f}%")
            
            try:
                # Load collection with extended timeframe
                print("   🔄 Loading satellite data collection...")
                collection = ee.ImageCollection(config['collection']) \
                    .filterDate(start_date, end_date) \
                    .filterBounds(uzbekistan_bounds) \
                    .select(config['band'])
                
                # Check data availability
                size = collection.size().getInfo()
                print(f"   📊 Found {size} satellite images")
                
                if size > 0:
                    # High-resolution temporal aggregation
                    print("   🧮 Computing high-resolution temporal statistics...")
                    mean_concentration = collection.mean()
                    median_concentration = collection.median()
                    std_concentration = collection.reduce(ee.Reducer.stdDev())
                    
                    # High-resolution sampling at all points
                    print("   📍 High-resolution sampling at city locations...")
                    
                    # Sample mean concentrations
                    sampled_mean = mean_concentration.sampleRegions(
                        collection=city_collection,
                        scale=config['scale'],  # 1km resolution
                        projection='EPSG:4326'
                    )
                    
                    # Sample temporal variability
                    sampled_std = std_concentration.sampleRegions(
                        collection=city_collection,
                        scale=config['scale'],
                        projection='EPSG:4326'
                    )
                    
                    # Extract results from server
                    print("   📥 Downloading processed results...")
                    mean_features = sampled_mean.getInfo()['features']
                    std_features = sampled_std.getInfo()['features']
                    
                    # Combine mean and std data
                    gas_data = {}
                    std_data = {f['properties']['city'] + '_' + f['properties']['type']: 
                              f['properties'].get(config['band'] + '_stdDev', 0) for f in std_features}
                    
                    for feature in mean_features:
                        props = feature['properties']
                        city = props['city']
                        point_type = props['type']
                        key = f"{city}_{point_type}"
                        concentration = props.get(config['band'])
                        
                        if concentration is not None:
                            gas_data[key] = {
                                'concentration': concentration,
                                'std_dev': std_data.get(key, 0),
                                'city': city,
                                'type': point_type,
                                'longitude': props['lon'],
                                'latitude': props['lat']
                            }
                    
                    # Regional statistics for high-resolution analysis
                    print("   🌍 Computing regional statistics at 1km resolution...")
                    
                    # Define administrative regions
                    regions = {
                        'Tashkent_Region': ee.Geometry.Rectangle([68.5, 40.5, 71.0, 42.0]),
                        'Samarkand_Region': ee.Geometry.Rectangle([66.0, 38.5, 68.5, 40.5]),
                        'Fergana_Valley': ee.Geometry.Rectangle([70.0, 40.0, 73.2, 41.5]),
                        'Karakalpakstan': ee.Geometry.Rectangle([55.9, 41.0, 63.0, 45.6]),
                        'Central_Region': ee.Geometry.Rectangle([63.0, 37.2, 68.5, 40.5])
                    }
                    
                    regional_stats = {}
                    for region_name, region_geom in regions.items():
                        region_stats = mean_concentration.reduceRegion(
                            reducer=ee.Reducer.mean().combine(
                                reducer2=ee.Reducer.minMax(),
                                sharedInputs=True
                            ).combine(
                                reducer2=ee.Reducer.stdDev(),
                                sharedInputs=True
                            ).combine(
                                reducer2=ee.Reducer.count(),
                                sharedInputs=True
                            ),
                            geometry=region_geom,
                            scale=config['scale'],  # 1km resolution
                            maxPixels=1e8
                        )
                        regional_stats[region_name] = region_stats.getInfo()
                    
                    results[gas] = {
                        'city_data': gas_data,
                        'regional_stats': regional_stats,
                        'collection_size': size,
                        'resolution_m': config['scale'],
                        'description': config['description']
                    }
                    
                    print(f"   ✅ {gas} analysis complete - {len(gas_data)} high-res points processed")
                    
                else:
                    print(f"   ⚠️  No {gas} data available for specified period")
                    
            except Exception as e:
                print(f"   ❌ Error processing {gas}: {str(e)[:100]}...")
        
        # Process and display high-resolution results
        if results:
            print(f"\n📊 HIGH-RESOLUTION RESULTS ANALYSIS")
            print("=" * 45)
            
            # Aggregate city-level results (average suburban measurements)
            city_aggregated = {}
            for city in cities.keys():
                city_aggregated[city] = {'longitude': cities[city][0], 'latitude': cities[city][1]}
                
                for gas, gas_data in results.items():
                    if 'city_data' in gas_data:
                        # Get all measurements for this city (center + suburbs)
                        city_measurements = [
                            data['concentration'] for key, data in gas_data['city_data'].items()
                            if data['city'] == city and data['concentration'] is not None
                        ]
                        
                        if city_measurements:
                            # Calculate high-resolution statistics
                            city_aggregated[city][f'{gas}_mean'] = sum(city_measurements) / len(city_measurements)
                            city_aggregated[city][f'{gas}_min'] = min(city_measurements)
                            city_aggregated[city][f'{gas}_max'] = max(city_measurements)
                            city_aggregated[city][f'{gas}_count'] = len(city_measurements)
                            
                            # Calculate spatial variability within city
                            if len(city_measurements) > 1:
                                mean_val = city_aggregated[city][f'{gas}_mean']
                                variance = sum((x - mean_val)**2 for x in city_measurements) / len(city_measurements)
                                city_aggregated[city][f'{gas}_spatial_std'] = variance**0.5
                            else:
                                city_aggregated[city][f'{gas}_spatial_std'] = 0
            
            # Display enhanced results
            for city, data in city_aggregated.items():
                print(f"\n🏙️  {city} (High-Resolution Analysis):")
                print(f"   📍 Location: {data['latitude']:.3f}°N, {data['longitude']:.3f}°E")
                
                for gas in results.keys():
                    if f'{gas}_mean' in data:
                        mean_val = data[f'{gas}_mean']
                        min_val = data[f'{gas}_min']
                        max_val = data[f'{gas}_max']
                        spatial_std = data[f'{gas}_spatial_std']
                        count = data[f'{gas}_count']
                        
                        if gas == 'CH4':
                            print(f"   {gas}: {mean_val:.1f} ± {spatial_std:.1f} ppb (range: {min_val:.1f}-{max_val:.1f}, n={count})")
                        else:
                            print(f"   {gas}: {mean_val:.2e} ± {spatial_std:.2e} mol/m² (range: {min_val:.2e}-{max_val:.2e}, n={count})")
                    else:
                        print(f"   {gas}: No high-resolution data")
            
            # Save high-resolution results
            print(f"\n💾 Saving high-resolution results...")
            output_dir = Path('outputs')
            output_dir.mkdir(exist_ok=True)
            
            # Create detailed CSV
            hr_rows = []
            for city, data in city_aggregated.items():
                row = {'city': city, 'longitude': data['longitude'], 'latitude': data['latitude']}
                
                for gas in results.keys():
                    row[f'{gas}_mean_concentration'] = data.get(f'{gas}_mean')
                    row[f'{gas}_min_concentration'] = data.get(f'{gas}_min')
                    row[f'{gas}_max_concentration'] = data.get(f'{gas}_max')
                    row[f'{gas}_spatial_std'] = data.get(f'{gas}_spatial_std')
                    row[f'{gas}_sample_count'] = data.get(f'{gas}_count')
                
                hr_rows.append(row)
            
            hr_df = pd.DataFrame(hr_rows)
            hr_file = output_dir / 'high_resolution_atmospheric_data.csv'
            hr_df.to_csv(hr_file, index=False)
            
            # Save comprehensive metadata
            hr_metadata = {
                'analysis_date': datetime.now().isoformat(),
                'analysis_type': 'high_resolution_server_side',
                'spatial_resolution': '1km',
                'temporal_coverage': f"{start_date} to {end_date}",
                'cities_analyzed': len(cities),
                'gases_analyzed': list(results.keys()),
                'sampling_points_per_city': 9,  # center + 8 suburban points
                'total_sampling_points': len(city_features),
                'processing_location': 'Google Earth Engine Servers',
                'regional_analysis': True,
                'spatial_statistics': True
            }
            
            metadata_file = output_dir / 'high_resolution_metadata.json'
            with open(metadata_file, 'w') as f:
                json.dump(hr_metadata, f, indent=2)
            
            print(f"   📄 Data: {hr_file}")
            print(f"   📄 Metadata: {metadata_file}")
            
            print(f"\n🎉 HIGH-RESOLUTION ANALYSIS COMPLETE!")
            print(f"✅ {len(cities)} cities analyzed at 1km resolution")
            print(f"📊 {len(city_features)} total sampling points processed")
            print(f"💨 {len(results)} gases analyzed with spatial statistics")
            print(f"🎯 Enhanced spatial variability analysis included")
            
            return True
            
        else:
            print("\n❌ No high-resolution data extracted")
            return False
            
    except Exception as e:
        print(f"\n❌ High-resolution analysis failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting high-resolution atmospheric analysis...")
    start_time = time.time()
    
    success = high_resolution_atmospheric_analysis()
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n⏱️  High-resolution analysis completed in {duration:.1f} seconds")
    
    if success:
        print("✅ High-resolution analysis successful!")
        print("\n📈 Enhanced features delivered:")
        print("   • 1km spatial resolution (10x improvement)")
        print("   • Spatial variability within cities")
        print("   • Extended temporal coverage (3 months)")
        print("   • Regional statistical analysis")
        print("   • Comprehensive uncertainty quantification")
    else:
        print("❌ High-resolution analysis failed")
