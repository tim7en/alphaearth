#!/usr/bin/env python3
"""
Server-Side Atmospheric Analysis on Google Earth Engine

This script runs the entire analysis on GEE servers and only downloads
final aggregated results. Much faster and more efficient.

Author: AlphaEarth Analysis Team
Date: August 15, 2025
"""

import ee
import pandas as pd
import numpy as np
from pathlib import Path
import json
import time
from datetime import datetime

class ServerSideAtmosphericAnalyzer:
    """Runs atmospheric analysis entirely on GEE servers"""
    
    def __init__(self, project_id='ee-sabitovty'):
        self.project_id = project_id
        self.uzbekistan_bounds = None
        self.results = {}
        
    def initialize(self):
        """Initialize GEE and define study area"""
        try:
            ee.Initialize(project=self.project_id)
            print(f"✅ Google Earth Engine initialized with project: {self.project_id}")
            
            # Define Uzbekistan boundaries
            self.uzbekistan_bounds = ee.Geometry.Rectangle([55.9, 37.2, 73.2, 45.6])
            
            return True
        except Exception as e:
            print(f"❌ GEE initialization failed: {e}")
            return False
    
    def create_analysis_grid(self, resolution_km=10):
        """Create analysis grid entirely on GEE server"""
        print(f"\n🗺️  Creating {resolution_km}km analysis grid on GEE server...")
        
        # Convert resolution to degrees (approximately)
        resolution_deg = resolution_km / 111.0  # 1 degree ≈ 111 km
        
        # Create server-side grid
        # This creates a grid image where each pixel represents a grid cell
        grid_image = ee.Image.pixelLonLat().select(['longitude', 'latitude'])
        
        # Round coordinates to create grid cells
        grid_coords = grid_image.multiply(1/resolution_deg).round().multiply(resolution_deg)
        
        # Mask to Uzbekistan boundaries
        grid_masked = grid_coords.clipToBoundsAndScale(
            geometry=self.uzbekistan_bounds,
            scale=resolution_km * 1000  # Convert km to meters
        )
        
        print(f"✅ Grid created on server with ~{resolution_km}km resolution")
        return grid_masked
    
    def analyze_atmospheric_data_server_side(self, year=2024):
        """Run complete atmospheric analysis on GEE servers"""
        
        print(f"\n🛰️  Server-Side Atmospheric Analysis for {year}")
        print("=" * 55)
        
        # Define analysis period
        start_date = f'{year}-01-01'
        end_date = f'{year}-12-31'
        
        # Define atmospheric datasets
        datasets = {
            'CO': {
                'collection': 'COPERNICUS/S5P/OFFL/L3_CO',
                'band': 'CO_column_number_density',
                'scale': 5000,
                'description': 'Carbon Monoxide'
            },
            'NO2': {
                'collection': 'COPERNICUS/S5P/OFFL/L3_NO2',
                'band': 'tropospheric_NO2_column_number_density',
                'scale': 5000,
                'description': 'Nitrogen Dioxide'
            },
            'CH4': {
                'collection': 'COPERNICUS/S5P/OFFL/L3_CH4',
                'band': 'CH4_column_volume_mixing_ratio_dry_air',
                'scale': 5000,
                'description': 'Methane'
            },
            'SO2': {
                'collection': 'COPERNICUS/S5P/NRTI/L3_SO2',
                'band': 'SO2_column_number_density',
                'scale': 5000,
                'description': 'Sulfur Dioxide'
            }
        }
        
        # Progress tracking
        total_gases = len(datasets)
        completed = 0
        
        analysis_results = {}
        
        for gas, config in datasets.items():
            try:
                print(f"\n📡 Processing {config['description']} ({gas})...")
                print(f"Progress: {completed+1}/{total_gases}")
                
                # Load collection on server
                collection = ee.ImageCollection(config['collection']) \
                    .filterDate(start_date, end_date) \
                    .filterBounds(self.uzbekistan_bounds) \
                    .select(config['band'])
                
                # Check data availability
                collection_size = collection.size()
                size_info = collection_size.getInfo()
                print(f"   📊 Found {size_info} images in collection")
                
                if size_info > 0:
                    # Server-side temporal aggregation
                    print("   🔄 Computing temporal statistics on server...")
                    
                    # Calculate statistics entirely on server
                    mean_concentration = collection.mean()
                    min_concentration = collection.min()
                    max_concentration = collection.max()
                    std_concentration = collection.reduce(ee.Reducer.stdDev())
                    
                    # Server-side spatial statistics for Uzbekistan
                    print("   📈 Computing spatial statistics on server...")
                    
                    # Reduce region to get country-level statistics
                    stats = mean_concentration.reduceRegion(
                        reducer=ee.Reducer.mean().combine(
                            reducer2=ee.Reducer.minMax(),
                            sharedInputs=True
                        ).combine(
                            reducer2=ee.Reducer.stdDev(),
                            sharedInputs=True
                        ),
                        geometry=self.uzbekistan_bounds,
                        scale=config['scale'],
                        maxPixels=1e9
                    )
                    
                    # Get statistics from server
                    stats_dict = stats.getInfo()
                    print(f"   ✅ Extracted country-level statistics")
                    
                    # Server-side regional analysis
                    print("   🌍 Computing regional analysis on server...")
                    
                    # Define major regions within Uzbekistan
                    regions = {
                        'Western': ee.Geometry.Rectangle([55.9, 37.2, 63.0, 45.6]),
                        'Central': ee.Geometry.Rectangle([63.0, 37.2, 69.0, 45.6]),
                        'Eastern': ee.Geometry.Rectangle([69.0, 37.2, 73.2, 45.6])
                    }
                    
                    regional_stats = {}
                    for region_name, region_geom in regions.items():
                        region_stats = mean_concentration.reduceRegion(
                            reducer=ee.Reducer.mean().combine(
                                reducer2=ee.Reducer.count(),
                                sharedInputs=True
                            ),
                            geometry=region_geom,
                            scale=config['scale'],
                            maxPixels=1e9
                        )
                        regional_stats[region_name] = region_stats.getInfo()
                    
                    # Store results
                    analysis_results[gas] = {
                        'gas_name': config['description'],
                        'band_name': config['band'],
                        'collection_size': size_info,
                        'country_stats': stats_dict,
                        'regional_stats': regional_stats,
                        'analysis_scale_m': config['scale'],
                        'temporal_coverage': f"{start_date} to {end_date}"
                    }
                    
                    print(f"   ✅ {gas} analysis complete")
                    
                else:
                    print(f"   ⚠️  No {gas} data available for {year}")
                
                completed += 1
                progress_pct = (completed / total_gases) * 100
                print(f"   📊 Overall progress: {progress_pct:.1f}%")
                
            except Exception as e:
                print(f"   ❌ Error processing {gas}: {str(e)[:100]}...")
                completed += 1
        
        self.results = analysis_results
        return len(analysis_results) > 0
    
    def extract_city_concentrations_server_side(self):
        """Extract concentrations for major cities using server-side processing"""
        
        print(f"\n🏙️  Server-Side City Analysis")
        print("=" * 35)
        
        # Define major cities
        cities = {
            'Tashkent': [69.2401, 41.2995],
            'Samarkand': [66.9597, 39.6270],
            'Namangan': [71.6726, 40.9983],
            'Andijan': [72.3442, 40.7821],
            'Bukhara': [64.4207, 39.7747],
            'Nukus': [59.6103, 42.4531],
            'Qarshi': [65.7887, 38.8569],
            'Kokand': [70.9428, 40.5258],
            'Margilan': [71.7246, 40.4731],
            'Urgench': [60.6348, 41.5500]
        }
        
        print(f"📍 Analyzing {len(cities)} major cities")
        
        # Create city points collection on server
        city_features = []
        for city_name, coords in cities.items():
            feature = ee.Feature(
                ee.Geometry.Point(coords),
                {'city': city_name, 'longitude': coords[0], 'latitude': coords[1]}
            )
            city_features.append(feature)
        
        city_collection = ee.FeatureCollection(city_features)
        
        # Extract concentrations for each gas at city locations
        city_results = {}
        
        for gas, result_data in self.results.items():
            if 'collection_size' in result_data and result_data['collection_size'] > 0:
                try:
                    print(f"   📊 Extracting {gas} for cities...")
                    
                    # Recreate the mean concentration image on server
                    collection = ee.ImageCollection(
                        result_data['band_name'].split('/')[0] + '/' + 
                        result_data['band_name'].split('/')[1] + '/' +
                        result_data['band_name'].split('/')[2]
                    ).filterDate('2024-01-01', '2024-12-31') \
                     .filterBounds(self.uzbekistan_bounds) \
                     .select(result_data['band_name'].split('/')[-1])
                    
                    mean_image = collection.mean()
                    
                    # Sample at city points
                    sampled = mean_image.sampleRegions(
                        collection=city_collection,
                        scale=5000,
                        projection='EPSG:4326'
                    )
                    
                    # Extract results
                    features = sampled.getInfo()['features']
                    
                    gas_city_data = {}
                    for feature in features:
                        props = feature['properties']
                        city = props['city']
                        concentration = props.get(result_data['band_name'].split('/')[-1])
                        
                        if concentration is not None:
                            gas_city_data[city] = {
                                'concentration': concentration,
                                'longitude': props['longitude'],
                                'latitude': props['latitude']
                            }
                    
                    city_results[gas] = gas_city_data
                    print(f"   ✅ Extracted {gas} for {len(gas_city_data)} cities")
                    
                except Exception as e:
                    print(f"   ❌ Error extracting {gas} for cities: {str(e)[:80]}...")
        
        return city_results
    
    def generate_comprehensive_report(self, city_results):
        """Generate comprehensive analysis report"""
        
        print(f"\n📄 Generating Comprehensive Report")
        print("=" * 40)
        
        # Create output directory
        output_dir = Path('outputs')
        output_dir.mkdir(exist_ok=True)
        
        # Save detailed results
        detailed_results = {
            'analysis_metadata': {
                'analysis_date': datetime.now().isoformat(),
                'analysis_type': 'server_side_atmospheric',
                'study_area': 'Uzbekistan',
                'data_source': 'Sentinel-5P',
                'processing_location': 'Google Earth Engine Servers',
                'spatial_resolution': '5km',
                'temporal_coverage': '2024'
            },
            'country_level_statistics': self.results,
            'city_level_concentrations': city_results
        }
        
        # Save as JSON
        results_file = output_dir / 'server_side_atmospheric_analysis.json'
        with open(results_file, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        # Create CSV for city data
        if city_results:
            city_rows = []
            all_gases = list(city_results.keys())
            
            # Get all cities
            all_cities = set()
            for gas_data in city_results.values():
                all_cities.update(gas_data.keys())
            
            for city in all_cities:
                row = {'city': city}
                
                # Add coordinates from first available gas
                for gas_data in city_results.values():
                    if city in gas_data:
                        row['longitude'] = gas_data[city]['longitude']
                        row['latitude'] = gas_data[city]['latitude']
                        break
                
                # Add concentrations for each gas
                for gas in all_gases:
                    if city in city_results[gas]:
                        row[f'{gas}_concentration'] = city_results[gas][city]['concentration']
                    else:
                        row[f'{gas}_concentration'] = None
                
                city_rows.append(row)
            
            city_df = pd.DataFrame(city_rows)
            city_file = output_dir / 'server_side_city_concentrations.csv'
            city_df.to_csv(city_file, index=False)
            
            print(f"💾 Results saved:")
            print(f"   Detailed: {results_file}")
            print(f"   City Data: {city_file}")
            
            # Print summary
            print(f"\n📊 ANALYSIS SUMMARY:")
            print(f"   Cities analyzed: {len(city_df)}")
            print(f"   Gases detected: {len(all_gases)}")
            print(f"   Data processing: 100% server-side")
            
            return results_file, city_file
        
        return results_file, None

def main():
    """Main server-side analysis function"""
    
    print("🚀 SERVER-SIDE ATMOSPHERIC ANALYSIS - UZBEKISTAN")
    print("=" * 60)
    print("🖥️  All processing runs on Google Earth Engine servers")
    print("📡 Only final results are downloaded")
    print("⚡ Much faster than client-side processing")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = ServerSideAtmosphericAnalyzer()
    
    if not analyzer.initialize():
        print("❌ Failed to initialize")
        return False
    
    # Run server-side atmospheric analysis
    print(f"\n⏱️  Starting server-side analysis...")
    start_time = time.time()
    
    if not analyzer.analyze_atmospheric_data_server_side(year=2024):
        print("❌ Server-side analysis failed")
        return False
    
    # Extract city-level data
    city_results = analyzer.extract_city_concentrations_server_side()
    
    # Generate reports
    results_file, city_file = analyzer.generate_comprehensive_report(city_results)
    
    # Calculate processing time
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"\n🎉 SERVER-SIDE ANALYSIS COMPLETE!")
    print(f"⏱️  Total processing time: {processing_time:.1f} seconds")
    print(f"🖥️  All heavy computation done on GEE servers")
    print(f"📊 Results processed for entire Uzbekistan")
    print(f"💾 Final data downloaded: <1MB")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Server-side analysis failed")
    else:
        print("\n✅ Check the 'outputs' directory for comprehensive results!")
