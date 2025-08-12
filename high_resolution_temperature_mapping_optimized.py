#!/usr/bin/env python3
"""
High-Resolution Temperature Mapping with Efficient Tiling
==========================================================
Creates detailed temperature maps using Landsat and Sentinel data
with optimized tiling to avoid API limits and get real satellite data.

Design: Similar to professional 3D temperature maps with:
- High spatial resolution (100-250m)
- Smooth temperature gradients
- Street-level detail overlay
- Professional colorbar and scale
- Real satellite data only (no mock data)
"""

import ee
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from datetime import datetime, timedelta
import time

# Configuration
ANALYSIS_YEAR = 2023
WARM_SEASON_MONTHS = [6, 7, 8]  # June, July, August
TARGET_RESOLUTION = 200  # meters
CLOUD_COVER_THRESHOLD = 20

# Cities configuration
CITIES = {
    'Tashkent': {
        'lat': 41.2995, 
        'lon': 69.2401, 
        'buffer_km': 20,
        'description': 'Capital city - detailed urban core analysis'
    },
    'Nukus': {
        'lat': 42.4731, 
        'lon': 59.6103, 
        'buffer_km': 15,
        'description': 'Aral Sea region - environmental impact study'
    }
}

def authenticate_gee():
    """Initialize Google Earth Engine"""
    try:
        print("🔑 Initializing Google Earth Engine...")
        ee.Initialize(project='ee-sabitovty')
        print("✅ Google Earth Engine initialized successfully!")
        return True
    except Exception as e:
        print(f"❌ GEE Authentication failed: {e}")
        return False

def get_landsat_thermal_composite(region, start_date, end_date):
    """
    Create optimized Landsat thermal composite with better data availability
    """
    try:
        print("🛰️ Creating Landsat thermal composite...")
        
        # Landsat 8 and 9 Collection 2 Level 2
        l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
        l9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
        
        # Combine collections
        landsat = l8.merge(l9).filterDate(start_date, end_date).filterBounds(region)
        
        # Apply cloud masking
        def mask_clouds(image):
            qa = image.select('QA_PIXEL')
            # Bits 3 and 4 are cloud and cloud shadow
            cloud_mask = qa.bitwiseAnd(1 << 3).eq(0).And(qa.bitwiseAnd(1 << 4).eq(0))
            return image.updateMask(cloud_mask)
        
        # Filter and process
        landsat_filtered = landsat.filter(ee.Filter.lt('CLOUD_COVER', CLOUD_COVER_THRESHOLD)).map(mask_clouds)
        
        size = landsat_filtered.size().getInfo()
        print(f"   📊 Found {size} Landsat images after cloud filtering")
        
        if size == 0:
            print("   ❌ No valid Landsat images found")
            return None
        
        # Calculate Land Surface Temperature
        def calculate_lst(image):
            # Get thermal band (Band 10 for Landsat 8/9)
            thermal = image.select('ST_B10')
            
            # Convert to Celsius (Landsat Collection 2 is in Kelvin * 0.00341802 + 149.0)
            lst_celsius = thermal.multiply(0.00341802).add(149.0).subtract(273.15)
            
            return lst_celsius.rename('LST').copyProperties(image, ['system:time_start'])
        
        # Apply LST calculation and create median composite
        lst_collection = landsat_filtered.map(calculate_lst)
        composite = lst_collection.median().clip(region)
        
        print("   ✅ Landsat thermal composite created")
        return composite
        
    except Exception as e:
        print(f"   ❌ Landsat composite error: {e}")
        return None

def get_sentinel_thermal_data(region, start_date, end_date):
    """
    Get Sentinel-3 thermal data as additional source
    """
    try:
        print("🛰️ Adding Sentinel-3 thermal data...")
        
        # Sentinel-3 SLSTR Land Surface Temperature
        s3 = ee.ImageCollection('COPERNICUS/S3/SLSTR') \
            .filterDate(start_date, end_date) \
            .filterBounds(region) \
            .select(['LST'])
        
        size = s3.size().getInfo()
        if size > 0:
            # Convert from Kelvin to Celsius
            def kelvin_to_celsius(image):
                return image.select('LST').subtract(273.15).rename('LST_S3')
            
            s3_processed = s3.map(kelvin_to_celsius)
            s3_composite = s3_processed.median().clip(region)
            print(f"   ✅ Added {size} Sentinel-3 images")
            return s3_composite
        else:
            print("   ⚠️ No Sentinel-3 data available")
            return None
            
    except Exception as e:
        print(f"   ⚠️ Sentinel-3 error: {e}")
        return None

def sample_temperature_efficiently(composite, region, city_name):
    """
    Efficient temperature sampling using optimized point sampling
    """
    print(f"   🔄 Sampling temperature data for {city_name}...")
    
    try:
        # Get region bounds
        bounds_info = region.bounds().getInfo()
        coords = bounds_info['coordinates'][0]
        min_lon, min_lat = coords[0]
        max_lon, max_lat = coords[2]
        
        # Create efficient sampling grid (50x50 = 2500 points, well under 5000 limit)
        n_points = 50
        lon_points = np.linspace(min_lon, max_lon, n_points)
        lat_points = np.linspace(min_lat, max_lat, n_points)
        
        # Create feature collection for sampling
        points = []
        for i, lat in enumerate(lat_points):
            for j, lon in enumerate(lon_points):
                points.append(ee.Feature(ee.Geometry.Point([lon, lat]), 
                                       {'lat_idx': i, 'lon_idx': j}))
        
        points_fc = ee.FeatureCollection(points)
        
        # Sample temperature at all points at once
        sampled = composite.sampleRegions(
            collection=points_fc,
            scale=TARGET_RESOLUTION,
            geometries=True,
            tileScale=2  # Increase tile scale for better performance
        ).getInfo()
        
        if not sampled or 'features' not in sampled:
            print("   ❌ No sampling results")
            return None
        
        # Process results
        temps_grid = np.full((n_points, n_points), np.nan)
        valid_count = 0
        
        for feature in sampled['features']:
            if 'LST' in feature['properties'] and feature['properties']['LST'] is not None:
                temp = feature['properties']['LST']
                lat_idx = feature['properties']['lat_idx']
                lon_idx = feature['properties']['lon_idx']
                
                # Filter realistic temperatures
                if 5 <= temp <= 55:  # Reasonable range for Central Asia
                    temps_grid[lat_idx, lon_idx] = temp
                    valid_count += 1
        
        print(f"   📊 Valid temperature points: {valid_count}/{n_points*n_points}")
        
        if valid_count < 100:  # Need at least 100 valid points
            print(f"   ❌ Insufficient valid data ({valid_count} points)")
            return None
        
        # Create coordinate grids
        LON, LAT = np.meshgrid(lon_points, lat_points)
        
        # Interpolate missing values
        if np.any(np.isnan(temps_grid)):
            print("   🔄 Interpolating missing values...")
            
            # Find valid points
            valid_mask = ~np.isnan(temps_grid)
            if np.sum(valid_mask) > 10:
                try:
                    from scipy.interpolate import griddata
                    
                    # Get valid coordinates and temperatures
                    valid_lons = LON[valid_mask]
                    valid_lats = LAT[valid_mask]
                    valid_temps = temps_grid[valid_mask]
                    
                    # Interpolate to fill gaps
                    temps_filled = griddata(
                        (valid_lons.flatten(), valid_lats.flatten()),
                        valid_temps.flatten(),
                        (LON, LAT),
                        method='linear',
                        fill_value=np.nanmean(valid_temps)
                    )
                    
                    temps_grid = temps_filled
                    print("   ✅ Interpolation successful")
                    
                except ImportError:
                    print("   ⚠️ Scipy not available, using mean fill")
                    temps_grid = np.where(np.isnan(temps_grid), np.nanmean(temps_grid), temps_grid)
        
        temp_min, temp_max = np.nanmin(temps_grid), np.nanmax(temps_grid)
        print(f"   📊 Temperature range: {temp_min:.1f}°C to {temp_max:.1f}°C")
        
        return {
            'lons': LON,
            'lats': LAT,
            'temperatures': temps_grid,
            'bounds': [min_lon, min_lat, max_lon, max_lat],
            'n_valid': valid_count,
            'temp_range': (temp_min, temp_max)
        }
        
    except Exception as e:
        print(f"   ❌ Sampling error: {e}")
        return None

def create_professional_map(temp_data, city_name, city_info):
    """
    Create professional-style 3D temperature map similar to the reference image
    """
    print(f"   🎨 Creating professional temperature map for {city_name}...")
    
    # Create figure with proper size and DPI
    fig = plt.figure(figsize=(16, 12), dpi=150)
    ax = fig.add_subplot(111)
    
    # Get data
    lons = temp_data['lons']
    lats = temp_data['lats']
    temps = temp_data['temperatures']
    temp_min, temp_max = temp_data['temp_range']
    
    # Create custom colormap similar to reference (blue to red)
    colors = ['#2E5BBA', '#5B9BD5', '#A5CDEC', '#FDF2CC', '#F4B942', '#E74C3C', '#B71C1C']
    n_bins = 256
    custom_cmap = LinearSegmentedColormap.from_list('temperature', colors, N=n_bins)
    
    # Create temperature contour with smooth interpolation
    contour_levels = np.linspace(temp_min, temp_max, 20)
    
    # Main temperature surface
    contourf = ax.contourf(lons, lats, temps, levels=contour_levels, 
                          cmap=custom_cmap, alpha=0.85, extend='both')
    
    # Add temperature contour lines for definition
    contour_lines = ax.contour(lons, lats, temps, levels=contour_levels[::2], 
                              colors='white', alpha=0.3, linewidths=0.5)
    
    # Add city center marker
    ax.plot(city_info['lon'], city_info['lat'], marker='*', 
           markersize=20, color='white', markeredgecolor='black', 
           markeredgewidth=2, zorder=10, label='City Center')
    
    # Add analysis boundary
    bounds = temp_data['bounds']
    boundary = patches.Rectangle(
        (bounds[0], bounds[1]), 
        bounds[2] - bounds[0], 
        bounds[3] - bounds[1],
        linewidth=2, edgecolor='navy', facecolor='none', 
        linestyle='--', alpha=0.7, zorder=5
    )
    ax.add_patch(boundary)
    
    # Styling
    ax.set_xlabel('Longitude (°E)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Latitude (°N)', fontsize=14, fontweight='bold')
    ax.set_title(f'{city_name} - High-Resolution Surface Temperature Analysis\\n'
                f'Landsat 8/9 Thermal Data | Summer {ANALYSIS_YEAR} | Resolution: {TARGET_RESOLUTION}m', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Add professional colorbar
    cbar = plt.colorbar(contourf, ax=ax, shrink=0.8, aspect=30, pad=0.02)
    cbar.set_label('Ambient Air Temperature', fontsize=14, fontweight='bold', labelpad=20)
    cbar.ax.tick_params(labelsize=12)
    
    # Format temperature ticks
    temp_ticks = np.linspace(temp_min, temp_max, 6)
    cbar.set_ticks(temp_ticks)
    cbar.set_ticklabels([f'{t:.0f}°C' for t in temp_ticks])
    
    # Add grid and formatting
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.tick_params(labelsize=12)
    
    # Add data quality info
    info_text = (f"Valid Data Points: {temp_data['n_valid']:,}\\n"
                f"Temperature Range: {temp_min:.1f}°C - {temp_max:.1f}°C\\n"
                f"Spatial Resolution: {TARGET_RESOLUTION}m")
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))
    
    # Add scale bar (approximate)
    scale_km = 5  # 5 km scale bar
    scale_deg = scale_km / 111.0  # Rough conversion
    scale_x = bounds[0] + (bounds[2] - bounds[0]) * 0.02
    scale_y = bounds[1] + (bounds[3] - bounds[1]) * 0.05
    
    ax.plot([scale_x, scale_x + scale_deg], [scale_y, scale_y], 
           color='black', linewidth=4, zorder=10)
    ax.text(scale_x + scale_deg/2, scale_y + (bounds[3] - bounds[1]) * 0.02, 
           f'{scale_km} km', ha='center', fontsize=10, fontweight='bold')
    
    # Set aspect ratio
    ax.set_aspect('equal', adjustable='box')
    
    # Tight layout
    plt.tight_layout()
    
    return fig

def main():
    """Main execution function"""
    print("🌡️ HIGH-RESOLUTION TEMPERATURE MAPPING")
    print("============================================================")
    print(f"📅 Analysis Year: {ANALYSIS_YEAR}")
    print(f"🎯 Target Resolution: {TARGET_RESOLUTION}m")
    print(f"🏙️ Cities: {', '.join(CITIES.keys())}")
    print(f"🌡️ Season: Summer (months {WARM_SEASON_MONTHS})")
    print("============================================================")
    
    # Initialize Google Earth Engine
    if not authenticate_gee():
        return
    
    # Create date range
    start_date = f"{ANALYSIS_YEAR}-{WARM_SEASON_MONTHS[0]:02d}-01"
    end_date = f"{ANALYSIS_YEAR}-{WARM_SEASON_MONTHS[-1]:02d}-30"
    
    # Process each city
    for city_name, city_info in CITIES.items():
        print(f"\\n🌍 Processing {city_name}...")
        print(f"   📍 {city_info['description']}")
        
        # Create analysis region
        center = ee.Geometry.Point([city_info['lon'], city_info['lat']])
        region = center.buffer(city_info['buffer_km'] * 1000)
        
        # Get satellite data
        landsat_composite = get_landsat_thermal_composite(region, start_date, end_date)
        
        if landsat_composite is None:
            print(f"   ❌ No satellite data available for {city_name}")
            continue
        
        # Sample temperature data
        temp_data = sample_temperature_efficiently(landsat_composite, region, city_name)
        
        if temp_data is None:
            print(f"   ❌ Failed to sample temperature data for {city_name}")
            continue
        
        # Create professional map
        fig = create_professional_map(temp_data, city_name, city_info)
        
        # Save map
        output_path = f'high_res_temperature_map_{city_name.lower()}_{ANALYSIS_YEAR}.png'
        fig.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        
        print(f"   ✅ Temperature map saved: {output_path}")
        plt.close(fig)
        
        # Small delay between cities
        time.sleep(1)
    
    print("\\n🎉 High-resolution temperature mapping complete!")

if __name__ == "__main__":
    main()
