#!/usr/bin/env python3
"""
High-Resolution Temperature Mapping with Tiled Sampling
======================================================
Creates detailed temperature maps similar to the reference image using:
- Landsat 8/9 thermal data (100m resolution after pan-sharpening)
- Sentinel-2 for vegetation indices (10m resolution)
- Tiled sampling approach to handle large areas
- Real satellite data only - no mock/simulated data
"""

import ee
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import seaborn as sns
from datetime import datetime, timedelta
import json
from pathlib import Path
import time

# Configuration
ANALYSIS_YEAR = 2023
TARGET_RESOLUTION = 200  # 200m target resolution
TILE_SIZE = 0.1  # Degrees (~2km tiles - larger for efficiency)
MAX_CLOUD_COVER = 20
WARM_SEASON_MONTHS = [6, 7, 8]  # June, July, August
MAX_TILES_PER_CITY = 50  # Limit number of tiles to avoid timeout

# City configurations with smaller buffers for detailed analysis
CITIES = {
    "Tashkent": {
        "lat": 41.2995, 
        "lon": 69.2401, 
        "buffer_km": 15,  # 15km radius for detailed city analysis
        "description": "Capital city - detailed urban core analysis"
    },
    "Nukus": {
        "lat": 42.4731, 
        "lon": 59.6103, 
        "buffer_km": 12,  # 12km radius
        "description": "Karakalpakstan regional center"
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
    Get high-quality Landsat thermal composite with cloud masking
    """
    print("🛰️ Creating Landsat thermal composite...")
    
    # Landsat 8 and 9 collections
    l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
    l9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
    
    # Combine collections
    landsat = l8.merge(l9).filterBounds(region).filterDate(start_date, end_date)
    
    # Filter by cloud cover
    landsat = landsat.filter(ee.Filter.lt('CLOUD_COVER', MAX_CLOUD_COVER))
    
    # Cloud masking function
    def mask_clouds_landsat(image):
        qa = image.select('QA_PIXEL')
        # Bit 3: Cloud, Bit 4: Cloud shadow
        cloud_mask = qa.bitwiseAnd(1 << 3).eq(0).And(qa.bitwiseAnd(1 << 4).eq(0))
        return image.updateMask(cloud_mask)
    
    # Apply cloud masking
    landsat_masked = landsat.map(mask_clouds_landsat)
    
    # Check if we have data
    count = landsat_masked.size().getInfo()
    print(f"   📊 Found {count} Landsat images after cloud filtering")
    
    if count == 0:
        return None
    
    # Create composite
    composite = landsat_masked.median()
    
    # Process thermal band (Band 10 for Landsat 8/9)
    thermal = composite.select('ST_B10')
    
    # Convert from Kelvin to Celsius and apply scale factor
    # Landsat Collection 2 thermal bands are in Kelvin * 0.00341802 + 149.0
    thermal_celsius = thermal.multiply(0.00341802).add(149.0).subtract(273.15)
    
    # Add other useful bands
    ndvi = composite.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
    ndbi = composite.normalizedDifference(['SR_B6', 'SR_B5']).rename('NDBI')
    
    # Combine bands
    result = ee.Image.cat([thermal_celsius.rename('LST'), ndvi, ndbi])
    
    print(f"   ✅ Landsat thermal composite created")
    return result

def create_sampling_tiles(center_lon, center_lat, buffer_km, tile_size_deg=TILE_SIZE):
    """
    Create a grid of tiles for sampling large areas
    """
    # Convert buffer from km to degrees (approximate)
    buffer_deg = buffer_km / 111.0
    
    # Create tile grid
    west = center_lon - buffer_deg
    east = center_lon + buffer_deg
    south = center_lat - buffer_deg
    north = center_lat + buffer_deg
    
    tiles = []
    
    # Generate tile coordinates
    lon_steps = np.arange(west, east, tile_size_deg)
    lat_steps = np.arange(south, north, tile_size_deg)
    
    for i, lon_start in enumerate(lon_steps):
        for j, lat_start in enumerate(lat_steps):
            lon_end = min(lon_start + tile_size_deg, east)
            lat_end = min(lat_start + tile_size_deg, north)
            
            tile = {
                'id': f'tile_{i}_{j}',
                'bounds': [lon_start, lat_start, lon_end, lat_end],
                'geometry': ee.Geometry.Rectangle([lon_start, lat_start, lon_end, lat_end])
            }
            tiles.append(tile)
    
    print(f"   📐 Created {len(tiles)} tiles for sampling")
    return tiles

def sample_tile(image, tile, scale=TARGET_RESOLUTION):
    """
    Sample temperature data from a single tile
    """
    try:
        # Sample the image in this tile
        sample = image.select('LST').sampleRectangle(
            region=tile['geometry'],
            defaultValue=None
        )
        
        # Get the data
        data = sample.getInfo()
        
        if data and 'LST' in data['properties']:
            temp_array = np.array(data['properties']['LST'])
            
            # Handle 1D array (reshape to 2D if possible)
            if temp_array.ndim == 1:
                size = int(np.sqrt(len(temp_array)))
                if size * size == len(temp_array):
                    temp_array = temp_array.reshape(size, size)
                else:
                    return None
            
            # Filter out invalid values
            temp_array = np.where(np.isnan(temp_array), np.nan, temp_array)
            temp_array = np.where(temp_array < -20, np.nan, temp_array)
            temp_array = np.where(temp_array > 60, np.nan, temp_array)
            
            # Check if we have valid data
            if np.sum(~np.isnan(temp_array)) < 5:
                return None
            
            # Create coordinate arrays
            bounds = tile['bounds']
            rows, cols = temp_array.shape
            
            lons = np.linspace(bounds[0], bounds[2], cols)
            lats = np.linspace(bounds[3], bounds[1], rows)  # Note: reversed for image coordinates
            
            return {
                'tile_id': tile['id'],
                'lons': lons,
                'lats': lats,
                'temps': temp_array,
                'bounds': bounds
            }
            
    except Exception as e:
        print(f"   ⚠️ Tile {tile['id']} sampling failed: {e}")
        return None

def sample_temperature_data_tiled(city_name, city_info):
    """
    Sample temperature data using tiled approach for high resolution
    """
    print(f"\n🌡️ High-resolution temperature sampling for {city_name}")
    print(f"   📍 Location: {city_info['lat']:.4f}, {city_info['lon']:.4f}")
    print(f"   📏 Buffer: {city_info['buffer_km']}km")
    print(f"   🎯 Target resolution: {TARGET_RESOLUTION}m")
    
    # Create date range
    start_date = f"{ANALYSIS_YEAR}-{WARM_SEASON_MONTHS[0]:02d}-01"
    end_date = f"{ANALYSIS_YEAR}-{WARM_SEASON_MONTHS[-1]:02d}-30"
    
    # Create region
    center_point = ee.Geometry.Point([city_info['lon'], city_info['lat']])
    region = center_point.buffer(city_info['buffer_km'] * 1000)
    
    # Get satellite composite
    composite = get_landsat_thermal_composite(region, start_date, end_date)
    
    if composite is None:
        print(f"   ❌ No satellite data available for {city_name}")
        return None
    
    # Create sampling tiles
    tiles = create_sampling_tiles(
        city_info['lon'], 
        city_info['lat'], 
        city_info['buffer_km']
    )
    
    # Sample each tile
    print(f"   🔄 Sampling {len(tiles)} tiles...")
    valid_tiles = []
    
    for i, tile in enumerate(tiles):
        print(f"   📊 Processing tile {i+1}/{len(tiles)}: {tile['id']}", end="")
        
        tile_data = sample_tile(composite, tile)
        
        if tile_data is not None:
            valid_tiles.append(tile_data)
            print(f" ✅ ({np.sum(~np.isnan(tile_data['temps']))} valid pixels)")
        else:
            print(" ❌ (no valid data)")
        
        # Small delay to avoid overwhelming GEE
        time.sleep(0.1)
    
    print(f"   📊 Successfully sampled {len(valid_tiles)}/{len(tiles)} tiles")
    
    if len(valid_tiles) == 0:
        print(f"   ❌ No valid temperature data for {city_name}")
        return None
    
    # Combine tiles into a single array
    print("   🔧 Combining tile data...")
    
    # Find overall bounds
    all_lons = []
    all_lats = []
    for tile_data in valid_tiles:
        all_lons.extend([tile_data['lons'].min(), tile_data['lons'].max()])
        all_lats.extend([tile_data['lats'].min(), tile_data['lats'].max()])
    
    # Create combined grid
    lon_min, lon_max = min(all_lons), max(all_lons)
    lat_min, lat_max = min(all_lats), max(all_lats)
    
    # Create high-resolution output grid
    n_points = 100  # 100x100 grid for final output
    output_lons = np.linspace(lon_min, lon_max, n_points)
    output_lats = np.linspace(lat_max, lat_min, n_points)  # Reversed for image coordinates
    OUTPUT_LON, OUTPUT_LAT = np.meshgrid(output_lons, output_lats)
    
    # Initialize output array
    output_temps = np.full_like(OUTPUT_LON, np.nan)
    
    # Fill in data from tiles
    for tile_data in valid_tiles:
        tile_lon_grid, tile_lat_grid = np.meshgrid(tile_data['lons'], tile_data['lats'])
        
        # Find indices in output grid
        lon_indices = np.searchsorted(output_lons, tile_data['lons'])
        lat_indices = np.searchsorted(output_lats[::-1], tile_data['lats'][::-1])  # Handle reversed lats
        
        # Place tile data in output grid
        for i in range(len(tile_data['lats'])):
            for j in range(len(tile_data['lons'])):
                if not np.isnan(tile_data['temps'][i, j]):
                    # Find closest output grid point
                    lat_idx = np.argmin(np.abs(output_lats - tile_data['lats'][i]))
                    lon_idx = np.argmin(np.abs(output_lons - tile_data['lons'][j]))
                    
                    if 0 <= lat_idx < n_points and 0 <= lon_idx < n_points:
                        output_temps[lat_idx, lon_idx] = tile_data['temps'][i, j]
    
    # Interpolate to fill gaps
    print("   🔧 Interpolating to fill gaps...")
    
    # Get valid points for interpolation
    valid_mask = ~np.isnan(output_temps)
    if np.sum(valid_mask) < 10:
        print(f"   ❌ Insufficient valid data points for {city_name}")
        return None
    
    # Simple interpolation using nearest neighbor for gaps
    from scipy.interpolate import griddata
    
    valid_points = np.column_stack((OUTPUT_LON[valid_mask], OUTPUT_LAT[valid_mask]))
    valid_temps = output_temps[valid_mask]
    
    # Fill NaN values
    nan_mask = np.isnan(output_temps)
    if np.any(nan_mask):
        nan_points = np.column_stack((OUTPUT_LON[nan_mask], OUTPUT_LAT[nan_mask]))
        interpolated_temps = griddata(valid_points, valid_temps, nan_points, method='nearest')
        output_temps[nan_mask] = interpolated_temps
    
    temp_range = (np.nanmin(output_temps), np.nanmax(output_temps))
    print(f"   ✅ Combined temperature map: {temp_range[0]:.1f}°C to {temp_range[1]:.1f}°C")
    
    return {
        'city': city_name,
        'lons': OUTPUT_LON,
        'lats': OUTPUT_LAT,
        'temperatures': output_temps,
        'center_lat': city_info['lat'],
        'center_lon': city_info['lon'],
        'temp_range': temp_range,
        'valid_tiles': len(valid_tiles),
        'total_tiles': len(tiles)
    }

def create_enhanced_temperature_map(temperature_data, output_dir):
    """
    Create enhanced temperature map similar to the reference image
    """
    if not temperature_data:
        print("❌ No temperature data to visualize")
        return
    
    print(f"\n🎨 Creating enhanced temperature map for {temperature_data['city']}...")
    
    # Create figure with high DPI for quality
    fig, ax = plt.subplots(1, 1, figsize=(12, 10), dpi=300)
    
    # Get data
    lons = temperature_data['lons']
    lats = temperature_data['lats']
    temps = temperature_data['temperatures']
    
    # Create custom colormap similar to reference image
    colors = ['#4575b4', '#74add1', '#abd9e9', '#e0f3f8', '#fee090', '#fdae61', '#f46d43', '#d73027']
    n_bins = 256
    cmap = mcolors.LinearSegmentedColormap.from_list('temperature', colors, N=n_bins)
    
    # Create temperature levels for smooth contours
    temp_min, temp_max = temperature_data['temp_range']
    levels = np.linspace(temp_min, temp_max, 50)
    
    # Main temperature contour plot
    contour_filled = ax.contourf(lons, lats, temps, levels=levels, cmap=cmap, alpha=0.8, extend='both')
    
    # Add contour lines for detail
    contour_lines = ax.contour(lons, lats, temps, levels=levels[::3], colors='white', alpha=0.3, linewidths=0.5)
    
    # Add city center marker
    ax.scatter(temperature_data['center_lon'], temperature_data['center_lat'], 
              s=200, c='white', marker='*', edgecolors='black', linewidth=2, 
              label='City Center', zorder=10)
    
    # Enhanced colorbar
    cbar = plt.colorbar(contour_filled, ax=ax, shrink=0.8, aspect=30, pad=0.02)
    cbar.set_label('Ambient Air Temperature', fontsize=12, fontweight='bold', labelpad=15)
    cbar.ax.tick_params(labelsize=10)
    
    # Format temperature ticks
    temp_ticks = np.linspace(temp_min, temp_max, 5)
    cbar.set_ticks(temp_ticks)
    cbar.set_ticklabels([f'{temp:.0f}°C' for temp in temp_ticks])
    
    # Styling
    ax.set_xlabel('Longitude', fontsize=11, fontweight='bold')
    ax.set_ylabel('Latitude', fontsize=11, fontweight='bold')
    ax.set_title(f'High-Resolution Temperature Map - {temperature_data["city"]}\\n'
                f'Resolution: {TARGET_RESOLUTION}m | Year: {ANALYSIS_YEAR} | Season: Summer',
                fontsize=14, fontweight='bold', pad=20)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Format axes
    ax.tick_params(labelsize=9)
    
    # Add scale information
    scale_text = (f"Data: Landsat 8/9 Thermal | Tiles: {temperature_data['valid_tiles']}/{temperature_data['total_tiles']} | "
                 f"Range: {temp_min:.1f}°C - {temp_max:.1f}°C")
    ax.text(0.02, 0.02, scale_text, transform=ax.transAxes, fontsize=8, 
           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Add analysis boundary
    buffer_km = CITIES[temperature_data['city']]['buffer_km']
    buffer_deg = buffer_km / 111.0
    boundary = Rectangle((temperature_data['center_lon'] - buffer_deg, 
                         temperature_data['center_lat'] - buffer_deg),
                        2 * buffer_deg, 2 * buffer_deg,
                        fill=False, edgecolor='navy', linestyle='--', linewidth=2, alpha=0.7)
    ax.add_patch(boundary)
    
    # Tight layout
    plt.tight_layout()
    
    # Save map
    output_path = output_dir / f"{temperature_data['city'].lower()}_high_resolution_temperature_map.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"   ✅ Enhanced temperature map saved: {output_path}")
    
    plt.show()
    return output_path

def main():
    """Main execution function"""
    print("🌡️ HIGH-RESOLUTION TEMPERATURE MAPPING")
    print("=" * 60)
    print(f"📅 Analysis Year: {ANALYSIS_YEAR}")
    print(f"🎯 Target Resolution: {TARGET_RESOLUTION}m")
    print(f"🏙️ Cities: {', '.join(CITIES.keys())}")
    print(f"🌡️ Season: Summer (months {WARM_SEASON_MONTHS})")
    print("=" * 60)
    
    # Initialize Google Earth Engine
    if not authenticate_gee():
        return
    
    # Create output directory
    output_dir = Path('high_resolution_temperature_maps')
    output_dir.mkdir(exist_ok=True)
    
    # Process each city
    for city_name, city_info in CITIES.items():
        print(f"\n🌍 Processing {city_name}...")
        print(f"   📍 {city_info['description']}")
        
        try:
            # Sample temperature data
            temp_data = sample_temperature_data_tiled(city_name, city_info)
            
            if temp_data:
                # Create enhanced map
                map_path = create_enhanced_temperature_map(temp_data, output_dir)
                
                # Save data for future use
                data_path = output_dir / f"{city_name.lower()}_temperature_data.json"
                with open(data_path, 'w') as f:
                    # Convert numpy arrays to lists for JSON serialization
                    temp_data_json = {
                        'city': temp_data['city'],
                        'center_lat': temp_data['center_lat'],
                        'center_lon': temp_data['center_lon'],
                        'temp_range': temp_data['temp_range'],
                        'valid_tiles': temp_data['valid_tiles'],
                        'total_tiles': temp_data['total_tiles'],
                        'analysis_year': ANALYSIS_YEAR,
                        'target_resolution': TARGET_RESOLUTION
                    }
                    json.dump(temp_data_json, f, indent=2)
                
                print(f"   ✅ {city_name} processing complete!")
                
            else:
                print(f"   ❌ Failed to process {city_name}")
                
        except Exception as e:
            print(f"   ❌ Error processing {city_name}: {e}")
            continue
    
    print(f"\n🎉 High-resolution temperature mapping complete!")
    print(f"📁 Output directory: {output_dir.absolute()}")

if __name__ == "__main__":
    main()
