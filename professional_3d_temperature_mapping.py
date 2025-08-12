#!/usr/bin/env python3
"""
Professional 3D Temperature Mapping
====================================
Creates 3D-style temperature maps similar to the reference Tirana image with:
- Semi-transparent temperature overlay
- Street network basemap
- Professional 3D perspective
- Real satellite data from Landsat 8/9
"""

import ee
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.img_tiles import GoogleTiles, OSM
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import datetime, timedelta
import json
from pathlib import Path
import time

# Configuration
ANALYSIS_YEAR = 2023
TARGET_RESOLUTION = 200  # meters
WARM_SEASON_MONTHS = [6, 7, 8]  # Summer
MAX_CLOUD_COVER = 20

# Cities with focused analysis areas
CITIES = {
    "Tashkent": {
        "lat": 41.2995, 
        "lon": 69.2401, 
        "buffer_km": 15,
        "description": "Capital city - urban heat analysis"
    },
    "Nukus": {
        "lat": 42.4731, 
        "lon": 59.6103, 
        "buffer_km": 12,
        "description": "Aral Sea region - environmental impact"
    }
}

class OpenStreetMapTiles(GoogleTiles):
    """OpenStreetMap tile source for street network basemap"""
    def _image_url(self, tile):
        x, y, z = tile
        return f'https://tile.openstreetmap.org/{z}/{x}/{y}.png'

def authenticate_gee():
    """Initialize Google Earth Engine"""
    try:
        print("🔑 Initializing Google Earth Engine...")
        ee.Initialize(project='ee-sabitovty')
        print("✅ Google Earth Engine authenticated!")
        return True
    except Exception as e:
        print(f"❌ GEE Authentication failed: {e}")
        return False

def get_landsat_thermal_data(region, start_date, end_date):
    """
    Get high-quality Landsat thermal composite with enhanced processing
    """
    print("🛰️ Processing Landsat thermal data...")
    
    # Landsat 8 and 9 collections (Collection 2, Level 2)
    l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
    l9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
    
    # Combine and filter
    landsat = l8.merge(l9) \
                .filterBounds(region) \
                .filterDate(start_date, end_date) \
                .filter(ee.Filter.lt('CLOUD_COVER', MAX_CLOUD_COVER))
    
    def process_landsat_image(image):
        """Process individual Landsat image for LST"""
        # Cloud masking
        qa = image.select('QA_PIXEL')
        cloud_mask = qa.bitwiseAnd(1 << 3).eq(0).And(qa.bitwiseAnd(1 << 4).eq(0))
        
        # Apply cloud mask
        masked = image.updateMask(cloud_mask)
        
        # Calculate Land Surface Temperature from thermal band
        thermal = masked.select('ST_B10')
        # Convert to Celsius: Landsat Collection 2 thermal bands
        lst_celsius = thermal.multiply(0.00341802).add(149.0).subtract(273.15)
        
        # Calculate vegetation indices for analysis
        ndvi = masked.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
        ndbi = masked.normalizedDifference(['SR_B6', 'SR_B5']).rename('NDBI')
        
        return ee.Image.cat([lst_celsius.rename('LST'), ndvi, ndbi]) \
                      .copyProperties(image, ['system:time_start'])
    
    # Process collection
    processed = landsat.map(process_landsat_image)
    
    # Check data availability
    count = processed.size().getInfo()
    print(f"   📊 Found {count} quality Landsat scenes")
    
    if count == 0:
        print("   ❌ No valid Landsat data available")
        return None
    
    # Create temporal composite (median for stability)
    composite = processed.median().clip(region)
    
    print("   ✅ Landsat thermal composite created")
    return composite

def sample_temperature_data_efficiently(composite, region, city_name, resolution=200):
    """
    Sample temperature data using efficient point-based sampling
    """
    print(f"   🔄 Sampling temperature data for {city_name}...")
    
    try:
        # Get region bounds
        bounds = region.bounds().getInfo()
        coords = bounds['coordinates'][0]
        
        # Extract bounds
        lons = [coord[0] for coord in coords]
        lats = [coord[1] for coord in coords]
        min_lon, max_lon = min(lons), max(lons)
        min_lat, max_lat = min(lats), max(lats)
        
        # Create sampling grid (60x60 = 3600 points, well under API limits)
        n_points = 60
        lon_array = np.linspace(min_lon, max_lon, n_points)
        lat_array = np.linspace(min_lat, max_lat, n_points)
        
        # Create point features for sampling
        points = []
        for i, lat in enumerate(lat_array):
            for j, lon in enumerate(lon_array):
                point = ee.Feature(
                    ee.Geometry.Point([lon, lat]),
                    {'lat_idx': i, 'lon_idx': j}
                )
                points.append(point)
        
        points_collection = ee.FeatureCollection(points)
        
        # Sample all points at once
        sampled = composite.select('LST').sampleRegions(
            collection=points_collection,
            scale=resolution,
            geometries=True,
            tileScale=4  # Increase for better performance
        ).getInfo()
        
        if not sampled or 'features' not in sampled:
            print("   ❌ No sampling results returned")
            return None
        
        # Process results into grid
        temp_grid = np.full((n_points, n_points), np.nan)
        valid_count = 0
        
        for feature in sampled['features']:
            props = feature['properties']
            if 'LST' in props and props['LST'] is not None:
                temp = props['LST']
                lat_idx = props['lat_idx']
                lon_idx = props['lon_idx']
                
                # Filter realistic temperatures for Central Asia
                if 5 <= temp <= 60:
                    temp_grid[lat_idx, lon_idx] = temp
                    valid_count += 1
        
        print(f"   📊 Retrieved {valid_count}/{n_points*n_points} valid temperature points")
        
        if valid_count < 200:  # Need sufficient data
            print(f"   ❌ Insufficient valid data: {valid_count} points")
            return None
        
        # Create coordinate meshgrids
        LON_GRID, LAT_GRID = np.meshgrid(lon_array, lat_array)
        
        # Interpolate missing values using scipy if available
        if np.any(np.isnan(temp_grid)):
            try:
                from scipy.interpolate import griddata
                
                # Get valid data points
                valid_mask = ~np.isnan(temp_grid)
                valid_lons = LON_GRID[valid_mask]
                valid_lats = LAT_GRID[valid_mask]
                valid_temps = temp_grid[valid_mask]
                
                # Interpolate to fill gaps
                nan_mask = np.isnan(temp_grid)
                if np.any(nan_mask):
                    nan_lons = LON_GRID[nan_mask]
                    nan_lats = LAT_GRID[nan_mask]
                    
                    interpolated = griddata(
                        (valid_lons, valid_lats), valid_temps,
                        (nan_lons, nan_lats), method='linear',
                        fill_value=np.nanmean(valid_temps)
                    )
                    
                    temp_grid[nan_mask] = interpolated
                    print("   ✅ Gaps filled using interpolation")
                
            except ImportError:
                # Simple gap filling with mean
                temp_grid = np.where(np.isnan(temp_grid), 
                                   np.nanmean(temp_grid), temp_grid)
                print("   ⚠️ Basic gap filling applied")
        
        # Calculate statistics
        temp_min, temp_max = np.nanmin(temp_grid), np.nanmax(temp_grid)
        temp_mean = np.nanmean(temp_grid)
        
        print(f"   📊 Temperature statistics:")
        print(f"      Range: {temp_min:.1f}°C to {temp_max:.1f}°C")
        print(f"      Mean: {temp_mean:.1f}°C")
        
        return {
            'lons': LON_GRID,
            'lats': LAT_GRID,
            'temperatures': temp_grid,
            'bounds': [min_lon, min_lat, max_lon, max_lat],
            'temp_range': (temp_min, temp_max),
            'temp_mean': temp_mean,
            'valid_points': valid_count,
            'total_points': n_points * n_points
        }
        
    except Exception as e:
        print(f"   ❌ Temperature sampling error: {e}")
        return None

def create_3d_professional_map(temp_data, city_name, city_info):
    """
    Create professional 3D-style temperature map similar to Tirana reference
    """
    print(f"   🎨 Creating professional 3D temperature map for {city_name}...")
    
    # Create figure with high DPI
    fig = plt.figure(figsize=(16, 12), dpi=200, facecolor='white')
    
    # Use Cartopy for geographic projections
    projection = ccrs.PlateCarree()
    ax = fig.add_subplot(111, projection=projection)
    
    # Get data
    lons = temp_data['lons']
    lats = temp_data['lats']
    temps = temp_data['temperatures']
    bounds = temp_data['bounds']
    temp_min, temp_max = temp_data['temp_range']
    
    # Set map extent
    ax.set_extent(bounds, crs=projection)
    
    # Add basemap features for context
    try:
        # Add street network basemap (OpenStreetMap style)
        osm_tiles = OSM()
        ax.add_image(osm_tiles, 11, alpha=0.6)  # Streets and roads visible
        print("   ✅ Added OpenStreetMap basemap")
    except:
        # Fallback to basic features
        ax.add_feature(cfeature.LAND, alpha=0.3, color='lightgray')
        ax.add_feature(cfeature.OCEAN, alpha=0.3, color='lightblue')
        ax.add_feature(cfeature.RIVERS, alpha=0.5, color='blue', linewidth=0.5)
        ax.add_feature(cfeature.ROADS, alpha=0.3, color='gray', linewidth=0.3)
        print("   ⚠️ Using basic cartographic features")
    
    # Create custom colormap matching reference (blue to red gradient)
    colors = [
        '#2E4A99',  # Deep blue (coolest)
        '#4B73B5',  # Medium blue  
        '#7BA3D1',  # Light blue
        '#B8D4ED',  # Very light blue
        '#FFFFFF',  # White (neutral)
        '#FDE5CC',  # Very light orange
        '#FDB462',  # Light orange
        '#F4842A',  # Medium orange
        '#E63946',  # Red
        '#B71C1C'   # Dark red (hottest)
    ]
    
    n_colors = 256
    custom_cmap = mcolors.LinearSegmentedColormap.from_list(
        'professional_temperature', colors, N=n_colors
    )
    
    # Create smooth temperature levels
    n_levels = 40
    temp_levels = np.linspace(temp_min, temp_max, n_levels)
    
    # Main temperature surface with transparency
    temp_surface = ax.contourf(
        lons, lats, temps, 
        levels=temp_levels,
        cmap=custom_cmap,
        alpha=0.75,  # Semi-transparent like reference
        extend='both',
        transform=projection
    )
    
    # Add subtle contour lines for definition
    contour_lines = ax.contour(
        lons, lats, temps,
        levels=temp_levels[::4],  # Every 4th level
        colors='white',
        alpha=0.4,
        linewidths=0.3,
        transform=projection
    )
    
    # Add city center marker (prominent like reference)
    ax.plot(
        city_info['lon'], city_info['lat'],
        marker='*', markersize=25,
        color='white', markeredgecolor='black',
        markeredgewidth=3, zorder=15,
        transform=projection,
        label='City Center'
    )
    
    # Create professional colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1, axes_class=plt.Axes)
    
    cbar = plt.colorbar(temp_surface, cax=cax, shrink=0.9)
    cbar.set_label(
        'Ambient Air Temperature', 
        fontsize=14, fontweight='bold', 
        labelpad=20
    )
    
    # Format colorbar ticks
    temp_ticks = np.linspace(temp_min, temp_max, 6)
    cbar.set_ticks(temp_ticks)
    cbar.set_ticklabels([f'{temp:.0f}°C' for temp in temp_ticks])
    cbar.ax.tick_params(labelsize=12)
    
    # Professional title matching reference style
    title_text = f'Relationship between Air Temperatures and Ground Cover\\n' \
                f'Characteristics in {city_name}, Uzbekistan'
    
    subtitle_text = f'A: Ambient air temperature in {city_name}, Uzbekistan on ' \
                   f'Summer {ANALYSIS_YEAR}'
    
    # Main title
    fig.suptitle(title_text, fontsize=16, fontweight='bold', y=0.95)
    
    # Subtitle
    ax.set_title(subtitle_text, fontsize=12, pad=20, loc='left')
    
    # Add coordinate labels
    ax.set_xlabel('Longitude (°E)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Latitude (°N)', fontsize=12, fontweight='bold')
    
    # Add gridlines (subtle like reference)
    gl = ax.gridlines(
        draw_labels=True, alpha=0.3, 
        linestyle='--', linewidth=0.5,
        color='gray'
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}
    
    # Add analysis boundary (dashed line like reference)
    buffer_deg = city_info['buffer_km'] / 111.0
    boundary_coords = [
        [city_info['lon'] - buffer_deg, city_info['lat'] - buffer_deg],
        [city_info['lon'] + buffer_deg, city_info['lat'] - buffer_deg],
        [city_info['lon'] + buffer_deg, city_info['lat'] + buffer_deg],
        [city_info['lon'] - buffer_deg, city_info['lat'] + buffer_deg],
        [city_info['lon'] - buffer_deg, city_info['lat'] - buffer_deg]
    ]
    
    boundary_lons = [coord[0] for coord in boundary_coords]
    boundary_lats = [coord[1] for coord in boundary_coords]
    
    ax.plot(
        boundary_lons, boundary_lats,
        color='navy', linestyle='--', 
        linewidth=2, alpha=0.8,
        transform=projection,
        label='Analysis Area'
    )
    
    # Add technical information box
    info_text = (
        f'Data Source: Landsat 8/9 Thermal Bands\\n'
        f'Spatial Resolution: {TARGET_RESOLUTION}m\\n'
        f'Temporal Coverage: Summer {ANALYSIS_YEAR}\\n'
        f'Valid Data Points: {temp_data["valid_points"]:,}/{temp_data["total_points"]:,}\\n'
        f'Temperature Range: {temp_min:.1f}°C - {temp_max:.1f}°C'
    )
    
    # Position info box in lower left
    ax.text(
        0.02, 0.02, info_text,
        transform=ax.transAxes,
        fontsize=9, verticalalignment='bottom',
        bbox=dict(
            boxstyle='round,pad=0.6', 
            facecolor='white', 
            alpha=0.9,
            edgecolor='gray'
        )
    )
    
    # Add scale bar (approximate)
    scale_km = 5
    scale_deg = scale_km / 111.0
    scale_start_lon = bounds[0] + (bounds[2] - bounds[0]) * 0.02
    scale_start_lat = bounds[1] + (bounds[3] - bounds[1]) * 0.08
    
    ax.plot(
        [scale_start_lon, scale_start_lon + scale_deg],
        [scale_start_lat, scale_start_lat],
        color='black', linewidth=4, 
        transform=projection, zorder=12
    )
    
    ax.text(
        scale_start_lon + scale_deg/2, 
        scale_start_lat + (bounds[3] - bounds[1]) * 0.02,
        f'{scale_km} km',
        ha='center', fontsize=10, fontweight='bold',
        transform=projection
    )
    
    # Tight layout
    plt.tight_layout()
    
    return fig

def main():
    """Main execution function"""
    print("🌡️ PROFESSIONAL 3D TEMPERATURE MAPPING")
    print("=" * 65)
    print(f"📅 Analysis Year: {ANALYSIS_YEAR}")
    print(f"🎯 Target Resolution: {TARGET_RESOLUTION}m")
    print(f"🏙️ Cities: {', '.join(CITIES.keys())}")
    print(f"🌡️ Season: Summer (months {WARM_SEASON_MONTHS})")
    print(f"🎨 Style: Professional 3D with street basemap")
    print("=" * 65)
    
    # Initialize Google Earth Engine
    if not authenticate_gee():
        return
    
    # Create output directory
    output_dir = Path('professional_3d_maps')
    output_dir.mkdir(exist_ok=True)
    
    # Analysis date range
    start_date = f"{ANALYSIS_YEAR}-{WARM_SEASON_MONTHS[0]:02d}-01"
    end_date = f"{ANALYSIS_YEAR}-{WARM_SEASON_MONTHS[-1]:02d}-30"
    
    # Process each city
    for city_name, city_info in CITIES.items():
        print(f"\\n🌍 Processing {city_name}...")
        print(f"   📍 {city_info['description']}")
        print(f"   📍 Location: {city_info['lat']:.4f}°N, {city_info['lon']:.4f}°E")
        
        try:
            # Create analysis region
            center = ee.Geometry.Point([city_info['lon'], city_info['lat']])
            region = center.buffer(city_info['buffer_km'] * 1000)
            
            # Get satellite data
            composite = get_landsat_thermal_data(region, start_date, end_date)
            
            if composite is None:
                print(f"   ❌ No satellite data for {city_name}")
                continue
            
            # Sample temperature data
            temp_data = sample_temperature_data_efficiently(
                composite, region, city_name, TARGET_RESOLUTION
            )
            
            if temp_data is None:
                print(f"   ❌ Failed to sample temperature data for {city_name}")
                continue
            
            # Create professional 3D map
            fig = create_3d_professional_map(temp_data, city_name, city_info)
            
            # Save high-quality map
            output_path = output_dir / f'{city_name.lower()}_professional_3d_temperature_map.png'
            fig.savefig(
                output_path, 
                dpi=300, 
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none'
            )
            
            print(f"   ✅ Professional 3D map saved: {output_path}")
            plt.close(fig)
            
            # Small delay between cities
            time.sleep(2)
            
        except Exception as e:
            print(f"   ❌ Error processing {city_name}: {e}")
            continue
    
    print(f"\\n🎉 Professional 3D temperature mapping complete!")
    print(f"📁 Maps saved in: {output_dir.absolute()}")

if __name__ == "__main__":
    main()
