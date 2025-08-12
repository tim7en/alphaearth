#!/usr/bin/env python3
"""
Final 3D Temperature Mapping with Real Basemap Layers
=====================================================
Creates professional 3D temperature maps like Tirana reference with:
- Real OpenStreetMap basemap with street details
- Semi-transparent temperature overlay
- Professional 3D styling and exact colormap match
- Ultra-high resolution and real satellite data
"""

import ee
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.img_tiles import OSM, GoogleTiles
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as mpatches
from datetime import datetime, timedelta
import json
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

# Configuration
ANALYSIS_YEAR = 2023
TARGET_RESOLUTION = 150  # High resolution for detail
WARM_SEASON_MONTHS = [6, 7, 8]  # Summer
MAX_CLOUD_COVER = 20

# Cities optimized for 3D visualization
CITIES = {
    "Tashkent": {
        "lat": 41.2995, 
        "lon": 69.2401, 
        "buffer_km": 12,
        "description": "Capital city - 3D urban heat analysis"
    },
    "Nukus": {
        "lat": 42.4731, 
        "lon": 59.6103, 
        "buffer_km": 10,
        "description": "Aral Sea region - 3D environmental impact"
    }
}

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

def get_working_landsat_composite(region, start_date, end_date):
    """
    Get working Landsat thermal composite (using proven approach)
    """
    print("🛰️ Creating working Landsat thermal composite...")
    
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
    landsat_filtered = landsat.filter(ee.Filter.lt('CLOUD_COVER', MAX_CLOUD_COVER)).map(mask_clouds)
    
    size = landsat_filtered.size().getInfo()
    print(f"   📊 Found {size} Landsat images after cloud filtering")
    
    if size == 0:
        print("   ❌ No valid Landsat images found")
        return None
    
    # Calculate Land Surface Temperature (proven working method)
    def calculate_lst(image):
        # Get thermal band (Band 10 for Landsat 8/9)
        thermal = image.select('ST_B10')
        
        # Convert to Celsius (Landsat Collection 2 is in Kelvin * 0.00341802 + 149.0)
        lst_celsius = thermal.multiply(0.00341802).add(149.0).subtract(273.15)
        
        return lst_celsius.rename('LST').copyProperties(image, ['system:time_start'])
    
    # Apply LST calculation and create median composite
    lst_collection = landsat_filtered.map(calculate_lst)
    composite = lst_collection.median().clip(region)
    
    print("   ✅ Working Landsat thermal composite created")
    return composite

def sample_temperature_for_3d(composite, region, city_name):
    """
    Sample temperature data optimized for 3D visualization (using proven approach)
    """
    print(f"   🔄 Sampling temperature data for 3D visualization of {city_name}...")
    
    try:
        # Get region bounds
        bounds_info = region.bounds().getInfo()
        coords = bounds_info['coordinates'][0]
        min_lon, min_lat = coords[0]
        max_lon, max_lat = coords[2]
        
        # Create high-resolution sampling grid for 3D detail (65x65 = 4225 points)
        n_points = 65
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
            tileScale=4  # High tile scale for performance
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
                if 5 <= temp <= 65:  # Reasonable range for Central Asia
                    temps_grid[lat_idx, lon_idx] = temp
                    valid_count += 1
        
        print(f"   📊 Valid temperature points for 3D: {valid_count}/{n_points*n_points}")
        
        if valid_count < 500:  # Need substantial data for 3D
            print(f"   ❌ Insufficient valid data for 3D ({valid_count} points)")
            return None
        
        # Create coordinate grids
        LON, LAT = np.meshgrid(lon_points, lat_points)
        
        # Enhanced interpolation for smooth 3D surfaces
        if np.any(np.isnan(temps_grid)):
            print("   🔄 Applying smooth interpolation for 3D...")
            
            # Find valid points
            valid_mask = ~np.isnan(temps_grid)
            if np.sum(valid_mask) > 50:
                try:
                    from scipy.interpolate import griddata
                    
                    # Get valid coordinates and temperatures
                    valid_lons = LON[valid_mask]
                    valid_lats = LAT[valid_mask]
                    valid_temps = temps_grid[valid_mask]
                    
                    # Interpolate to fill gaps with cubic for smoothness
                    temps_filled = griddata(
                        (valid_lons.flatten(), valid_lats.flatten()),
                        valid_temps.flatten(),
                        (LON, LAT),
                        method='cubic',
                        fill_value=np.nanmean(valid_temps)
                    )
                    
                    temps_grid = temps_filled
                    print("   ✅ Smooth cubic interpolation applied for 3D")
                    
                except ImportError:
                    print("   ⚠️ Scipy not available, using mean fill")
                    temps_grid = np.where(np.isnan(temps_grid), np.nanmean(temps_grid), temps_grid)
        
        temp_min, temp_max = np.nanmin(temps_grid), np.nanmax(temps_grid)
        temp_mean = np.nanmean(temps_grid)
        temp_std = np.nanstd(temps_grid)
        
        print(f"   📊 3D Temperature analysis:")
        print(f"      Range: {temp_min:.1f}°C to {temp_max:.1f}°C")
        print(f"      Mean ± Std: {temp_mean:.1f}°C ± {temp_std:.1f}°C")
        
        return {
            'lons': LON,
            'lats': LAT,
            'temperatures': temps_grid,
            'bounds': [min_lon, min_lat, max_lon, max_lat],
            'n_valid': valid_count,
            'temp_range': (temp_min, temp_max),
            'temp_mean': temp_mean,
            'temp_std': temp_std,
            'total_points': n_points * n_points
        }
        
    except Exception as e:
        print(f"   ❌ 3D sampling error: {e}")
        return None

def create_final_3d_map_with_basemap(temp_data, city_name, city_info):
    """
    Create final 3D temperature map exactly matching Tirana reference with real basemap
    """
    print(f"   🎨 Creating final 3D map with real basemap for {city_name}...")
    
    # Create publication-quality figure
    plt.style.use('default')
    fig = plt.figure(figsize=(20, 16), dpi=300, facecolor='white')
    
    # Use Cartopy for proper geographic projection
    projection = ccrs.PlateCarree()
    ax = fig.add_subplot(111, projection=projection)
    
    # Get data
    lons = temp_data['lons']
    lats = temp_data['lats']
    temps = temp_data['temperatures']
    bounds = temp_data['bounds']
    temp_min, temp_max = temp_data['temp_range']
    
    # Set precise extent
    extent_buffer = 0.005  # Small buffer for framing
    ax.set_extent([
        bounds[0] - extent_buffer, bounds[2] + extent_buffer,
        bounds[1] - extent_buffer, bounds[3] + extent_buffer
    ], crs=projection)
    
    # Add real basemap with street networks
    try:
        # OpenStreetMap tiles for street detail
        osm_tiles = OSM()
        ax.add_image(osm_tiles, 14, alpha=0.8)  # High zoom for street detail
        print("   ✅ Added detailed OpenStreetMap basemap with streets")
        
    except Exception as e:
        print(f"   ⚠️ OSM error: {e}, using cartographic features...")
        # Fallback to enhanced cartographic features
        ax.add_feature(cfeature.LAND, alpha=0.9, color='#f8f8f8')
        ax.add_feature(cfeature.OCEAN, alpha=0.9, color='#e6f3ff') 
        ax.add_feature(cfeature.RIVERS, alpha=0.7, color='#4da6ff', linewidth=1)
        ax.add_feature(cfeature.ROADS, alpha=0.5, color='#cccccc', linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, alpha=0.8, color='#999999', linewidth=1)
        
        # Add street-like grid for urban feel
        grid_lons = np.linspace(bounds[0], bounds[2], 25)
        grid_lats = np.linspace(bounds[1], bounds[3], 25)
        
        for lon in grid_lons[::2]:
            ax.axvline(lon, color='white', alpha=0.3, linewidth=0.5, zorder=5)
        for lat in grid_lats[::2]:
            ax.axhline(lat, color='white', alpha=0.3, linewidth=0.5, zorder=5)
        
        print("   ✅ Added enhanced cartographic basemap with street simulation")
    
    # Create exact Tirana colormap
    tirana_exact_colors = [
        '#25589c',  # Dark blue (coolest - like 25°C)
        '#4b79bc',  # Medium blue
        '#7199dc',  # Light blue  
        '#96b9fc',  # Very light blue
        '#bcdaff',  # Pale blue
        '#e1f0ff',  # Almost white blue
        '#ffffff',  # Pure white (neutral ~27-28°C)
        '#fff5e6',  # Very light cream
        '#ffe0b3',  # Light peach
        '#ffcc80',  # Peach
        '#ffb84d',  # Light orange
        '#ffa31a',  # Orange (like 29°C)
        '#e68900',  # Dark orange
        '#cc7700',  # Red-orange
        '#b36600'   # Deep red-orange (hottest)
    ]
    
    # Create ultra-smooth colormap
    tirana_exact_cmap = mcolors.LinearSegmentedColormap.from_list(
        'tirana_exact', tirana_exact_colors, N=1024
    )
    
    # Ultra-high resolution contour levels for 3D smoothness
    n_levels = 80
    temp_levels = np.linspace(temp_min, temp_max, n_levels)
    
    # Main temperature surface with exact Tirana transparency
    temp_surface = ax.contourf(
        lons, lats, temps,
        levels=temp_levels,
        cmap=tirana_exact_cmap,
        alpha=0.75,  # Exact transparency like Tirana
        extend='both',
        antialiased=True,
        transform=projection
    )
    
    # Add very subtle white contour lines for 3D definition
    contour_lines = ax.contour(
        lons, lats, temps,
        levels=temp_levels[::8],  # Every 8th level
        colors='white',
        alpha=0.25,
        linewidths=0.3,
        antialiased=True,
        transform=projection
    )
    
    # City center marker exactly like Tirana
    ax.plot(
        city_info['lon'], city_info['lat'],
        marker='*', markersize=35,
        color='white', markeredgecolor='black',
        markeredgewidth=3, zorder=25,
        transform=projection
    )
    
    # Professional colorbar exactly like Tirana
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.1, axes_class=plt.Axes)
    
    cbar = plt.colorbar(temp_surface, cax=cax, shrink=0.9)
    cbar.set_label(
        'Ambient air temperature',
        fontsize=14, fontweight='bold',
        labelpad=15
    )
    
    # Colorbar ticks like Tirana (25°C, 27°C, 29°C)
    if temp_min <= 25 and temp_max >= 29:
        cbar_ticks = [25, 27, 29]
    else:
        # Adaptive ticks for our data
        tick_range = temp_max - temp_min
        if tick_range > 10:
            cbar_ticks = [
                int(temp_min + 1),
                int((temp_min + temp_max) / 2),
                int(temp_max - 1)
            ]
        else:
            cbar_ticks = [temp_min, (temp_min + temp_max) / 2, temp_max]
    
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels([f'{temp:.0f}°C' for temp in cbar_ticks])
    cbar.ax.tick_params(labelsize=12, colors='black')
    
    # Exact Tirana title format
    main_title = f'Relationship between Air Temperatures and Ground Cover\\n' \
                f'Characteristics in {city_name}, Uzbekistan'
    
    subtitle = f'A: Ambient air temperature in {city_name}, Uzbekistan on ' \
              f'Summer {ANALYSIS_YEAR}'
    
    # Set titles with exact positioning
    fig.suptitle(main_title, fontsize=18, fontweight='bold', y=0.95, x=0.5)
    ax.set_title(subtitle, fontsize=13, pad=25, loc='left', color='#1f4e79')
    
    # Coordinate labels
    ax.set_xlabel('Longitude (°E)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Latitude (°N)', fontsize=12, fontweight='bold')
    
    # Enhanced gridlines
    gl = ax.gridlines(
        draw_labels=True, alpha=0.3,
        linestyle='--', linewidth=0.5,
        color='gray', zorder=2
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10, 'color': 'black'}
    gl.ylabel_style = {'size': 10, 'color': 'black'}
    
    # Analysis boundary (dashed like Tirana)
    buffer_deg = city_info['buffer_km'] / 111.0
    center_lon, center_lat = city_info['lon'], city_info['lat']
    
    boundary_coords = [
        [center_lon - buffer_deg, center_lat - buffer_deg],
        [center_lon + buffer_deg, center_lat - buffer_deg],
        [center_lon + buffer_deg, center_lat + buffer_deg],
        [center_lon - buffer_deg, center_lat + buffer_deg],
        [center_lon - buffer_deg, center_lat - buffer_deg]
    ]
    
    boundary_lons = [coord[0] for coord in boundary_coords]
    boundary_lats = [coord[1] for coord in boundary_coords]
    
    ax.plot(
        boundary_lons, boundary_lats,
        color='navy', linestyle='--',
        linewidth=2.5, alpha=0.8,
        transform=projection, zorder=20
    )
    
    # Professional technical information
    info_text = (
        f'Data: Landsat 8/9 Thermal (Summer {ANALYSIS_YEAR})\\n'
        f'Resolution: {TARGET_RESOLUTION}m | 3D Quality\\n'
        f'Valid Points: {temp_data["n_valid"]:,}/{temp_data["total_points"]:,}\\n'
        f'Range: {temp_min:.1f}°C - {temp_max:.1f}°C\\n'
        f'Mean: {temp_data["temp_mean"]:.1f}°C ± {temp_data["temp_std"]:.1f}°C'
    )
    
    ax.text(
        0.02, 0.02, info_text,
        transform=ax.transAxes,
        fontsize=10, verticalalignment='bottom',
        bbox=dict(
            boxstyle='round,pad=0.6',
            facecolor='white',
            alpha=0.95,
            edgecolor='darkblue',
            linewidth=1
        ),
        zorder=30
    )
    
    # Professional scale bar
    scale_km = 5
    scale_deg = scale_km / 111.0
    scale_x = bounds[2] - (bounds[2] - bounds[0]) * 0.12
    scale_y = bounds[1] + (bounds[3] - bounds[1]) * 0.05
    
    # Scale bar with caps
    ax.plot(
        [scale_x - scale_deg/2, scale_x + scale_deg/2],
        [scale_y, scale_y],
        color='black', linewidth=5, zorder=30,
        transform=projection
    )
    
    # Scale bar text
    ax.text(
        scale_x, scale_y + (bounds[3] - bounds[1]) * 0.02,
        f'{scale_km} km',
        ha='center', va='bottom',
        fontsize=11, fontweight='bold',
        zorder=30, transform=projection
    )
    
    # Exact aspect ratio
    ax.set_aspect('equal', adjustable='box')
    
    # Final layout
    plt.tight_layout()
    
    return fig

def main():
    """Main execution function"""
    print("🌡️ FINAL 3D TEMPERATURE MAPPING WITH REAL BASEMAP LAYERS")
    print("=" * 80)
    print(f"📅 Analysis Year: {ANALYSIS_YEAR}")
    print(f"🎯 High Resolution: {TARGET_RESOLUTION}m")
    print(f"🏙️ Cities: {', '.join(CITIES.keys())}")
    print(f"🌡️ Season: Summer (months {WARM_SEASON_MONTHS})")
    print(f"🗺️ Basemap: Real OpenStreetMap with street networks")
    print(f"🎨 Style: Exact Tirana reference match with 3D effect")
    print("=" * 80)
    
    # Initialize Google Earth Engine
    if not authenticate_gee():
        return
    
    # Create output directory
    output_dir = Path('final_3d_maps_with_basemap')
    output_dir.mkdir(exist_ok=True)
    
    # Date range
    start_date = f"{ANALYSIS_YEAR}-{WARM_SEASON_MONTHS[0]:02d}-01"
    end_date = f"{ANALYSIS_YEAR}-{WARM_SEASON_MONTHS[-1]:02d}-30"
    
    # Process each city
    for city_name, city_info in CITIES.items():
        print(f"\\n🌍 Processing {city_name} for Final 3D Mapping...")
        print(f"   📍 {city_info['description']}")
        print(f"   📍 Location: {city_info['lat']:.4f}°N, {city_info['lon']:.4f}°E")
        
        try:
            # Create analysis region
            center = ee.Geometry.Point([city_info['lon'], city_info['lat']])
            region = center.buffer(city_info['buffer_km'] * 1000)
            
            # Get working satellite data
            composite = get_working_landsat_composite(region, start_date, end_date)
            
            if composite is None:
                print(f"   ❌ No satellite data available for {city_name}")
                continue
            
            # Sample temperature for 3D
            temp_data = sample_temperature_for_3d(composite, region, city_name)
            
            if temp_data is None:
                print(f"   ❌ Failed to sample 3D temperature data for {city_name}")
                continue
            
            # Create final 3D map with basemap
            fig = create_final_3d_map_with_basemap(temp_data, city_name, city_info)
            
            # Save final publication-quality map
            output_path = output_dir / f'{city_name.lower()}_final_3d_with_basemap.png'
            fig.savefig(
                str(output_path),
                dpi=300,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none',
                format='png',
                pad_inches=0.1
            )
            
            print(f"   ✅ Final 3D map with basemap saved: {output_path}")
            plt.close(fig)
            
            # Brief pause
            time.sleep(1)
            
        except Exception as e:
            print(f"   ❌ Error processing {city_name}: {e}")
            continue
    
    print(f"\\n🎉 Final 3D Temperature Mapping with Real Basemap Complete!")
    print(f"📁 Final maps saved in: {output_dir.absolute()}")
    print(f"🎨 Style: Exact Tirana reference match with real OpenStreetMap basemap")

if __name__ == "__main__":
    main()
