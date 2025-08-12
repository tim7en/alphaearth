#!/usr/bin/env python3
"""
Professional Temperature Mapping with Basemap Style
===================================================
Creates professional temperature maps similar to the Tirana reference with:
- Semi-transparent temperature overlay
- Clean basemap styling  
- Professional 3D appearance
- Street-level detail simulation
"""

import ee
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
from datetime import datetime, timedelta
import json
from pathlib import Path
import time

# Configuration
ANALYSIS_YEAR = 2023
TARGET_RESOLUTION = 200  # meters
WARM_SEASON_MONTHS = [6, 7, 8]  # Summer
MAX_CLOUD_COVER = 20

# Cities configuration
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

def get_enhanced_landsat_data(region, start_date, end_date):
    """
    Get enhanced Landsat thermal data with urban features
    """
    print("🛰️ Processing enhanced Landsat data...")
    
    # Landsat 8 and 9 collections
    l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
    l9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
    
    # Combine and filter
    landsat = l8.merge(l9) \
                .filterBounds(region) \
                .filterDate(start_date, end_date) \
                .filter(ee.Filter.lt('CLOUD_COVER', MAX_CLOUD_COVER))
    
    def process_enhanced_image(image):
        """Enhanced processing for urban analysis"""
        # Cloud masking
        qa = image.select('QA_PIXEL')
        cloud_mask = qa.bitwiseAnd(1 << 3).eq(0).And(qa.bitwiseAnd(1 << 4).eq(0))
        masked = image.updateMask(cloud_mask)
        
        # Land Surface Temperature
        thermal = masked.select('ST_B10')
        lst_celsius = thermal.multiply(0.00341802).add(149.0).subtract(273.15)
        
        # Urban indices
        ndvi = masked.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
        ndbi = masked.normalizedDifference(['SR_B6', 'SR_B5']).rename('NDBI')
        
        # Built-up index (enhanced for urban areas)
        ui = masked.normalizedDifference(['SR_B7', 'SR_B5']).rename('UI')
        
        # Bare soil index
        bsi = masked.expression(
            '((RED + SWIR1) - (NIR + BLUE)) / ((RED + SWIR1) + (NIR + BLUE))',
            {
                'RED': masked.select('SR_B4'),
                'NIR': masked.select('SR_B5'), 
                'SWIR1': masked.select('SR_B6'),
                'BLUE': masked.select('SR_B2')
            }
        ).rename('BSI')
        
        return ee.Image.cat([lst_celsius.rename('LST'), ndvi, ndbi, ui, bsi]) \
                      .copyProperties(image, ['system:time_start'])
    
    # Process collection
    processed = landsat.map(process_enhanced_image)
    
    count = processed.size().getInfo()
    print(f"   📊 Found {count} enhanced Landsat scenes")
    
    if count == 0:
        return None
    
    # Create composite
    composite = processed.median().clip(region)
    print("   ✅ Enhanced Landsat composite created")
    
    return composite

def sample_enhanced_temperature_data(composite, region, city_name):
    """
    Enhanced temperature sampling with urban context
    """
    print(f"   🔄 Enhanced sampling for {city_name}...")
    
    try:
        # Get bounds
        bounds = region.bounds().getInfo()
        coords = bounds['coordinates'][0]
        
        lons = [coord[0] for coord in coords]
        lats = [coord[1] for coord in coords]
        min_lon, max_lon = min(lons), max(lons)
        min_lat, max_lat = min(lats), max(lats)
        
        # High-resolution sampling grid (70x70 = 4900 points)
        n_points = 70
        lon_array = np.linspace(min_lon, max_lon, n_points)
        lat_array = np.linspace(min_lat, max_lat, n_points)
        
        # Create sampling points
        points = []
        for i, lat in enumerate(lat_array):
            for j, lon in enumerate(lon_array):
                point = ee.Feature(
                    ee.Geometry.Point([lon, lat]),
                    {'lat_idx': i, 'lon_idx': j}
                )
                points.append(point)
        
        points_collection = ee.FeatureCollection(points)
        
        # Sample all bands
        sampled = composite.sampleRegions(
            collection=points_collection,
            scale=TARGET_RESOLUTION,
            geometries=True,
            tileScale=4
        ).getInfo()
        
        if not sampled or 'features' not in sampled:
            return None
        
        # Initialize grids
        temp_grid = np.full((n_points, n_points), np.nan)
        ndvi_grid = np.full((n_points, n_points), np.nan)
        ndbi_grid = np.full((n_points, n_points), np.nan)
        ui_grid = np.full((n_points, n_points), np.nan)
        
        valid_count = 0
        
        # Process sampling results
        for feature in sampled['features']:
            props = feature['properties']
            
            if all(key in props and props[key] is not None 
                   for key in ['LST', 'lat_idx', 'lon_idx']):
                
                temp = props['LST']
                lat_idx = props['lat_idx']
                lon_idx = props['lon_idx']
                
                # Filter realistic temperatures
                if 0 <= temp <= 65:
                    temp_grid[lat_idx, lon_idx] = temp
                    
                    # Store other indices if available
                    if 'NDVI' in props and props['NDVI'] is not None:
                        ndvi_grid[lat_idx, lon_idx] = props['NDVI']
                    if 'NDBI' in props and props['NDBI'] is not None:
                        ndbi_grid[lat_idx, lon_idx] = props['NDBI']
                    if 'UI' in props and props['UI'] is not None:
                        ui_grid[lat_idx, lon_idx] = props['UI']
                    
                    valid_count += 1
        
        print(f"   📊 Retrieved {valid_count}/{n_points*n_points} valid points")
        
        if valid_count < 300:
            print(f"   ❌ Insufficient data: {valid_count} points")
            return None
        
        # Create coordinate grids
        LON_GRID, LAT_GRID = np.meshgrid(lon_array, lat_array)
        
        # Enhanced gap filling
        if np.any(np.isnan(temp_grid)):
            try:
                from scipy.interpolate import griddata
                
                # Interpolate temperature
                valid_mask = ~np.isnan(temp_grid)
                if np.sum(valid_mask) > 50:
                    valid_lons = LON_GRID[valid_mask]
                    valid_lats = LAT_GRID[valid_mask]
                    valid_temps = temp_grid[valid_mask]
                    
                    # Fill gaps
                    nan_mask = np.isnan(temp_grid)
                    if np.any(nan_mask):
                        nan_lons = LON_GRID[nan_mask]
                        nan_lats = LAT_GRID[nan_mask]
                        
                        interpolated = griddata(
                            (valid_lons, valid_lats), valid_temps,
                            (nan_lons, nan_lats), method='cubic',
                            fill_value=np.nanmean(valid_temps)
                        )
                        
                        temp_grid[nan_mask] = interpolated
                        print("   ✅ High-quality cubic interpolation applied")
                
            except (ImportError, Exception):
                # Fallback interpolation
                temp_grid = np.where(np.isnan(temp_grid), 
                                   np.nanmean(temp_grid), temp_grid)
                print("   ⚠️ Basic interpolation applied")
        
        # Calculate enhanced statistics
        temp_min, temp_max = np.nanmin(temp_grid), np.nanmax(temp_grid)
        temp_mean = np.nanmean(temp_grid)
        temp_std = np.nanstd(temp_grid)
        
        print(f"   📊 Temperature statistics:")
        print(f"      Range: {temp_min:.1f}°C to {temp_max:.1f}°C")
        print(f"      Mean ± Std: {temp_mean:.1f}°C ± {temp_std:.1f}°C")
        
        return {
            'lons': LON_GRID,
            'lats': LAT_GRID,
            'temperatures': temp_grid,
            'ndvi': ndvi_grid,
            'ndbi': ndbi_grid,
            'urban_index': ui_grid,
            'bounds': [min_lon, min_lat, max_lon, max_lat],
            'temp_range': (temp_min, temp_max),
            'temp_mean': temp_mean,
            'temp_std': temp_std,
            'valid_points': valid_count,
            'total_points': n_points * n_points
        }
        
    except Exception as e:
        print(f"   ❌ Enhanced sampling error: {e}")
        return None

def create_professional_basemap_style(temp_data, city_name, city_info):
    """
    Create professional temperature map with basemap style like Tirana reference
    """
    print(f"   🎨 Creating professional basemap-style map for {city_name}...")
    
    # Create high-quality figure
    plt.style.use('default')  # Clean style
    fig = plt.figure(figsize=(18, 14), dpi=250, facecolor='white')
    
    # Main map subplot
    ax = fig.add_subplot(111)
    
    # Get data
    lons = temp_data['lons']
    lats = temp_data['lats']
    temps = temp_data['temperatures']
    bounds = temp_data['bounds']
    temp_min, temp_max = temp_data['temp_range']
    
    # Create street-like basemap effect
    ax.set_facecolor('#f8f8f8')  # Light gray background like streets
    
    # Create custom colormap matching Tirana reference exactly
    reference_colors = [
        '#2E4A99',  # Deep blue (25°C)
        '#5B73B5',  # Medium blue (26°C)
        '#89A2D1',  # Light blue (27°C)
        '#B7D1ED',  # Very light blue (28°C)
        '#E5F0FA',  # Almost white blue (29°C)
        '#FFFFFF',  # Pure white (30°C - neutral)
        '#FDF0E5',  # Very light peach
        '#FBE1CB',  # Light peach
        '#F9D2B1',  # Medium peach
        '#F7C397',  # Orange-peach
        '#F5B47D',  # Light orange
        '#F3A563',  # Medium orange
        '#F19649',  # Orange
        '#EF872F',  # Dark orange
        '#ED7815',  # Red-orange
        '#CC5500',  # Dark red-orange
        '#B71C1C'   # Deep red (35°C+)
    ]
    
    # Create smooth colormap
    custom_cmap = mcolors.LinearSegmentedColormap.from_list(
        'tirana_style', reference_colors, N=512
    )
    
    # High-resolution contour levels
    n_levels = 60
    temp_levels = np.linspace(temp_min, temp_max, n_levels)
    
    # Main temperature surface (semi-transparent like reference)
    temp_surface = ax.contourf(
        lons, lats, temps,
        levels=temp_levels,
        cmap=custom_cmap,
        alpha=0.85,  # Semi-transparent overlay
        extend='both',
        antialiased=True
    )
    
    # Add subtle contour lines for definition (white lines like reference)
    contour_lines = ax.contour(
        lons, lats, temps,
        levels=temp_levels[::6],  # Every 6th level
        colors='white',
        alpha=0.4,
        linewidths=0.3,
        antialiased=True
    )
    
    # Simulate street network with grid lines
    # Add fine grid to simulate street patterns
    grid_lons = np.linspace(bounds[0], bounds[2], 50)
    grid_lats = np.linspace(bounds[1], bounds[3], 50)
    
    # Vertical "streets"
    for lon in grid_lons[::3]:  # Every 3rd line
        ax.axvline(lon, color='white', alpha=0.2, linewidth=0.3, zorder=5)
    
    # Horizontal "streets"  
    for lat in grid_lats[::3]:  # Every 3rd line
        ax.axhline(lat, color='white', alpha=0.2, linewidth=0.3, zorder=5)
    
    # Add major "roads" (thicker lines)
    for lon in grid_lons[::8]:  # Major vertical roads
        ax.axvline(lon, color='white', alpha=0.4, linewidth=0.8, zorder=6)
    
    for lat in grid_lats[::8]:  # Major horizontal roads
        ax.axhline(lat, color='white', alpha=0.4, linewidth=0.8, zorder=6)
    
    # City center marker (prominent white star like reference)
    ax.scatter(
        city_info['lon'], city_info['lat'],
        s=400, marker='*', 
        c='white', edgecolors='black',
        linewidths=3, zorder=20,
        label='City Center'
    )
    
    # Professional colorbar (right side, matching reference)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2.5%", pad=0.15)
    
    cbar = plt.colorbar(temp_surface, cax=cax)
    cbar.set_label(
        'Ambient air temperature', 
        fontsize=14, fontweight='bold',
        labelpad=20
    )
    
    # Format colorbar exactly like reference
    temp_ticks = [25, 27, 29]  # Like in Tirana image
    if temp_min < 25:
        temp_ticks = [temp_min, temp_min + (temp_max-temp_min)/2, temp_max]
    
    cbar.set_ticks(temp_ticks)
    cbar.set_ticklabels([f'{temp:.0f}°C' for temp in temp_ticks])
    cbar.ax.tick_params(labelsize=12, colors='black')
    
    # Professional title matching reference exactly
    main_title = f'Relationship between Air Temperatures and Ground Cover\\n' \
                f'Characteristics in {city_name}, Uzbekistan'
    
    subtitle = f'A: Ambient air temperature in {city_name}, Uzbekistan on ' \
              f'Summer {ANALYSIS_YEAR}'
    
    # Set titles with exact positioning like reference
    fig.suptitle(main_title, fontsize=18, fontweight='bold', y=0.95, x=0.5)
    ax.set_title(subtitle, fontsize=13, pad=25, loc='left', color='#1f4e79')
    
    # Axis labels
    ax.set_xlabel('Longitude (°E)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Latitude (°N)', fontsize=12, fontweight='bold')
    
    # Format axes ticks
    ax.tick_params(labelsize=11, colors='black')
    
    # Add analysis boundary (dashed navy line like reference)
    buffer_deg = city_info['buffer_km'] / 111.0
    center_lon, center_lat = city_info['lon'], city_info['lat']
    
    # Create boundary polygon
    boundary_coords = np.array([
        [center_lon - buffer_deg, center_lat - buffer_deg],
        [center_lon + buffer_deg, center_lat - buffer_deg], 
        [center_lon + buffer_deg, center_lat + buffer_deg],
        [center_lon - buffer_deg, center_lat + buffer_deg],
        [center_lon - buffer_deg, center_lat - buffer_deg]
    ])
    
    boundary_patch = patches.Polygon(
        boundary_coords, 
        fill=False, 
        edgecolor='navy',
        linestyle='--',
        linewidth=2.5,
        alpha=0.8,
        zorder=15
    )
    ax.add_patch(boundary_patch)
    
    # Add technical information (lower left, clean box)
    info_text = (
        f'Data: Landsat 8/9 Thermal (Summer {ANALYSIS_YEAR})\\n'
        f'Resolution: {TARGET_RESOLUTION}m | Points: {temp_data["valid_points"]:,}\\n'
        f'Range: {temp_min:.1f}°C - {temp_max:.1f}°C\\n'
        f'Mean: {temp_data["temp_mean"]:.1f}°C ± {temp_data["temp_std"]:.1f}°C'
    )
    
    ax.text(
        0.02, 0.02, info_text,
        transform=ax.transAxes,
        fontsize=10, verticalalignment='bottom',
        bbox=dict(
            boxstyle='round,pad=0.8',
            facecolor='white',
            alpha=0.95,
            edgecolor='gray',
            linewidth=1
        ),
        zorder=18
    )
    
    # Add scale bar (bottom right)
    scale_km = 5
    scale_deg = scale_km / 111.0
    scale_x = bounds[2] - (bounds[2] - bounds[0]) * 0.15
    scale_y = bounds[1] + (bounds[3] - bounds[1]) * 0.05
    
    # Scale bar line
    ax.plot(
        [scale_x - scale_deg/2, scale_x + scale_deg/2],
        [scale_y, scale_y],
        color='black', linewidth=5, zorder=19
    )
    
    # Scale bar text
    ax.text(
        scale_x, scale_y + (bounds[3] - bounds[1]) * 0.02,
        f'{scale_km} km',
        ha='center', va='bottom',
        fontsize=11, fontweight='bold',
        zorder=19
    )
    
    # Set exact bounds to match data
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    
    # Ensure aspect ratio maintains geographic accuracy
    ax.set_aspect('equal', adjustable='box')
    
    # Clean layout
    plt.tight_layout()
    
    return fig

def main():
    """Main execution function"""
    print("🌡️ PROFESSIONAL BASEMAP-STYLE TEMPERATURE MAPPING")
    print("=" * 70)
    print(f"📅 Analysis Year: {ANALYSIS_YEAR}")
    print(f"🎯 Target Resolution: {TARGET_RESOLUTION}m") 
    print(f"🏙️ Cities: {', '.join(CITIES.keys())}")
    print(f"🌡️ Season: Summer (months {WARM_SEASON_MONTHS})")
    print(f"🎨 Style: Professional basemap with street-like overlay")
    print("=" * 70)
    
    # Initialize Google Earth Engine
    if not authenticate_gee():
        return
    
    # Create output directory
    output_dir = Path('professional_basemap_maps')
    output_dir.mkdir(exist_ok=True)
    
    # Date range
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
            
            # Get enhanced satellite data
            composite = get_enhanced_landsat_data(region, start_date, end_date)
            
            if composite is None:
                print(f"   ❌ No satellite data for {city_name}")
                continue
            
            # Sample enhanced temperature data
            temp_data = sample_enhanced_temperature_data(composite, region, city_name)
            
            if temp_data is None:
                print(f"   ❌ Failed to sample data for {city_name}")
                continue
            
            # Create professional basemap-style map
            fig = create_professional_basemap_style(temp_data, city_name, city_info)
            
            # Save ultra-high quality map
            output_path = output_dir / f'{city_name.lower()}_professional_basemap_temperature.png'
            fig.savefig(
                output_path,
                dpi=300,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none',
                format='png'
            )
            
            print(f"   ✅ Professional basemap saved: {output_path}")
            plt.close(fig)
            
            # Brief pause between cities
            time.sleep(1)
            
        except Exception as e:
            print(f"   ❌ Error processing {city_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\\n🎉 Professional basemap-style mapping complete!")
    print(f"📁 Maps saved in: {output_dir.absolute()}")

if __name__ == "__main__":
    main()
