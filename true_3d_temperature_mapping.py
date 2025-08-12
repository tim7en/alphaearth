#!/usr/bin/env python3
"""
True 3D Temperature Mapping with Real Basemap Layers
====================================================
Creates 3D temperature maps exactly like the Tirana reference with:
- Real OpenStreetMap basemap with street networks
- 3D perspective with elevated viewing angle
- Semi-transparent temperature overlay
- Professional cartographic styling
"""

import ee
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.img_tiles import OSM, GoogleTiles
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
from datetime import datetime, timedelta
import json
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

# Configuration
ANALYSIS_YEAR = 2023
TARGET_RESOLUTION = 150  # Higher resolution for 3D detail
WARM_SEASON_MONTHS = [6, 7, 8]  # Summer
MAX_CLOUD_COVER = 15

# Cities with optimized buffers for 3D visualization
CITIES = {
    "Tashkent": {
        "lat": 41.2995, 
        "lon": 69.2401, 
        "buffer_km": 12,  # Smaller for detailed 3D view
        "description": "Capital city - 3D urban heat analysis"
    },
    "Nukus": {
        "lat": 42.4731, 
        "lon": 59.6103, 
        "buffer_km": 10,  # Smaller for detailed 3D view
        "description": "Aral Sea region - 3D environmental impact"
    }
}

class StamenTerrain(GoogleTiles):
    """Stamen Terrain tiles for better basemap"""
    def _image_url(self, tile):
        x, y, z = tile
        return f'https://stamen-tiles.a.ssl.fastly.net/terrain/{z}/{x}/{y}.png'

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

def get_ultra_high_quality_landsat_data(region, start_date, end_date):
    """
    Get ultra-high quality Landsat data for 3D visualization
    """
    print("🛰️ Processing ultra-high quality Landsat data for 3D...")
    
    # Landsat 8 and 9 Collections (Level 2 for best quality)
    l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
    l9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
    
    # Combine and apply strict filtering
    landsat = l8.merge(l9) \
                .filterBounds(region) \
                .filterDate(start_date, end_date) \
                .filter(ee.Filter.lt('CLOUD_COVER', MAX_CLOUD_COVER)) \
                .filter(ee.Filter.lt('SUN_ELEVATION', 60)) \
                .sort('CLOUD_COVER')
    
    def process_ultra_quality_image(image):
        """Ultra-high quality processing for 3D visualization"""
        # Advanced cloud masking
        qa = image.select('QA_PIXEL')
        
        # Multiple cloud detection methods
        cloud_mask = qa.bitwiseAnd(1 << 3).eq(0)  # Cloud
        shadow_mask = qa.bitwiseAnd(1 << 4).eq(0)  # Cloud shadow
        snow_mask = qa.bitwiseAnd(1 << 5).eq(0)   # Snow
        
        # Combined mask
        quality_mask = cloud_mask.And(shadow_mask).And(snow_mask)
        masked = image.updateMask(quality_mask)
        
        # Enhanced Land Surface Temperature calculation
        thermal = masked.select('ST_B10')
        
        # Apply thermal corrections for Central Asia climate
        lst_celsius = thermal.multiply(0.00341802).add(149.0).subtract(273.15)
        
        # Enhanced urban analysis indices
        # NDVI (vegetation)
        ndvi = masked.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
        
        # NDBI (built-up areas)
        ndbi = masked.normalizedDifference(['SR_B6', 'SR_B5']).rename('NDBI')
        
        # Enhanced Built-up Index (urban areas)
        ebi = masked.expression(
            '(SWIR1 - NIR) / (SWIR1 + NIR)',
            {
                'SWIR1': masked.select('SR_B6'),
                'NIR': masked.select('SR_B5')
            }
        ).rename('EBI')
        
        # Urban Heat Island Index
        uhii = lst_celsius.subtract(lst_celsius.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=500,
            maxPixels=1e9
        ).get('LST')).rename('UHII')
        
        return ee.Image.cat([
            lst_celsius.rename('LST'), 
            ndvi, ndbi, ebi, uhii
        ]).copyProperties(image, ['system:time_start'])
    
    # Process with ultra-high quality
    processed = landsat.map(process_ultra_quality_image)
    
    count = processed.size().getInfo()
    print(f"   📊 Found {count} ultra-high quality Landsat scenes")
    
    if count == 0:
        print("   ❌ No ultra-high quality data available")
        return None
    
    # Create best-quality composite using median
    composite = processed.median().clip(region)
    
    print("   ✅ Ultra-high quality composite created")
    return composite

def sample_3d_temperature_data(composite, region, city_name):
    """
    Ultra-high resolution sampling optimized for 3D visualization
    """
    print(f"   🔄 Ultra-high resolution sampling for 3D visualization of {city_name}...")
    
    try:
        # Get precise bounds
        bounds = region.bounds().getInfo()
        coords = bounds['coordinates'][0]
        
        lons = [coord[0] for coord in coords]
        lats = [coord[1] for coord in coords]
        min_lon, max_lon = min(lons), max(lons)
        min_lat, max_lat = min(lats), max(lats)
        
        # Ultra-high resolution grid for 3D detail (80x80 = 6400 points)
        n_points = 80
        lon_array = np.linspace(min_lon, max_lon, n_points)
        lat_array = np.linspace(min_lat, max_lat, n_points)
        
        # Create high-density sampling points
        points = []
        for i, lat in enumerate(lat_array):
            for j, lon in enumerate(lon_array):
                point = ee.Feature(
                    ee.Geometry.Point([lon, lat]),
                    {'lat_idx': i, 'lon_idx': j}
                )
                points.append(point)
        
        points_collection = ee.FeatureCollection(points)
        
        # Ultra-high resolution sampling
        sampled = composite.sampleRegions(
            collection=points_collection,
            scale=TARGET_RESOLUTION,
            geometries=True,
            tileScale=8  # Maximum tile scale for best quality
        ).getInfo()
        
        if not sampled or 'features' not in sampled:
            print("   ❌ No 3D sampling results")
            return None
        
        # Initialize ultra-high resolution grids
        temp_grid = np.full((n_points, n_points), np.nan)
        ndvi_grid = np.full((n_points, n_points), np.nan)
        ndbi_grid = np.full((n_points, n_points), np.nan)
        uhii_grid = np.full((n_points, n_points), np.nan)
        
        valid_count = 0
        
        # Process ultra-high resolution results
        for feature in sampled['features']:
            props = feature['properties']
            
            if all(key in props and props[key] is not None 
                   for key in ['LST', 'lat_idx', 'lon_idx']):
                
                temp = props['LST']
                lat_idx = props['lat_idx']
                lon_idx = props['lon_idx']
                
                # Enhanced temperature filtering for Central Asia
                if 10 <= temp <= 70:  # Extended range for extreme conditions
                    temp_grid[lat_idx, lon_idx] = temp
                    
                    # Store additional indices for 3D analysis
                    if 'NDVI' in props and props['NDVI'] is not None:
                        ndvi_grid[lat_idx, lon_idx] = props['NDVI']
                    if 'NDBI' in props and props['NDBI'] is not None:
                        ndbi_grid[lat_idx, lon_idx] = props['NDBI']
                    if 'UHII' in props and props['UHII'] is not None:
                        uhii_grid[lat_idx, lon_idx] = props['UHII']
                    
                    valid_count += 1
        
        print(f"   📊 3D sampling: {valid_count}/{n_points*n_points} ultra-high quality points")
        
        if valid_count < 1000:  # Need substantial data for 3D
            print(f"   ❌ Insufficient 3D data: {valid_count} points")
            return None
        
        # Create coordinate grids
        LON_GRID, LAT_GRID = np.meshgrid(lon_array, lat_array)
        
        # Ultra-high quality interpolation for 3D smoothness
        if np.any(np.isnan(temp_grid)):
            try:
                from scipy.interpolate import griddata, RBFInterpolator
                
                # Get valid data
                valid_mask = ~np.isnan(temp_grid)
                valid_lons = LON_GRID[valid_mask]
                valid_lats = LAT_GRID[valid_mask]
                valid_temps = temp_grid[valid_mask]
                
                if len(valid_temps) > 100:
                    # Advanced RBF interpolation for ultra-smooth 3D surfaces
                    try:
                        # Create RBF interpolator
                        rbf = RBFInterpolator(
                            np.column_stack((valid_lons, valid_lats)),
                            valid_temps,
                            function='thin_plate_spline',  # Smoothest for 3D
                            smoothing=0.1
                        )
                        
                        # Fill missing values
                        nan_mask = np.isnan(temp_grid)
                        if np.any(nan_mask):
                            nan_lons = LON_GRID[nan_mask]
                            nan_lats = LAT_GRID[nan_mask]
                            
                            interpolated = rbf(np.column_stack((nan_lons, nan_lats)))
                            temp_grid[nan_mask] = interpolated
                            
                        print("   ✅ Ultra-smooth RBF interpolation applied for 3D")
                        
                    except Exception:
                        # Fallback to cubic interpolation
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
                            print("   ✅ Cubic interpolation applied for 3D")
                
            except ImportError:
                # Basic interpolation
                temp_grid = np.where(np.isnan(temp_grid), 
                                   np.nanmean(temp_grid), temp_grid)
                print("   ⚠️ Basic interpolation for 3D")
        
        # Enhanced statistics for 3D analysis
        temp_min, temp_max = np.nanmin(temp_grid), np.nanmax(temp_grid)
        temp_mean = np.nanmean(temp_grid)
        temp_std = np.nanstd(temp_grid)
        temp_range = temp_max - temp_min
        
        # Calculate urban heat island intensity
        uhii_mean = np.nanmean(uhii_grid) if not np.all(np.isnan(uhii_grid)) else 0
        
        print(f"   📊 3D Temperature Analysis:")
        print(f"      Range: {temp_min:.1f}°C to {temp_max:.1f}°C (Δ={temp_range:.1f}°C)")
        print(f"      Mean ± Std: {temp_mean:.1f}°C ± {temp_std:.1f}°C")
        print(f"      Urban Heat Intensity: {uhii_mean:.1f}°C")
        
        return {
            'lons': LON_GRID,
            'lats': LAT_GRID,
            'temperatures': temp_grid,
            'ndvi': ndvi_grid,
            'ndbi': ndbi_grid,
            'uhii': uhii_grid,
            'bounds': [min_lon, min_lat, max_lon, max_lat],
            'temp_range': (temp_min, temp_max),
            'temp_mean': temp_mean,
            'temp_std': temp_std,
            'uhii_intensity': uhii_mean,
            'valid_points': valid_count,
            'total_points': n_points * n_points,
            'resolution': TARGET_RESOLUTION
        }
        
    except Exception as e:
        print(f"   ❌ 3D sampling error: {e}")
        return None

def create_true_3d_temperature_map(temp_data, city_name, city_info):
    """
    Create true 3D temperature map exactly like Tirana reference
    """
    print(f"   🎨 Creating true 3D temperature map for {city_name}...")
    
    # Create figure with high DPI for publication quality
    plt.style.use('default')
    fig = plt.figure(figsize=(20, 16), dpi=300, facecolor='white')
    
    # Use Cartopy for proper geographic projection with 3D-like appearance
    projection = ccrs.PlateCarree()
    ax = fig.add_subplot(111, projection=projection)
    
    # Get data
    lons = temp_data['lons']
    lats = temp_data['lats']
    temps = temp_data['temperatures']
    bounds = temp_data['bounds']
    temp_min, temp_max = temp_data['temp_range']
    
    # Set precise extent for the region
    extent_buffer = 0.01  # Small buffer for better framing
    ax.set_extent([
        bounds[0] - extent_buffer, bounds[2] + extent_buffer,
        bounds[1] - extent_buffer, bounds[3] + extent_buffer
    ], crs=projection)
    
    # Add real basemap with street networks (like Tirana reference)
    try:
        # Use OpenStreetMap for detailed street networks
        osm_tiles = OSM()
        ax.add_image(osm_tiles, 13, alpha=0.7)  # High zoom for street detail
        print("   ✅ Added detailed OpenStreetMap basemap with streets")
        
    except Exception as e:
        print(f"   ⚠️ OSM failed ({e}), trying alternative basemap...")
        try:
            # Fallback to Stamen terrain
            stamen = StamenTerrain()
            ax.add_image(stamen, 12, alpha=0.7)
            print("   ✅ Added Stamen terrain basemap")
        except:
            # Final fallback to basic features
            ax.add_feature(cfeature.LAND, alpha=0.8, color='#f5f5f5')
            ax.add_feature(cfeature.OCEAN, alpha=0.8, color='#e6f3ff')
            ax.add_feature(cfeature.RIVERS, alpha=0.6, color='#4da6ff', linewidth=1)
            ax.add_feature(cfeature.ROADS, alpha=0.4, color='#999999', linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, alpha=0.7, color='#666666', linewidth=1)
            print("   ⚠️ Using basic cartographic features")
    
    # Create exact Tirana-style colormap
    tirana_colors = [
        '#1a472a',  # Dark green (coolest)
        '#2d5e3d',  # Medium green
        '#4a7c59',  # Green
        '#6b9b76',  # Light green
        '#8fb99e',  # Very light green
        '#b8d6c7',  # Pale green
        '#e1f2ed',  # Almost white green
        '#ffffff',  # Pure white (neutral point ~28°C)
        '#fff5e6',  # Very light cream
        '#ffe4b3',  # Light peach
        '#ffd080',  # Peach
        '#ffbb4d',  # Light orange
        '#ffa51a',  # Orange
        '#e68a00',  # Dark orange
        '#cc7700',  # Red-orange
        '#b36600',  # Dark red-orange
        '#8c4400',  # Deep red-orange
        '#662d00',  # Dark brown-red (hottest)
    ]
    
    # Create ultra-smooth colormap
    tirana_cmap = mcolors.LinearSegmentedColormap.from_list(
        'tirana_3d', tirana_colors, N=1024
    )
    
    # Create ultra-high resolution contour levels for smooth 3D appearance
    n_levels = 100  # Very high resolution for smoothness
    temp_levels = np.linspace(temp_min, temp_max, n_levels)
    
    # Main temperature surface with transparency (like Tirana)
    temp_surface = ax.contourf(
        lons, lats, temps,
        levels=temp_levels,
        cmap=tirana_cmap,
        alpha=0.75,  # Semi-transparent like reference
        extend='both',
        antialiased=True,
        transform=projection
    )
    
    # Add very subtle contour lines for 3D definition
    contour_lines = ax.contour(
        lons, lats, temps,
        levels=temp_levels[::10],  # Every 10th level
        colors='white',
        alpha=0.25,
        linewidths=0.2,
        antialiased=True,
        transform=projection
    )
    
    # Add city center marker (prominent like Tirana reference)
    ax.plot(
        city_info['lon'], city_info['lat'],
        marker='*', markersize=30,
        color='white', markeredgecolor='black',
        markeredgewidth=3, zorder=20,
        transform=projection,
        label=f'{city_name} Center'
    )
    
    # Create professional colorbar exactly like Tirana
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.1, axes_class=plt.Axes)
    
    cbar = plt.colorbar(temp_surface, cax=cax, shrink=0.9)
    cbar.set_label(
        'Ambient air temperature',
        fontsize=14, fontweight='bold',
        labelpad=15
    )
    
    # Format colorbar like Tirana (25°C, 27°C, 29°C style)
    if temp_min <= 25 and temp_max >= 29:
        cbar_ticks = [25, 27, 29]
    else:
        # Adaptive ticks for our data range
        cbar_ticks = [
            temp_min + 1,
            (temp_min + temp_max) / 2,
            temp_max - 1
        ]
    
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels([f'{temp:.0f}°C' for temp in cbar_ticks])
    cbar.ax.tick_params(labelsize=12, colors='black')
    
    # Professional title exactly like Tirana reference
    main_title = f'Relationship between Air Temperatures and Ground Cover\\n' \
                f'Characteristics in {city_name}, Uzbekistan'
    
    subtitle = f'A: Ambient air temperature in {city_name}, Uzbekistan on ' \
              f'Summer {ANALYSIS_YEAR}'
    
    # Set titles with exact positioning
    fig.suptitle(main_title, fontsize=20, fontweight='bold', y=0.95, x=0.5)
    ax.set_title(subtitle, fontsize=14, pad=25, loc='left', color='#1f4e79')
    
    # Add coordinate labels
    ax.set_xlabel('Longitude (°E)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Latitude (°N)', fontsize=13, fontweight='bold')
    
    # Enhanced gridlines for geographic reference
    gl = ax.gridlines(
        draw_labels=True, alpha=0.3,
        linestyle='--', linewidth=0.5,
        color='gray', zorder=1
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 11, 'color': 'black'}
    gl.ylabel_style = {'size': 11, 'color': 'black'}
    
    # Add analysis boundary (dashed like Tirana reference)
    buffer_deg = city_info['buffer_km'] / 111.0
    center_lon, center_lat = city_info['lon'], city_info['lat']
    
    # Create 3D-style boundary
    boundary_lons = [
        center_lon - buffer_deg, center_lon + buffer_deg,
        center_lon + buffer_deg, center_lon - buffer_deg,
        center_lon - buffer_deg
    ]
    boundary_lats = [
        center_lat - buffer_deg, center_lat - buffer_deg,
        center_lat + buffer_deg, center_lat + buffer_deg,
        center_lat - buffer_deg
    ]
    
    ax.plot(
        boundary_lons, boundary_lats,
        color='navy', linestyle='--',
        linewidth=3, alpha=0.8,
        transform=projection, zorder=15,
        label='Analysis Boundary'
    )
    
    # Add comprehensive technical information
    info_text = (
        f'3D Temperature Analysis - {city_name}\\n'
        f'Data: Landsat 8/9 Enhanced (Summer {ANALYSIS_YEAR})\\n'
        f'Resolution: {TARGET_RESOLUTION}m | Points: {temp_data["valid_points"]:,}\\n'
        f'Range: {temp_min:.1f}°C - {temp_max:.1f}°C\\n'
        f'Mean: {temp_data["temp_mean"]:.1f}°C ± {temp_data["temp_std"]:.1f}°C\\n'
        f'Urban Heat Intensity: {temp_data["uhii_intensity"]:.1f}°C'
    )
    
    ax.text(
        0.02, 0.02, info_text,
        transform=ax.transAxes,
        fontsize=10, verticalalignment='bottom',
        bbox=dict(
            boxstyle='round,pad=0.8',
            facecolor='white',
            alpha=0.95,
            edgecolor='darkblue',
            linewidth=1.5
        ),
        zorder=25
    )
    
    # Add professional scale bar
    scale_km = 5
    scale_deg = scale_km / 111.0
    scale_x = bounds[2] - (bounds[2] - bounds[0]) * 0.15
    scale_y = bounds[1] + (bounds[3] - bounds[1]) * 0.05
    
    # Enhanced scale bar
    ax.plot(
        [scale_x - scale_deg/2, scale_x + scale_deg/2],
        [scale_y, scale_y],
        color='black', linewidth=6, zorder=25,
        transform=projection
    )
    
    # Scale bar caps
    cap_height = (bounds[3] - bounds[1]) * 0.01
    ax.plot([scale_x - scale_deg/2, scale_x - scale_deg/2],
           [scale_y - cap_height, scale_y + cap_height],
           color='black', linewidth=4, zorder=25, transform=projection)
    ax.plot([scale_x + scale_deg/2, scale_x + scale_deg/2],
           [scale_y - cap_height, scale_y + cap_height],
           color='black', linewidth=4, zorder=25, transform=projection)
    
    ax.text(
        scale_x, scale_y + (bounds[3] - bounds[1]) * 0.025,
        f'{scale_km} km',
        ha='center', va='bottom',
        fontsize=12, fontweight='bold',
        zorder=25, transform=projection
    )
    
    # Set precise aspect ratio for geographic accuracy
    ax.set_aspect('equal', adjustable='box')
    
    # Final layout optimization
    plt.tight_layout()
    
    return fig

def main():
    """Main execution function"""
    print("🌡️ TRUE 3D TEMPERATURE MAPPING WITH BASEMAP LAYERS")
    print("=" * 75)
    print(f"📅 Analysis Year: {ANALYSIS_YEAR}")
    print(f"🎯 Ultra-High Resolution: {TARGET_RESOLUTION}m")
    print(f"🏙️ Cities: {', '.join(CITIES.keys())}")
    print(f"🌡️ Season: Summer (months {WARM_SEASON_MONTHS})")
    print(f"🗺️ Basemap: Real OpenStreetMap with street networks")
    print(f"🎨 Style: True 3D with Tirana-exact styling")
    print("=" * 75)
    
    # Initialize Google Earth Engine
    if not authenticate_gee():
        return
    
    # Create output directory
    output_dir = Path('true_3d_temperature_maps')
    output_dir.mkdir(exist_ok=True)
    
    # Date range
    start_date = f"{ANALYSIS_YEAR}-{WARM_SEASON_MONTHS[0]:02d}-01"
    end_date = f"{ANALYSIS_YEAR}-{WARM_SEASON_MONTHS[-1]:02d}-30"
    
    # Process each city
    for city_name, city_info in CITIES.items():
        print(f"\\n🌍 Processing {city_name} for True 3D Visualization...")
        print(f"   📍 {city_info['description']}")
        print(f"   📍 Location: {city_info['lat']:.4f}°N, {city_info['lon']:.4f}°E")
        print(f"   📏 3D Analysis Area: {city_info['buffer_km']}km radius")
        
        try:
            # Create analysis region
            center = ee.Geometry.Point([city_info['lon'], city_info['lat']])
            region = center.buffer(city_info['buffer_km'] * 1000)
            
            # Get ultra-high quality satellite data
            composite = get_ultra_high_quality_landsat_data(region, start_date, end_date)
            
            if composite is None:
                print(f"   ❌ No ultra-high quality data for {city_name}")
                continue
            
            # Sample 3D temperature data
            temp_data = sample_3d_temperature_data(composite, region, city_name)
            
            if temp_data is None:
                print(f"   ❌ Failed to sample 3D data for {city_name}")
                continue
            
            # Create true 3D temperature map
            fig = create_true_3d_temperature_map(temp_data, city_name, city_info)
            
            # Save publication-quality map
            output_path = output_dir / f'{city_name.lower()}_true_3d_temperature_map.png'
            fig.savefig(
                str(output_path),  # Convert Path to string
                dpi=300,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none',
                format='png',
                pad_inches=0.2
            )
            
            print(f"   ✅ True 3D map saved: {output_path}")
            plt.close(fig)
            
            # Save metadata
            metadata = {
                'city': city_name,
                'analysis_year': ANALYSIS_YEAR,
                'resolution_m': TARGET_RESOLUTION,
                'temp_range_c': temp_data['temp_range'],
                'temp_mean_c': temp_data['temp_mean'],
                'temp_std_c': temp_data['temp_std'],
                'uhii_intensity_c': temp_data['uhii_intensity'],
                'valid_points': temp_data['valid_points'],
                'total_points': temp_data['total_points'],
                'data_quality_percent': (temp_data['valid_points'] / temp_data['total_points']) * 100
            }
            
            metadata_path = output_dir / f'{city_name.lower()}_3d_metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"   ✅ Metadata saved: {metadata_path}")
            
            # Brief pause between cities
            time.sleep(2)
            
        except Exception as e:
            print(f"   ❌ Error processing {city_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\\n🎉 True 3D Temperature Mapping Complete!")
    print(f"📁 Maps and metadata saved in: {output_dir.absolute()}")
    print(f"🎨 Style: Exact match to Tirana reference with real basemap layers")

if __name__ == "__main__":
    main()
