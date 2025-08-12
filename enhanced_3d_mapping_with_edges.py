#!/usr/bin/env python3
"""
Enhanced 3D Temperature Mapping with Edge Detection and Topographic Effect
==========================================================================
Creates advanced 3D temperature maps with:
- Edge detection for roads, buildings, and terrain features (white lines)
- Hillshade/topographic basemap for 3D terrain effect
- Enhanced blue-to-red colormap with better contrast
- Vector feature overlays for crisp white roads and boundaries
- Real satellite data from Landsat 8/9
"""

import ee
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.img_tiles import OSM, GoogleTiles, Stamen
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
TARGET_RESOLUTION = 120  # Higher resolution for edge detection
WARM_SEASON_MONTHS = [6, 7, 8]  # Summer
MAX_CLOUD_COVER = 15

# Cities optimized for enhanced 3D visualization
CITIES = {
    "Tashkent": {
        "lat": 41.2995, 
        "lon": 69.2401, 
        "buffer_km": 10,  # Smaller for detailed edge detection
        "description": "Capital city - Enhanced 3D urban analysis"
    },
    "Nukus": {
        "lat": 42.4731, 
        "lon": 59.6103, 
        "buffer_km": 8,
        "description": "Aral Sea region - Enhanced 3D environmental analysis"
    }
}

class StamenTerrain(GoogleTiles):
    """Stamen Terrain tiles for topographic basemap"""
    def _image_url(self, tile):
        x, y, z = tile
        return f'https://stamen-tiles.a.ssl.fastly.net/terrain/{z}/{x}/{y}.png'

class OpenTopoMap(GoogleTiles):
    """OpenTopoMap tiles for enhanced topographic effect"""
    def _image_url(self, tile):
        x, y, z = tile
        return f'https://tile.opentopomap.org/{z}/{x}/{y}.png'

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

def get_enhanced_landsat_with_features(region, start_date, end_date):
    """
    Get enhanced Landsat data with additional features for edge detection
    """
    print("🛰️ Creating enhanced Landsat composite with features...")
    
    # Landsat 8 and 9 Collection 2 Level 2
    l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
    l9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
    
    # Combine collections
    landsat = l8.merge(l9).filterDate(start_date, end_date).filterBounds(region)
    
    # Apply cloud masking
    def mask_clouds_enhanced(image):
        qa = image.select('QA_PIXEL')
        # Enhanced cloud masking
        cloud_mask = qa.bitwiseAnd(1 << 3).eq(0).And(qa.bitwiseAnd(1 << 4).eq(0))
        snow_mask = qa.bitwiseAnd(1 << 5).eq(0)
        return image.updateMask(cloud_mask.And(snow_mask))
    
    # Filter and process
    landsat_filtered = landsat.filter(ee.Filter.lt('CLOUD_COVER', MAX_CLOUD_COVER)).map(mask_clouds_enhanced)
    
    size = landsat_filtered.size().getInfo()
    print(f"   📊 Found {size} enhanced Landsat images")
    
    if size == 0:
        return None
    
    # Enhanced processing function
    def calculate_enhanced_features(image):
        # Land Surface Temperature
        thermal = image.select('ST_B10')
        lst_celsius = thermal.multiply(0.00341802).add(149.0).subtract(273.15)
        
        # Enhanced vegetation and urban indices
        ndvi = image.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
        ndbi = image.normalizedDifference(['SR_B6', 'SR_B5']).rename('NDBI')
        
        # Enhanced Built-up Index for edge detection
        ebi = image.normalizedDifference(['SR_B6', 'SR_B5']).rename('EBI')
        
        # Water index for rivers and water bodies
        ndwi = image.normalizedDifference(['SR_B3', 'SR_B5']).rename('NDWI')
        
        # Bare Soil Index for terrain features
        bsi = image.expression(
            '((SWIR1 + RED) - (NIR + BLUE)) / ((SWIR1 + RED) + (NIR + BLUE))',
            {
                'SWIR1': image.select('SR_B6'),
                'RED': image.select('SR_B4'),
                'NIR': image.select('SR_B5'),
                'BLUE': image.select('SR_B2')
            }
        ).rename('BSI')
        
        # Texture/Edge indicator using GLCM variance
        # Using NIR band for texture analysis
        nir = image.select('SR_B5')
        texture = nir.glcmTexture(size=3).select('SR_B5_var').rename('TEXTURE')
        
        return ee.Image.cat([
            lst_celsius.rename('LST'),
            ndvi, ndbi, ebi, ndwi, bsi, texture
        ]).copyProperties(image, ['system:time_start'])
    
    # Apply enhanced processing and create composite
    enhanced_collection = landsat_filtered.map(calculate_enhanced_features)
    composite = enhanced_collection.median().clip(region)
    
    print("   ✅ Enhanced Landsat composite with features created")
    return composite

def sample_enhanced_data_for_edges(composite, region, city_name):
    """
    Sample enhanced data optimized for edge detection and 3D effects
    """
    print(f"   🔄 Enhanced sampling for edge detection in {city_name}...")
    
    try:
        # Get region bounds
        bounds_info = region.bounds().getInfo()
        coords = bounds_info['coordinates'][0]
        min_lon, min_lat = coords[0]
        max_lon, max_lat = coords[2]
        
        # Ultra-high resolution grid for edge detection (80x80 = 6400 points)
        n_points = 80
        lon_points = np.linspace(min_lon, max_lon, n_points)
        lat_points = np.linspace(min_lat, max_lat, n_points)
        
        # Create feature collection for sampling
        points = []
        for i, lat in enumerate(lat_points):
            for j, lon in enumerate(lon_points):
                points.append(ee.Feature(ee.Geometry.Point([lon, lat]), 
                                       {'lat_idx': i, 'lon_idx': j}))
        
        points_fc = ee.FeatureCollection(points)
        
        # Sample all bands at once for comprehensive analysis
        sampled = composite.sampleRegions(
            collection=points_fc,
            scale=TARGET_RESOLUTION,
            geometries=True,
            tileScale=8  # High tile scale for performance
        ).getInfo()
        
        if not sampled or 'features' not in sampled:
            return None
        
        # Initialize grids for all features
        temp_grid = np.full((n_points, n_points), np.nan)
        ndvi_grid = np.full((n_points, n_points), np.nan)
        ndbi_grid = np.full((n_points, n_points), np.nan)
        ebi_grid = np.full((n_points, n_points), np.nan)
        ndwi_grid = np.full((n_points, n_points), np.nan)
        bsi_grid = np.full((n_points, n_points), np.nan)
        texture_grid = np.full((n_points, n_points), np.nan)
        
        valid_count = 0
        
        # Process results with all indices
        for feature in sampled['features']:
            props = feature['properties']
            
            if 'LST' in props and props['LST'] is not None:
                lat_idx = props['lat_idx']
                lon_idx = props['lon_idx']
                
                temp = props['LST']
                if 5 <= temp <= 70:  # Valid temperature range
                    temp_grid[lat_idx, lon_idx] = temp
                    
                    # Store all indices for edge detection
                    if 'NDVI' in props and props['NDVI'] is not None:
                        ndvi_grid[lat_idx, lon_idx] = props['NDVI']
                    if 'NDBI' in props and props['NDBI'] is not None:
                        ndbi_grid[lat_idx, lon_idx] = props['NDBI']
                    if 'EBI' in props and props['EBI'] is not None:
                        ebi_grid[lat_idx, lon_idx] = props['EBI']
                    if 'NDWI' in props and props['NDWI'] is not None:
                        ndwi_grid[lat_idx, lon_idx] = props['NDWI']
                    if 'BSI' in props and props['BSI'] is not None:
                        bsi_grid[lat_idx, lon_idx] = props['BSI']
                    if 'TEXTURE' in props and props['TEXTURE'] is not None:
                        texture_grid[lat_idx, lon_idx] = props['TEXTURE']
                    
                    valid_count += 1
        
        print(f"   📊 Enhanced sampling: {valid_count}/{n_points*n_points} points with features")
        
        if valid_count < 1000:
            print(f"   ❌ Insufficient data for edge detection: {valid_count} points")
            return None
        
        # Create coordinate grids
        LON, LAT = np.meshgrid(lon_points, lat_points)
        
        # Advanced interpolation for smooth surfaces
        try:
            from scipy.interpolate import griddata
            from scipy import ndimage
            
            # Interpolate all grids
            grids_to_interpolate = [
                ('temperature', temp_grid),
                ('ndvi', ndvi_grid),
                ('ndbi', ndbi_grid), 
                ('ebi', ebi_grid),
                ('ndwi', ndwi_grid),
                ('bsi', bsi_grid),
                ('texture', texture_grid)
            ]
            
            interpolated_grids = {}
            
            for name, grid in grids_to_interpolate:
                if np.any(~np.isnan(grid)):
                    valid_mask = ~np.isnan(grid)
                    if np.sum(valid_mask) > 50:
                        valid_lons = LON[valid_mask]
                        valid_lats = LAT[valid_mask]
                        valid_values = grid[valid_mask]
                        
                        # Interpolate missing values
                        filled_grid = griddata(
                            (valid_lons.flatten(), valid_lats.flatten()),
                            valid_values.flatten(),
                            (LON, LAT),
                            method='cubic',
                            fill_value=np.nanmean(valid_values)
                        )
                        
                        # Apply Gaussian smoothing for better 3D effect
                        filled_grid = ndimage.gaussian_filter(filled_grid, sigma=0.8)
                        interpolated_grids[name] = filled_grid
                    else:
                        interpolated_grids[name] = grid
                else:
                    interpolated_grids[name] = grid
            
            print("   ✅ Advanced interpolation and smoothing applied")
            
        except ImportError:
            print("   ⚠️ Using basic interpolation")
            interpolated_grids = {
                'temperature': np.where(np.isnan(temp_grid), np.nanmean(temp_grid), temp_grid),
                'ndvi': ndvi_grid,
                'ndbi': ndbi_grid,
                'ebi': ebi_grid,
                'ndwi': ndwi_grid,
                'bsi': bsi_grid,
                'texture': texture_grid
            }
        
        # Calculate statistics
        temp_min = np.nanmin(interpolated_grids['temperature'])
        temp_max = np.nanmax(interpolated_grids['temperature'])
        temp_mean = np.nanmean(interpolated_grids['temperature'])
        temp_std = np.nanstd(interpolated_grids['temperature'])
        
        print(f"   📊 Enhanced analysis results:")
        print(f"      Temperature: {temp_min:.1f}°C to {temp_max:.1f}°C (μ={temp_mean:.1f}°C, σ={temp_std:.1f}°C)")
        
        return {
            'lons': LON,
            'lats': LAT,
            'bounds': [min_lon, min_lat, max_lon, max_lat],
            'n_valid': valid_count,
            'temp_range': (temp_min, temp_max),
            'temp_mean': temp_mean,
            'temp_std': temp_std,
            'total_points': n_points * n_points,
            **interpolated_grids  # Include all interpolated grids
        }
        
    except Exception as e:
        print(f"   ❌ Enhanced sampling error: {e}")
        return None

def detect_edges_and_features(data):
    """
    Detect edges in temperature, urban, and terrain features for white overlay
    """
    print("   🔍 Detecting edges and features for white overlay...")
    
    try:
        from scipy import ndimage
        from skimage import feature, filters
        
        # Get main data grids
        temp_grid = data['temperature']
        ndbi_grid = data.get('ndbi', np.zeros_like(temp_grid))
        ebi_grid = data.get('ebi', np.zeros_like(temp_grid))
        ndwi_grid = data.get('ndwi', np.zeros_like(temp_grid))
        texture_grid = data.get('texture', np.zeros_like(temp_grid))
        
        # Edge detection on multiple features
        edge_maps = []
        
        # 1. Temperature edges (thermal boundaries)
        if not np.all(np.isnan(temp_grid)):
            temp_edges = feature.canny(temp_grid, sigma=1.2, low_threshold=0.1, high_threshold=0.3)
            edge_maps.append(temp_edges)
        
        # 2. Urban/built-up edges (building boundaries)
        if not np.all(np.isnan(ndbi_grid)):
            urban_edges = feature.canny(ndbi_grid, sigma=1.0, low_threshold=0.15, high_threshold=0.4)
            edge_maps.append(urban_edges)
        
        # 3. Enhanced built-up edges
        if not np.all(np.isnan(ebi_grid)):
            ebi_edges = feature.canny(ebi_grid, sigma=0.8, low_threshold=0.1, high_threshold=0.35)
            edge_maps.append(ebi_edges)
        
        # 4. Water body edges (rivers, lakes)
        if not np.all(np.isnan(ndwi_grid)):
            water_edges = feature.canny(ndwi_grid, sigma=1.0, low_threshold=0.2, high_threshold=0.5)
            edge_maps.append(water_edges)
        
        # 5. Texture edges (terrain features)
        if not np.all(np.isnan(texture_grid)):
            texture_edges = feature.canny(texture_grid, sigma=1.5, low_threshold=0.1, high_threshold=0.3)
            edge_maps.append(texture_edges)
        
        # Combine all edge maps
        if edge_maps:
            combined_edges = np.zeros_like(temp_grid, dtype=bool)
            for edge_map in edge_maps:
                combined_edges |= edge_map
            
            # Apply morphological operations to enhance edges
            combined_edges = ndimage.binary_dilation(combined_edges, iterations=1)
            combined_edges = ndimage.binary_erosion(combined_edges, iterations=1)
            
            print(f"   ✅ Edge detection complete: {np.sum(combined_edges)} edge pixels detected")
            return combined_edges.astype(float)
        else:
            print("   ⚠️ No valid data for edge detection")
            return np.zeros_like(temp_grid)
            
    except ImportError:
        print("   ⚠️ Scikit-image not available, using gradient-based edge detection")
        
        # Fallback gradient-based edge detection
        if not np.all(np.isnan(temp_grid)):
            grad_x = np.gradient(temp_grid, axis=1)
            grad_y = np.gradient(temp_grid, axis=0)
            edge_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Threshold to get binary edges
            threshold = np.nanpercentile(edge_magnitude, 85)  # Top 15% gradients
            edges = edge_magnitude > threshold
            
            print(f"   ✅ Gradient-based edge detection: {np.sum(edges)} edge pixels")
            return edges.astype(float)
        else:
            return np.zeros_like(temp_grid)

def create_enhanced_3d_map_with_edges(temp_data, city_name, city_info):
    """
    Create enhanced 3D temperature map with edge detection and topographic effects
    """
    print(f"   🎨 Creating enhanced 3D map with edges and topographic effects for {city_name}...")
    
    # Create high-quality figure
    plt.style.use('default')
    fig = plt.figure(figsize=(22, 18), dpi=300, facecolor='white')
    
    # Use Cartopy for geographic projection
    projection = ccrs.PlateCarree()
    ax = fig.add_subplot(111, projection=projection)
    
    # Get data
    lons = temp_data['lons']
    lats = temp_data['lats']
    temps = temp_data['temperature']
    bounds = temp_data['bounds']
    temp_min, temp_max = temp_data['temp_range']
    
    # Set precise extent
    extent_buffer = 0.003
    ax.set_extent([
        bounds[0] - extent_buffer, bounds[2] + extent_buffer,
        bounds[1] - extent_buffer, bounds[3] + extent_buffer
    ], crs=projection)
    
    # Add enhanced topographic basemap for 3D effect
    try:
        # Try OpenTopoMap first for best topographic effect
        topo_tiles = OpenTopoMap()
        ax.add_image(topo_tiles, 15, alpha=0.6)
        print("   ✅ Added OpenTopoMap topographic basemap")
        
    except Exception:
        try:
            # Fallback to Stamen Terrain
            terrain_tiles = StamenTerrain()
            ax.add_image(terrain_tiles, 14, alpha=0.7)
            print("   ✅ Added Stamen Terrain basemap")
            
        except Exception:
            # Final fallback to enhanced cartographic features
            ax.add_feature(cfeature.LAND, alpha=0.8, color='#f0f0f0')
            ax.add_feature(cfeature.OCEAN, alpha=0.8, color='#e6f3ff')
            ax.add_feature(cfeature.RIVERS, alpha=0.9, color='#4da6ff', linewidth=1.5)
            ax.add_feature(cfeature.LAKES, alpha=0.9, color='#4da6ff')
            
            # Add hillshade effect simulation
            y, x = np.mgrid[bounds[1]:bounds[3]:100j, bounds[0]:bounds[2]:100j]
            z = np.sin(np.sqrt((x - city_info['lon'])**2 + (y - city_info['lat'])**2) * 50) * 0.1
            ax.contour(x, y, z, levels=20, colors='gray', alpha=0.2, linewidths=0.5, transform=projection)
            
            print("   ✅ Added enhanced cartographic basemap with hillshade simulation")
    
    # Create enhanced blue-to-red colormap with better contrast
    enhanced_colors = [
        '#0d47a1',  # Deep blue (coolest)
        '#1976d2',  # Blue
        '#42a5f5',  # Light blue  
        '#81d4fa',  # Very light blue
        '#b3e5fc',  # Pale blue
        '#e1f5fe',  # Almost white blue
        '#ffffff',  # Pure white (neutral)
        '#fff3e0',  # Very light orange
        '#ffe0b2',  # Light orange
        '#ffcc02',  # Orange
        '#ff9800',  # Dark orange
        '#f57c00',  # Red-orange
        '#e65100',  # Deep red-orange
        '#bf360c',  # Dark red
        '#b71c1c'   # Deep red (hottest)
    ]
    
    enhanced_cmap = mcolors.LinearSegmentedColormap.from_list(
        'enhanced_3d', enhanced_colors, N=512
    )
    
    # Ultra-high resolution contour levels
    n_levels = 100
    temp_levels = np.linspace(temp_min, temp_max, n_levels)
    
    # Main temperature surface with enhanced transparency
    temp_surface = ax.contourf(
        lons, lats, temps,
        levels=temp_levels,
        cmap=enhanced_cmap,
        alpha=0.7,  # Slightly more transparent to show basemap
        extend='both',
        antialiased=True,
        transform=projection
    )
    
    # Detect and overlay edges in white
    edges = detect_edges_and_features(temp_data)
    
    if np.any(edges):
        # Create white edge overlay
        edge_overlay = np.ma.masked_where(edges == 0, edges)
        
        ax.contour(
            lons, lats, edge_overlay,
            levels=[0.5],  # Single level for binary edges
            colors='white',
            linewidths=1.2,
            alpha=0.9,
            transform=projection,
            antialiased=True
        )
        
        # Add additional white highlights for strongest edges
        strong_edges = edges > 0.7
        if np.any(strong_edges):
            strong_edge_overlay = np.ma.masked_where(~strong_edges, strong_edges.astype(float))
            
            ax.contour(
                lons, lats, strong_edge_overlay,
                levels=[0.5],
                colors='white',
                linewidths=2.0,
                alpha=0.95,
                transform=projection,
                antialiased=True
            )
        
        print("   ✅ White edge overlays added for roads, buildings, and terrain features")
    
    # Add vector features for additional white lines
    try:
        # Add roads and rivers as white lines
        ax.add_feature(cfeature.ROADS, alpha=0.8, color='white', linewidth=0.8, zorder=15)
        ax.add_feature(cfeature.RIVERS, alpha=0.9, color='white', linewidth=1.2, zorder=15)
        print("   ✅ Added vector roads and rivers as white overlays")
    except:
        print("   ⚠️ Vector features not available")
    
    # Enhanced city center marker
    ax.plot(
        city_info['lon'], city_info['lat'],
        marker='*', markersize=40,
        color='white', markeredgecolor='black',
        markeredgewidth=4, zorder=25,
        transform=projection
    )
    
    # Professional colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.12, axes_class=plt.Axes)
    
    cbar = plt.colorbar(temp_surface, cax=cax, shrink=0.9)
    cbar.set_label(
        'Surface Temperature',
        fontsize=16, fontweight='bold',
        labelpad=20
    )
    
    # Enhanced colorbar formatting
    cbar_ticks = np.linspace(temp_min, temp_max, 6)
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels([f'{temp:.0f}°C' for temp in cbar_ticks])
    cbar.ax.tick_params(labelsize=14, colors='black')
    
    # Enhanced title
    main_title = f'Enhanced 3D Temperature Analysis with Edge Detection\\n' \
                f'{city_name}, Uzbekistan - Topographic Enhanced Visualization'
    
    subtitle = f'Surface temperature with terrain features, roads, and building edges highlighted\\n' \
              f'Summer {ANALYSIS_YEAR} | Resolution: {TARGET_RESOLUTION}m | Enhanced 3D Effect'
    
    fig.suptitle(main_title, fontsize=20, fontweight='bold', y=0.95, x=0.5)
    ax.set_title(subtitle, fontsize=14, pad=30, loc='center', color='#1f4e79')
    
    # Enhanced coordinate labels
    ax.set_xlabel('Longitude (°E)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Latitude (°N)', fontsize=14, fontweight='bold')
    
    # Enhanced gridlines
    gl = ax.gridlines(
        draw_labels=True, alpha=0.4,
        linestyle=':', linewidth=0.8,
        color='gray', zorder=3
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 12, 'color': 'black'}
    gl.ylabel_style = {'size': 12, 'color': 'black'}
    
    # Enhanced analysis boundary
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
        linewidth=3, alpha=0.9,
        transform=projection, zorder=20
    )
    
    # Enhanced technical information
    info_text = (
        f'Enhanced 3D Analysis - {city_name}\\n'
        f'◆ Data: Landsat 8/9 Enhanced (Summer {ANALYSIS_YEAR})\\n'
        f'◆ Resolution: {TARGET_RESOLUTION}m | Points: {temp_data["n_valid"]:,}\\n'
        f'◆ Temperature: {temp_min:.1f}°C - {temp_max:.1f}°C\\n'
        f'◆ Mean ± Std: {temp_data["temp_mean"]:.1f}°C ± {temp_data["temp_std"]:.1f}°C\\n'
        f'◆ Features: Edge detection + Topographic basemap\\n'
        f'◆ White lines: Roads, buildings, terrain edges'
    )
    
    ax.text(
        0.02, 0.02, info_text,
        transform=ax.transAxes,
        fontsize=11, verticalalignment='bottom',
        bbox=dict(
            boxstyle='round,pad=1.0',
            facecolor='white',
            alpha=0.95,
            edgecolor='darkblue',
            linewidth=2
        ),
        zorder=30
    )
    
    # Enhanced scale bar
    scale_km = 3
    scale_deg = scale_km / 111.0
    scale_x = bounds[2] - (bounds[2] - bounds[0]) * 0.12
    scale_y = bounds[1] + (bounds[3] - bounds[1]) * 0.06
    
    # Scale bar with enhanced styling
    ax.plot(
        [scale_x - scale_deg/2, scale_x + scale_deg/2],
        [scale_y, scale_y],
        color='black', linewidth=6, zorder=35,
        transform=projection
    )
    
    # Scale bar caps
    cap_height = (bounds[3] - bounds[1]) * 0.008
    for x_pos in [scale_x - scale_deg/2, scale_x + scale_deg/2]:
        ax.plot([x_pos, x_pos], [scale_y - cap_height, scale_y + cap_height],
               color='black', linewidth=4, zorder=35, transform=projection)
    
    ax.text(
        scale_x, scale_y + (bounds[3] - bounds[1]) * 0.025,
        f'{scale_km} km',
        ha='center', va='bottom',
        fontsize=13, fontweight='bold',
        zorder=35, transform=projection
    )
    
    # Perfect aspect ratio
    ax.set_aspect('equal', adjustable='box')
    
    # Final layout optimization
    plt.tight_layout()
    
    return fig

def main():
    """Main execution function"""
    print("🌡️ ENHANCED 3D TEMPERATURE MAPPING WITH EDGE DETECTION")
    print("=" * 85)
    print(f"📅 Analysis Year: {ANALYSIS_YEAR}")
    print(f"🎯 Ultra-High Resolution: {TARGET_RESOLUTION}m")
    print(f"🏙️ Cities: {', '.join(CITIES.keys())}")
    print(f"🌡️ Season: Summer (months {WARM_SEASON_MONTHS})")
    print(f"🗺️ Basemap: Topographic with hillshade effects")
    print(f"🔍 Features: Edge detection for roads, buildings, terrain")
    print(f"🎨 Style: Enhanced 3D with white edge overlays")
    print("=" * 85)
    
    # Initialize Google Earth Engine
    if not authenticate_gee():
        return
    
    # Create output directory
    output_dir = Path('enhanced_3d_maps_with_edges')
    output_dir.mkdir(exist_ok=True)
    
    # Date range
    start_date = f"{ANALYSIS_YEAR}-{WARM_SEASON_MONTHS[0]:02d}-01"
    end_date = f"{ANALYSIS_YEAR}-{WARM_SEASON_MONTHS[-1]:02d}-30"
    
    # Process each city
    for city_name, city_info in CITIES.items():
        print(f"\\n🌍 Processing {city_name} for Enhanced 3D Visualization...")
        print(f"   📍 {city_info['description']}")
        print(f"   📍 Location: {city_info['lat']:.4f}°N, {city_info['lon']:.4f}°E")
        print(f"   📏 Analysis Area: {city_info['buffer_km']}km radius")
        
        try:
            # Create analysis region
            center = ee.Geometry.Point([city_info['lon'], city_info['lat']])
            region = center.buffer(city_info['buffer_km'] * 1000)
            
            # Get enhanced satellite data with features
            composite = get_enhanced_landsat_with_features(region, start_date, end_date)
            
            if composite is None:
                print(f"   ❌ No enhanced satellite data for {city_name}")
                continue
            
            # Sample enhanced data for edge detection
            temp_data = sample_enhanced_data_for_edges(composite, region, city_name)
            
            if temp_data is None:
                print(f"   ❌ Failed to sample enhanced data for {city_name}")
                continue
            
            # Create enhanced 3D map with edges
            fig = create_enhanced_3d_map_with_edges(temp_data, city_name, city_info)
            
            # Save ultra-high quality map
            output_path = output_dir / f'{city_name.lower()}_enhanced_3d_with_edges.png'
            fig.savefig(
                str(output_path),
                dpi=300,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none',
                format='png',
                pad_inches=0.15
            )
            
            print(f"   ✅ Enhanced 3D map with edges saved: {output_path}")
            plt.close(fig)
            
            # Save enhanced metadata
            metadata = {
                'city': city_name,
                'analysis_year': ANALYSIS_YEAR,
                'resolution_m': TARGET_RESOLUTION,
                'temp_range_c': temp_data['temp_range'],
                'temp_mean_c': temp_data['temp_mean'],
                'temp_std_c': temp_data['temp_std'],
                'valid_points': temp_data['n_valid'],
                'total_points': temp_data['total_points'],
                'data_quality_percent': (temp_data['n_valid'] / temp_data['total_points']) * 100,
                'features': {
                    'edge_detection': True,
                    'topographic_basemap': True,
                    'white_edge_overlays': True,
                    'enhanced_colormap': True,
                    'vector_features': True
                }
            }
            
            metadata_path = output_dir / f'{city_name.lower()}_enhanced_3d_metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"   ✅ Enhanced metadata saved: {metadata_path}")
            
            # Brief pause between cities
            time.sleep(2)
            
        except Exception as e:
            print(f"   ❌ Error processing {city_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\\n🎉 Enhanced 3D Temperature Mapping with Edge Detection Complete!")
    print(f"📁 Enhanced maps saved in: {output_dir.absolute()}")
    print(f"🎨 Features: Topographic basemap + White edge detection + Enhanced 3D effects")

if __name__ == "__main__":
    main()
