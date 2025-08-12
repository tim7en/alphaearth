#!/usr/bin/env python3
"""
High-Resolution Urban Heat Analysis using Landsat 8/9 and Sentinel-2
====================================================================
Creates professional-quality temperature maps with 100-250m resolution
Uses real satellite data only - no mock or simulated data
"""

import ee
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
import seaborn as sns
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# High-resolution city configuration
UZBEKISTAN_CITIES = {
    "Tashkent": {"lat": 41.2995, "lon": 69.2401, "buffer": 20000},  # 20km buffer
    "Nukus": {"lat": 42.4731, "lon": 59.6103, "buffer": 15000},    # 15km buffer
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

def get_high_resolution_lst(geometry, start_date, end_date):
    """
    Get high-resolution land surface temperature from Landsat 8/9
    Spatial resolution: 30m thermal band resampled to 100m
    """
    print("🛰️ Retrieving high-resolution LST from Landsat 8/9...")
    
    try:
        # Landsat 8/9 Collection 2 Level 2
        landsat8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
        landsat9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
        
        # Combine collections
        landsat = landsat8.merge(landsat9)
        
        # Filter by date, location, and cloud cover
        filtered = landsat.filterDate(start_date, end_date) \
                         .filterBounds(geometry) \
                         .filter(ee.Filter.lt('CLOUD_COVER', 20))
        
        print(f"   📊 Found {filtered.size().getInfo()} Landsat scenes")
        
        if filtered.size().getInfo() == 0:
            print("   ❌ No Landsat data available for the specified period")
            return None
        
        def process_landsat_lst(image):
            """Process Landsat thermal data to LST"""
            # Thermal band (ST_B10 for Collection 2 Level 2)
            thermal = image.select('ST_B10')
            
            # Apply scale factor and convert to Celsius
            # Collection 2 Level 2 scale factor: 0.00341802, offset: 149.0
            lst_celsius = thermal.multiply(0.00341802).add(149.0).subtract(273.15)
            
            # Quality mask using QA_PIXEL
            qa = image.select('QA_PIXEL')
            
            # Bit positions for cloud, cloud shadow, snow
            cloud_bit = 1 << 3
            cloud_shadow_bit = 1 << 4
            snow_bit = 1 << 5
            
            # Create mask
            mask = qa.bitwiseAnd(cloud_bit).eq(0) \
                    .And(qa.bitwiseAnd(cloud_shadow_bit).eq(0)) \
                    .And(qa.bitwiseAnd(snow_bit).eq(0))
            
            return lst_celsius.updateMask(mask).rename('LST')
        
        # Process all images
        lst_collection = filtered.map(process_landsat_lst)
        
        # Create median composite
        lst_median = lst_collection.median()
        
        # Clamp to reasonable temperature range for Uzbekistan
        lst_final = lst_median.clamp(-10, 60)
        
        print("   ✅ High-resolution LST processing complete")
        return lst_final
        
    except Exception as e:
        print(f"   ❌ Error processing Landsat LST: {e}")
        return None

def get_sentinel2_ndvi(geometry, start_date, end_date):
    """
    Get high-resolution NDVI from Sentinel-2
    Spatial resolution: 10m
    """
    print("🛰️ Retrieving high-resolution NDVI from Sentinel-2...")
    
    try:
        # Sentinel-2 Surface Reflectance
        sentinel2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        
        # Filter by date, location, and cloud cover
        filtered = sentinel2.filterDate(start_date, end_date) \
                           .filterBounds(geometry) \
                           .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
        
        print(f"   📊 Found {filtered.size().getInfo()} Sentinel-2 scenes")
        
        if filtered.size().getInfo() == 0:
            print("   ❌ No Sentinel-2 data available")
            return None
        
        def process_sentinel_ndvi(image):
            """Calculate NDVI from Sentinel-2"""
            # Cloud masking using scene classification
            scl = image.select('SCL')
            
            # Mask clouds, cloud shadows, and snow
            clear_mask = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
            
            # Calculate NDVI
            ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
            
            return ndvi.updateMask(clear_mask)
        
        # Process all images
        ndvi_collection = filtered.map(process_sentinel_ndvi)
        
        # Create median composite
        ndvi_median = ndvi_collection.median()
        
        print("   ✅ High-resolution NDVI processing complete")
        return ndvi_median
        
    except Exception as e:
        print(f"   ❌ Error processing Sentinel-2 NDVI: {e}")
        return None

def sample_high_resolution_data(lst_image, ndvi_image, geometry, target_resolution=100):
    """
    Sample high-resolution data at specified resolution
    Returns coordinates and values for mapping
    """
    import numpy as np
    
    print(f"🔬 Sampling data at {target_resolution}m resolution...")
    
    try:
        # Create sampling grid
        bounds = geometry.bounds().getInfo()['coordinates'][0]
        
        # Calculate bounds
        min_lon = min([coord[0] for coord in bounds])
        max_lon = max([coord[0] for coord in bounds])
        min_lat = min([coord[1] for coord in bounds])
        max_lat = max([coord[1] for coord in bounds])
        
        # Calculate number of points based on target resolution
        # Approximate degrees per meter at this latitude
        lat_center = (min_lat + max_lat) / 2
        meters_per_degree = 111000 * np.cos(np.radians(lat_center))
        
        lon_range = max_lon - min_lon
        lat_range = max_lat - min_lat
        
        n_lon = max(10, int((lon_range * meters_per_degree) / target_resolution))
        n_lat = max(10, int((lat_range * 111000) / target_resolution))
        
        # Limit to manageable size
        n_lon = min(n_lon, 100)
        n_lat = min(n_lat, 100)
        
        print(f"   📐 Creating {n_lat}x{n_lon} sampling grid")
        
        # Create coordinate arrays
        lons = np.linspace(min_lon, max_lon, n_lon)
        lats = np.linspace(min_lat, max_lat, n_lat)
        
        # Create point collection for sampling
        points = []
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                points.append(ee.Feature(ee.Geometry.Point([lon, lat]), {'lat_idx': i, 'lon_idx': j}))
        
        point_collection = ee.FeatureCollection(points)
        
        # Sample LST data
        if lst_image is not None:
            print("   🌡️ Sampling LST data...")
            lst_sampled = lst_image.sampleRegions(
                collection=point_collection,
                scale=target_resolution,
                geometries=True
            ).getInfo()
        else:
            lst_sampled = None
        
        # Sample NDVI data
        if ndvi_image is not None:
            print("   🌿 Sampling NDVI data...")
            ndvi_sampled = ndvi_image.sampleRegions(
                collection=point_collection,
                scale=target_resolution,
                geometries=True
            ).getInfo()
        else:
            ndvi_sampled = None
        
        # Process results
        results = {}
        
        if lst_sampled and lst_sampled['features']:
            lst_data = []
            for feature in lst_sampled['features']:
                props = feature['properties']
                if 'LST' in props and props['LST'] is not None:
                    coords = feature['geometry']['coordinates']
                    lst_data.append([coords[0], coords[1], props['LST'], props['lat_idx'], props['lon_idx']])
            
            if lst_data:
                lst_array = np.array(lst_data)
                
                # Create regular grid
                LON, LAT = np.meshgrid(lons, lats)
                LST_grid = np.full_like(LON, np.nan)
                
                # Fill grid with sampled values
                for row in lst_array:
                    lon, lat, temp, lat_idx, lon_idx = row
                    if not np.isnan(temp) and -50 < temp < 70:
                        LST_grid[int(lat_idx), int(lon_idx)] = temp
                
                # Interpolate missing values
                if np.sum(~np.isnan(LST_grid)) > 10:
                    try:
                        from scipy.interpolate import griddata
                        
                        # Get valid points
                        valid_mask = ~np.isnan(LST_grid)
                        if np.sum(valid_mask) > 0:
                            valid_coords = np.column_stack([LON[valid_mask], LAT[valid_mask]])
                            valid_temps = LST_grid[valid_mask]
                            
                            # Interpolate
                            LST_interp = griddata(valid_coords, valid_temps, (LON, LAT), 
                                                method='linear', fill_value=np.nan)
                            
                            # Fill remaining NaNs with nearest neighbor
                            nan_mask = np.isnan(LST_interp)
                            if np.sum(nan_mask) > 0:
                                LST_nn = griddata(valid_coords, valid_temps, (LON, LAT), 
                                                method='nearest')
                                LST_interp = np.where(nan_mask, LST_nn, LST_interp)
                            
                            results['LST'] = {'lons': LON, 'lats': LAT, 'values': LST_interp}
                            print(f"   ✅ LST grid created: {np.nanmin(LST_interp):.1f}°C to {np.nanmax(LST_interp):.1f}°C")
                    
                    except ImportError:
                        print("   ⚠️ Scipy not available for interpolation")
                        results['LST'] = {'lons': LON, 'lats': LAT, 'values': LST_grid}
                else:
                    results['LST'] = {'lons': LON, 'lats': LAT, 'values': LST_grid}
        
        if ndvi_sampled and ndvi_sampled['features']:
            ndvi_data = []
            for feature in ndvi_sampled['features']:
                props = feature['properties']
                if 'NDVI' in props and props['NDVI'] is not None:
                    coords = feature['geometry']['coordinates']
                    ndvi_data.append([coords[0], coords[1], props['NDVI'], props['lat_idx'], props['lon_idx']])
            
            if ndvi_data:
                ndvi_array = np.array(ndvi_data)
                
                # Create NDVI grid
                NDVI_grid = np.full_like(LON, np.nan)
                
                for row in ndvi_array:
                    lon, lat, ndvi, lat_idx, lon_idx = row
                    if not np.isnan(ndvi) and -1 <= ndvi <= 1:
                        NDVI_grid[int(lat_idx), int(lon_idx)] = ndvi
                
                results['NDVI'] = {'lons': LON, 'lats': LAT, 'values': NDVI_grid}
                print(f"   ✅ NDVI grid created: {np.nanmin(NDVI_grid):.2f} to {np.nanmax(NDVI_grid):.2f}")
        
        return results
        
    except Exception as e:
        print(f"   ❌ Error sampling high-resolution data: {e}")
        return {}

def create_professional_temperature_map(city_name, city_info, output_dir):
    """
    Create a professional temperature map similar to the reference image
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import numpy as np
    
    print(f"\n🗺️ Creating professional temperature map for {city_name}...")
    
    try:
        # Import contextily for basemap
        import contextily as ctx
        basemap_available = True
    except ImportError:
        basemap_available = False
        print("   ⚠️ Contextily not available, creating map without basemap")
    
    # Set up geometry
    pt = ee.Geometry.Point([city_info['lon'], city_info['lat']])
    geometry = pt.buffer(city_info['buffer'])
    
    # Date range for analysis (recent summer period)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)  # Last 3 months
    
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    print(f"   📅 Analysis period: {start_str} to {end_str}")
    
    # Get high-resolution satellite data
    lst_image = get_high_resolution_lst(geometry, start_str, end_str)
    ndvi_image = get_sentinel2_ndvi(geometry, start_str, end_str)
    
    if lst_image is None:
        print(f"   ❌ No temperature data available for {city_name}")
        return None
    
    # Sample data at high resolution
    sampled_data = sample_high_resolution_data(lst_image, ndvi_image, geometry, target_resolution=100)
    
    if 'LST' not in sampled_data:
        print(f"   ❌ No valid temperature data sampled for {city_name}")
        return None
    
    # Extract data
    lst_data = sampled_data['LST']
    LON, LAT, TEMPS = lst_data['lons'], lst_data['lats'], lst_data['values']
    
    # Create the map
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    
    # Calculate map extent
    buffer_deg = city_info['buffer'] / 111000
    center_lon, center_lat = city_info['lon'], city_info['lat']
    
    west, east = center_lon - buffer_deg, center_lon + buffer_deg
    south, north = center_lat - buffer_deg, center_lat + buffer_deg
    
    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    
    # Add basemap if available
    if basemap_available:
        try:
            # Add satellite imagery basemap
            ctx.add_basemap(ax, crs='EPSG:4326', 
                          source=ctx.providers.Esri.WorldImagery, 
                          alpha=0.3, zoom=12)
            print("   ✅ Added satellite basemap")
        except Exception as e:
            print(f"   ⚠️ Could not add basemap: {e}")
            ax.set_facecolor('#f0f0f0')
    else:
        ax.set_facecolor('#f0f0f0')
    
    # Create temperature contours (similar to reference image)
    valid_mask = ~np.isnan(TEMPS)
    if np.sum(valid_mask) > 10:
        # Define temperature levels
        temp_min, temp_max = np.nanmin(TEMPS), np.nanmax(TEMPS)
        print(f"   📊 Temperature range: {temp_min:.1f}°C to {temp_max:.1f}°C")
        
        # Create smooth contours
        levels = np.linspace(temp_min, temp_max, 20)
        
        # Main temperature fill
        contourf = ax.contourf(LON, LAT, TEMPS, levels=levels, 
                              cmap='RdYlBu_r', alpha=0.8, extend='both')
        
        # Add contour lines for detail
        contour_lines = ax.contour(LON, LAT, TEMPS, levels=levels[::2], 
                                  colors='white', linewidths=0.5, alpha=0.6)
        
        # Add street-level detail if NDVI available
        if 'NDVI' in sampled_data:
            ndvi_data = sampled_data['NDVI']
            NDVI = ndvi_data['values']
            
            # Overlay vegetation areas in green tones
            vegetation_mask = NDVI > 0.3
            if np.sum(vegetation_mask) > 0:
                ax.contour(LON, LAT, NDVI, levels=[0.3, 0.5, 0.7], 
                          colors=['green'], linewidths=1, alpha=0.4)
        
        # Add city center marker
        ax.scatter(center_lon, center_lat, s=300, c='white', marker='*', 
                  edgecolors='black', linewidth=2, label='City Center', zorder=10)
        
        # Add city boundary
        city_boundary = Circle((center_lon, center_lat), buffer_deg*0.7, 
                              fill=False, edgecolor='black', 
                              linewidth=2, linestyle='--', alpha=0.8, label='City Boundary')
        ax.add_patch(city_boundary)
        
        # Create professional colorbar (similar to reference)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.1)
        
        cbar = plt.colorbar(contourf, cax=cax)
        cbar.set_label('Ambient air temperature', fontsize=14, fontweight='bold', pad=20)
        
        # Format colorbar like reference image
        cbar.ax.tick_params(labelsize=12)
        
        # Add temperature labels at key points
        temp_labels = np.linspace(temp_min, temp_max, 5)
        for temp in temp_labels:
            cbar.ax.text(1.5, temp, f'{temp:.0f}°C', 
                        transform=cbar.ax.transData, fontsize=11, 
                        ha='left', va='center')
        
        # Set title and labels
        ax.set_title(f'{city_name} - High-Resolution Temperature Analysis\n'
                    f'Land Surface Temperature from Landsat 8/9 (100m resolution)', 
                    fontsize=16, fontweight='bold', pad=20)
        
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        
        # Add scale bar
        try:
            # Calculate scale bar length (5 km)
            scale_length_deg = 5000 / 111000  # 5 km in degrees
            scale_x = west + (east - west) * 0.05
            scale_y = south + (north - south) * 0.05
            
            # Draw scale bar
            ax.plot([scale_x, scale_x + scale_length_deg], [scale_y, scale_y], 
                   color='black', linewidth=3)
            ax.text(scale_x + scale_length_deg/2, scale_y + (north-south)*0.02, 
                   '5 km', ha='center', va='bottom', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        except:
            pass
        
        # Add north arrow
        try:
            arrow_x = east - (east - west) * 0.1
            arrow_y = north - (north - south) * 0.1
            ax.annotate('N', xy=(arrow_x, arrow_y), xytext=(arrow_x, arrow_y - (north-south)*0.05),
                       arrowprops=dict(arrowstyle='->', color='black', lw=2),
                       fontsize=12, fontweight='bold', ha='center')
        except:
            pass
        
        # Add legend
        legend_elements = [
            Line2D([0], [0], marker='*', color='w', markerfacecolor='white', 
                   markersize=15, label='City Center', markeredgecolor='black', markeredgewidth=2),
            Line2D([0], [0], linestyle='--', color='black', linewidth=2, label='City Boundary')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=10, framealpha=0.9)
        
        # Format axes
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        # Save the map
        output_path = output_dir / f'{city_name.lower()}_high_resolution_temperature_map.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        
        print(f"   ✅ Professional temperature map saved: {output_path}")
        
        plt.close()
        return output_path
    
    else:
        print(f"   ❌ Insufficient valid temperature data for mapping")
        plt.close()
        return None

def main():
    """Main function to create high-resolution temperature maps"""
    
    print("🌡️ HIGH-RESOLUTION URBAN TEMPERATURE MAPPING")
    print("=" * 60)
    print("📊 Using Landsat 8/9 (100m) + Sentinel-2 (10m)")
    print("🚫 No mock data - real satellite observations only")
    print("=" * 60)
    
    # Initialize Google Earth Engine
    if not authenticate_gee():
        return
    
    # Create output directory
    output_dir = Path("high_resolution_temperature_maps")
    output_dir.mkdir(exist_ok=True)
    
    # Process each city
    for city_name, city_info in UZBEKISTAN_CITIES.items():
        print(f"\n🏙️ Processing {city_name}...")
        
        try:
            map_path = create_professional_temperature_map(city_name, city_info, output_dir)
            
            if map_path:
                print(f"✅ Successfully created temperature map for {city_name}")
            else:
                print(f"❌ Failed to create temperature map for {city_name}")
                
        except Exception as e:
            print(f"❌ Error processing {city_name}: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 HIGH-RESOLUTION TEMPERATURE MAPPING COMPLETE!")
    print(f"📁 Maps saved in: {output_dir.absolute()}")
    print("=" * 60)

if __name__ == "__main__":
    main()
