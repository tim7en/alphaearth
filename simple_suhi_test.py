#!/usr/bin/env python3
"""
Simple test to check SUHI calculation
"""

import ee

# Initialize Earth Engine
try:
    ee.Initialize(project='ee-sabitovty')
    print("✅ Google Earth Engine initialized successfully!")
except Exception as e:
    print(f"❌ Error initializing Google Earth Engine: {e}")
    exit(1)

# Simple test for Tashkent
def simple_suhi_test():
    print("🧪 SIMPLE SUHI TEST")
    print("=" * 30)
    
    # Tashkent coordinates
    lat, lon = 41.2995, 69.2401
    pt = ee.Geometry.Point([lon, lat])
    urban_area = pt.buffer(15000)  # 15km
    rural_ring = pt.buffer(30000).difference(urban_area)  # 30km ring
    
    print(f"📍 Testing Tashkent: {lat}, {lon}")
    
    # Get 2023 data
    print("📡 Getting 2023 data...")
    
    # MODIS LST
    lst_coll = (ee.ImageCollection('MODIS/061/MOD11A2')
                .filterDate('2023-01-01', '2023-12-31')
                .filterBounds(urban_area)
                .median())
    
    lst_day = lst_coll.select('LST_Day_1km').multiply(0.02).subtract(273.15)
    
    # Dynamic World
    dw = (ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
          .filterDate('2023-01-01', '2023-12-31')
          .filterBounds(urban_area)
          .median())
    
    built_prob = dw.select('built')
    
    # Try different approaches
    print("\n🏙️ Testing different urban detection methods...")
    
    # Method 1: Simple threshold
    print("\n1. Simple built threshold > 0.1:")
    urban_mask = built_prob.gt(0.1)
    
    urban_lst = lst_day.updateMask(urban_mask).reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=urban_area,
        scale=1000,
        bestEffort=True,
        maxPixels=1e8
    )
    
    rural_lst = lst_day.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=rural_ring,
        scale=1000,
        bestEffort=True,
        maxPixels=1e8
    )
    
    try:
        urban_temp = urban_lst.getInfo()['LST_Day_1km']
        rural_temp = rural_lst.getInfo()['LST_Day_1km']
        suhi = urban_temp - rural_temp
        
        print(f"   Urban LST: {urban_temp:.2f}°C")
        print(f"   Rural LST: {rural_temp:.2f}°C")
        print(f"   SUHI: {suhi:.2f}°C")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Method 2: No masking - just area comparison
    print("\n2. Area comparison (no urban masking):")
    
    urban_lst_nomask = lst_day.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=urban_area,
        scale=1000,
        bestEffort=True,
        maxPixels=1e8
    )
    
    try:
        urban_temp_nomask = urban_lst_nomask.getInfo()['LST_Day_1km']
        rural_temp_nomask = rural_lst.getInfo()['LST_Day_1km']
        suhi_nomask = urban_temp_nomask - rural_temp_nomask
        
        print(f"   Urban LST (no mask): {urban_temp_nomask:.2f}°C")
        print(f"   Rural LST: {rural_temp_nomask:.2f}°C")
        print(f"   SUHI (no mask): {suhi_nomask:.2f}°C")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Method 3: Check pixel counts
    print("\n3. Pixel count analysis:")
    
    pixel_count = urban_mask.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=urban_area,
        scale=1000,
        bestEffort=True,
        maxPixels=1e8
    )
    
    total_count = ee.Image.constant(1).reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=urban_area,
        scale=1000,
        bestEffort=True,
        maxPixels=1e8
    )
    
    try:
        urban_pixels = pixel_count.getInfo()['built']
        total_pixels = total_count.getInfo()['constant']
        
        print(f"   Urban pixels (threshold>0.1): {urban_pixels}")
        print(f"   Total pixels in area: {total_pixels}")
        print(f"   Urban percentage: {urban_pixels/total_pixels*100:.1f}%")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    simple_suhi_test()
