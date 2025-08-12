#!/usr/bin/env python3
"""
Debug script to investigate urban pixel detection issues
"""

import ee
import pandas as pd

# Initialize Earth Engine
try:
    ee.Initialize(project='ee-sabitovty')
    print("✅ Google Earth Engine initialized successfully!")
except Exception as e:
    print(f"❌ Error initializing Google Earth Engine: {e}")
    exit(1)

# Test with Tashkent
TASHKENT = {"lat": 41.2995, "lon": 69.2401, "buffer": 15000}

def debug_urban_detection():
    """Debug what's happening with urban pixel detection"""
    
    print("🔍 DEBUGGING URBAN PIXEL DETECTION")
    print("=" * 50)
    
    # Create geometry
    pt = ee.Geometry.Point([TASHKENT['lon'], TASHKENT['lat']])
    urban_area = pt.buffer(TASHKENT['buffer'])
    
    print(f"📍 Testing Tashkent: {TASHKENT['lat']}, {TASHKENT['lon']}")
    print(f"🎯 Buffer size: {TASHKENT['buffer']/1000} km")
    
    # Get Dynamic World for 2023
    print("\n📡 Getting Dynamic World data for 2023...")
    dw = (ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
          .filterDate('2023-01-01', '2023-12-31')
          .filterBounds(urban_area)
          .median())
    
    built = dw.select('built')
    
    print("🏗️ Testing different urban thresholds...")
    
    # Test different thresholds
    thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5]
    
    for threshold in thresholds:
        print(f"\n🎯 Testing threshold: {threshold}")
        
        urban_mask = built.gt(threshold)
        
        # Count urban pixels at 1km scale
        urban_count = urban_mask.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=urban_area,
            scale=1000,  # 1km like our analysis
            bestEffort=True,
            tileScale=4,
            maxPixels=1e8
        )
        
        try:
            count = urban_count.getInfo()['built']
            print(f"   Urban pixels (1km): {count}")
            
            # Also check at 100m scale for comparison
            urban_count_100m = urban_mask.reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=urban_area,
                scale=100,
                bestEffort=True,
                tileScale=4,
                maxPixels=1e8
            )
            count_100m = urban_count_100m.getInfo()['built']
            print(f"   Urban pixels (100m): {count_100m}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Check built probability statistics
    print("\n📊 Built probability statistics...")
    stats = built.reduceRegion(
        reducer=ee.Reducer.minMax().combine(
            ee.Reducer.mean(), sharedInputs=True
        ).combine(
            ee.Reducer.percentile([10, 25, 50, 75, 90]), sharedInputs=True
        ),
        geometry=urban_area,
        scale=100,
        bestEffort=True,
        tileScale=4,
        maxPixels=1e8
    )
    
    try:
        result = stats.getInfo()
        print(f"   Min: {result.get('built_min', 'N/A'):.4f}")
        print(f"   Max: {result.get('built_max', 'N/A'):.4f}")
        print(f"   Mean: {result.get('built_mean', 'N/A'):.4f}")
        print(f"   P10: {result.get('built_p10', 'N/A'):.4f}")
        print(f"   P25: {result.get('built_p25', 'N/A'):.4f}")
        print(f"   P50: {result.get('built_p50', 'N/A'):.4f}")
        print(f"   P75: {result.get('built_p75', 'N/A'):.4f}")
        print(f"   P90: {result.get('built_p90', 'N/A'):.4f}")
        
        # Suggest optimal threshold
        p75 = result.get('built_p75', 0)
        print(f"\n💡 Suggested threshold based on P75: {p75:.3f}")
        
    except Exception as e:
        print(f"   ❌ Error getting stats: {e}")

if __name__ == "__main__":
    debug_urban_detection()
