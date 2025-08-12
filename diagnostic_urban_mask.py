#!/usr/bin/env python3
"""
Diagnostic script to check Dynamic World built-up probabilities in Uzbekistan cities
"""

import ee
import numpy as np

# Initialize Earth Engine
try:
    ee.Initialize()
    print("✅ Google Earth Engine initialized successfully!")
except Exception as e:
    print(f"❌ Error initializing Google Earth Engine: {e}")
    exit(1)

# Test cities
UZBEKISTAN_CITIES = {
    'Tashkent': {'lat': 41.2995, 'lon': 69.2401, 'buffer': 25000},
    'Samarkand': {'lat': 39.6270, 'lon': 66.9749, 'buffer': 15000},
    'Namangan': {'lat': 40.9983, 'lon': 71.6726, 'buffer': 12000}
}

def check_urban_probabilities():
    """Check what built-up probabilities we actually get"""
    
    for city, info in UZBEKISTAN_CITIES.items():
        print(f"\n🏙️ Checking {city}...")
        
        # Create geometry
        pt = ee.Geometry.Point([info['lon'], info['lat']])
        urban_area = pt.buffer(info['buffer'])
        
        # Get Dynamic World for recent period
        dw = (ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
              .filterDate('2023-01-01', '2023-12-31')
              .filterBounds(urban_area)
              .median())
        
        built = dw.select('built')
        
        # Get statistics
        stats = built.reduceRegion(
            reducer=ee.Reducer.minMax().combine(
                ee.Reducer.mean(), sharedInputs=True
            ).combine(
                ee.Reducer.percentile([10, 25, 50, 75, 90]), sharedInputs=True
            ),
            geometry=urban_area,
            scale=100,  # Use 100m for this diagnostic
            bestEffort=True,
            tileScale=4,
            maxPixels=1e8
        )
        
        try:
            result = stats.getInfo()
            print(f"   Built probability stats:")
            print(f"   - Min: {result.get('built_min', 'N/A'):.3f}")
            print(f"   - Max: {result.get('built_max', 'N/A'):.3f}")
            print(f"   - Mean: {result.get('built_mean', 'N/A'):.3f}")
            print(f"   - P10: {result.get('built_p10', 'N/A'):.3f}")
            print(f"   - P25: {result.get('built_p25', 'N/A'):.3f}")
            print(f"   - P50: {result.get('built_p50', 'N/A'):.3f}")
            print(f"   - P75: {result.get('built_p75', 'N/A'):.3f}")
            print(f"   - P90: {result.get('built_p90', 'N/A'):.3f}")
            
            # Check how many pixels would be urban at different thresholds
            for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
                urban_mask = built.gt(threshold)
                urban_count = urban_mask.reduceRegion(
                    reducer=ee.Reducer.sum(),
                    geometry=urban_area,
                    scale=100,
                    bestEffort=True,
                    tileScale=4,
                    maxPixels=1e8
                ).getInfo()
                
                print(f"   - Urban pixels (>{threshold}): {urban_count.get('built', 0)}")
                
        except Exception as e:
            print(f"   ❌ Error getting stats: {e}")

if __name__ == "__main__":
    print("🔍 DIAGNOSTIC: Dynamic World Built-up Probabilities")
    print("=" * 60)
    
    check_urban_probabilities()
    
    print("\n✅ Diagnostic complete!")
