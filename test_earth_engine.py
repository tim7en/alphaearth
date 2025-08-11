#!/usr/bin/env python3
"""
Quick Earth Engine Test Script
Test if your project registration is approved
"""

import ee

def test_earth_engine():
    print("🌍 Testing Earth Engine Access")
    print("=" * 40)
    
    try:
        # Initialize with your project
        print("🔄 Initializing Earth Engine with your project...")
        ee.Initialize(project='my-earth-engine-project')
        print("✅ Earth Engine initialized successfully!")
        
        # Test basic dataset access
        print("🛰️ Testing satellite data access...")
        
        # Test MODIS data (for temperature)
        modis = ee.ImageCollection('MODIS/006/MOD11A1').first()
        print("✅ MODIS temperature data: accessible")
        
        # Test Sentinel-2 data
        sentinel = ee.ImageCollection('COPERNICUS/S2_SR').first()
        print("✅ Sentinel-2 imagery: accessible")
        
        # Test country boundaries
        uzbekistan = ee.FeatureCollection("FAO/GAUL/2015/level0").filter(ee.Filter.eq('ADM0_NAME', 'Uzbekistan'))
        print("✅ Uzbekistan boundaries: accessible")
        
        print("\n🎉 SUCCESS! Earth Engine is fully working!")
        print("🚀 Ready to run real satellite analysis!")
        print("📄 You can now run: python urban_heat_gee_clean.py")
        
        return True
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Error: {error_msg}")
        
        if "not registered" in error_msg or "not signed up" in error_msg:
            print("\n⏳ Registration still pending")
            print("💡 Check your email for approval notification")
            print("🔗 Or check: https://code.earthengine.google.com/")
        elif "permission" in error_msg:
            print("\n🔑 Project permissions need to be enabled")
            print("⏳ This should happen automatically after registration approval")
        else:
            print(f"\n🤔 Unexpected error: {error_msg}")
            
        return False

if __name__ == "__main__":
    test_earth_engine()
