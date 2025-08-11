#!/usr/bin/env python3
"""
Simple Google Earth Engine Authentication Test

This script tries multiple approaches to authenticate Google Earth Engine.
"""

import os
import sys

def try_ee_authenticate():
    """Try Earth Engine authentication with different methods"""
    
    print("🔄 Attempting Google Earth Engine authentication...")
    
    try:
        import ee
        
        # Method 1: Try direct initialization (in case already authenticated)
        try:
            print("  Testing existing authentication...")
            ee.Initialize()
            result = ee.Number(42).getInfo()
            print(f"  ✅ Already authenticated! Test result: {result}")
            return True
        except:
            print("  ❌ Not yet authenticated")
        
        # Method 2: Try ee.Authenticate() without parameters
        try:
            print("  Attempting ee.Authenticate()...")
            ee.Authenticate()
            ee.Initialize()
            result = ee.Number(123).getInfo()
            print(f"  ✅ Authentication successful! Test result: {result}")
            return True
        except Exception as e:
            print(f"  ❌ ee.Authenticate() failed: {e}")
        
        # Method 3: Try with use_cloud_api flag
        try:
            print("  Attempting with cloud API...")
            ee.Authenticate(use_cloud_api=True)
            ee.Initialize()
            result = ee.Number(456).getInfo()
            print(f"  ✅ Cloud API authentication successful! Test result: {result}")
            return True
        except Exception as e:
            print(f"  ❌ Cloud API authentication failed: {e}")
        
        return False
        
    except ImportError:
        print("❌ earthengine-api not available")
        return False

def test_basic_gee_functionality():
    """Test basic Google Earth Engine functionality"""
    
    print("\n🧪 Testing Google Earth Engine functionality...")
    
    try:
        import ee
        
        # Initialize
        ee.Initialize()
        
        # Test 1: Basic number operation
        num_test = ee.Number(2025).getInfo()
        print(f"  ✅ Number test: {num_test}")
        
        # Test 2: Image access
        image = ee.Image('USGS/SRTMGL1_003')
        scale = image.projection().nominalScale().getInfo()
        print(f"  ✅ Image test: SRTM scale = {scale}m")
        
        # Test 3: Geometry operation
        point = ee.Geometry.Point([69.2401, 41.2995])  # Tashkent
        buffer = point.buffer(1000)
        area = buffer.area().getInfo()
        print(f"  ✅ Geometry test: Buffer area = {area:.0f} m²")
        
        # Test 4: Image collection
        modis = ee.ImageCollection('MODIS/061/MOD11A1')
        count = modis.filterDate('2023-01-01', '2023-01-02').size().getInfo()
        print(f"  ✅ Collection test: {count} MODIS images found")
        
        print("✅ All Google Earth Engine tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ GEE functionality test failed: {e}")
        return False

def main():
    """Main test function"""
    
    print("🌍 Google Earth Engine Authentication & Functionality Test")
    print("=" * 60)
    
    # Try authentication
    auth_success = try_ee_authenticate()
    
    if auth_success:
        # Test functionality
        func_success = test_basic_gee_functionality()
        
        if func_success:
            print("\n🎉 Google Earth Engine is fully operational!")
            print("🚀 You can now run the urban heat analysis:")
            print("   python urban_heat_gee_standalone.py")
            return True
    
    print("\n❌ Authentication or functionality test failed")
    print("\n🔧 Next steps:")
    print("1. Try: earthengine authenticate")
    print("2. Or try: python gee_auth_helper.py")
    print("3. Check Google Cloud Console for API enablement")
    
    return False

if __name__ == "__main__":
    main()
