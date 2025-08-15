#!/usr/bin/env python3
"""
Google Earth Engine Authentication with Project ID

This script authenticates GEE using a specific project ID.

Author: AlphaEarth Analysis Team
Date: August 15, 2025
"""

import os
import sys
import json
from pathlib import Path

def authenticate_with_project():
    """Authenticate Google Earth Engine with specific project"""
    
    print("🔐 Google Earth Engine Authentication with Project")
    print("=" * 55)
    
    # Your project ID
    project_id = "ee-sabitovty"
    print(f"🎯 Using project: {project_id}")
    
    try:
        import ee
        print("✅ Earth Engine API imported successfully")
        
        # Try to initialize with your project
        print(f"\n🔄 Initializing Earth Engine with project: {project_id}")
        
        try:
            ee.Initialize(project=project_id)
            print("✅ Earth Engine initialized successfully with your project!")
            
            # Test connection
            return test_connection_with_project(project_id)
            
        except Exception as e:
            print(f"❌ Project initialization failed: {e}")
            print("🔄 Trying authentication first...")
            
            # Try authentication then initialization
            try:
                ee.Authenticate()
                print("✅ Authentication completed!")
                
                ee.Initialize(project=project_id)
                print("✅ Earth Engine initialized with your project!")
                
                return test_connection_with_project(project_id)
                
            except Exception as e2:
                print(f"❌ Authentication + initialization failed: {e2}")
                return try_alternative_project_methods(project_id)
        
    except ImportError:
        print("❌ Earth Engine API not available")
        print("Please install with: pip install earthengine-api")
        return False

def try_alternative_project_methods(project_id):
    """Try alternative methods with project"""
    
    print(f"\n🔄 Trying alternative methods with project {project_id}...")
    
    try:
        import ee
        
        # Method 1: Set environment variable
        print("1. Setting project environment variable...")
        os.environ['GOOGLE_CLOUD_PROJECT'] = project_id
        try:
            ee.Initialize()
            print("✅ Environment variable method successful!")
            return test_connection_with_project(project_id)
        except Exception as e:
            print(f"   Failed: {e}")
        
        # Method 2: Try with cloud platform URL
        print("2. Trying with cloud platform URL...")
        try:
            ee.Initialize(project=project_id, opt_url='https://earthengine.googleapis.com')
            print("✅ Cloud platform URL method successful!")
            return test_connection_with_project(project_id)
        except Exception as e:
            print(f"   Failed: {e}")
        
        # Method 3: Manual credentials path
        print("3. Checking manual credentials...")
        home_creds = Path.home() / '.config' / 'earthengine' / 'credentials'
        if home_creds.exists():
            try:
                ee.Initialize(project=project_id)
                print("✅ Manual credentials method successful!")
                return test_connection_with_project(project_id)
            except Exception as e:
                print(f"   Failed: {e}")
        
        print("⚠️ All project methods failed")
        return False
        
    except Exception as e:
        print(f"❌ Alternative project methods failed: {e}")
        return False

def test_connection_with_project(project_id):
    """Test Earth Engine connection with project"""
    
    print(f"\n🧪 Testing Google Earth Engine connection with project {project_id}...")
    
    try:
        import ee
        
        # Test basic computation
        test_number = ee.Number(2025).getInfo()
        print(f"✅ Basic computation test: {test_number}")
        
        # Test image access
        image = ee.Image('USGS/SRTMGL1_003')
        scale = image.projection().nominalScale().getInfo()
        print(f"✅ Image access test successful! SRTM scale: {scale}m")
        
        # Test collections access
        collection = ee.ImageCollection('MODIS/061/MOD11A1')
        size = collection.limit(1).size().getInfo()
        print(f"✅ Collection access test: {size} images accessible")
        
        # Test Uzbekistan-specific data
        print("🇺🇿 Testing Uzbekistan-specific data access...")
        uzbekistan = ee.Geometry.Rectangle([55.9, 37.2, 73.2, 45.6])
        
        modis_collection = ee.ImageCollection('MODIS/061/MOD11A1') \
            .filterDate('2023-01-01', '2023-01-31') \
            .filterBounds(uzbekistan)
        
        count = modis_collection.size().getInfo()
        print(f"✅ Uzbekistan data test: Found {count} MODIS images for January 2023")
        
        # Test emissions-specific datasets if available
        try:
            # Test CO2 emissions data
            emissions_test = ee.ImageCollection('ODIAC/FOSSILFUEL_CO2').limit(1)
            if emissions_test.size().getInfo() > 0:
                print("✅ CO2 emissions dataset accessible!")
            else:
                print("⚠️  CO2 emissions dataset may have limited access")
        except Exception as e:
            print(f"⚠️  CO2 emissions dataset test: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False

def save_project_auth_status(authenticated=False, project_id="ee-sabitovty"):
    """Save authentication status with project info"""
    
    status_file = Path.cwd() / ".gee_auth_status.json"
    
    status = {
        "authenticated": authenticated,
        "project_id": project_id,
        "timestamp": str(Path(__file__).stat().st_mtime),
        "method": "project_auth" if authenticated else "simulation"
    }
    
    with open(status_file, 'w') as f:
        json.dump(status, f, indent=2)
    
    print(f"💾 Authentication status saved to: {status_file}")

def provide_project_troubleshooting(project_id):
    """Provide project-specific troubleshooting"""
    
    print(f"\n🔧 Troubleshooting for project: {project_id}")
    print("=" * 50)
    
    print("\n**Check Project Access:**")
    print(f"  1. Go to: https://console.cloud.google.com/")
    print(f"  2. Select project: {project_id}")
    print(f"  3. Check if Earth Engine API is enabled")
    print(f"  4. Verify you have proper permissions")
    
    print("\n**Check Earth Engine Registration:**")
    print(f"  1. Go to: https://code.earthengine.google.com/")
    print(f"  2. Sign in with same Google account")
    print(f"  3. Check if you can access the code editor")
    print(f"  4. Try running a simple script there first")
    
    print("\n**Alternative Authentication:**")
    print(f"  1. Try: earthengine authenticate --project {project_id}")
    print(f"  2. Or set environment: $env:GOOGLE_CLOUD_PROJECT='{project_id}'")
    print(f"  3. Then run: python -c \"import ee; ee.Initialize(); print('Success!')\"")

def main():
    """Main authentication function with project"""
    
    project_id = "ee-sabitovty"
    success = authenticate_with_project()
    save_project_auth_status(success, project_id)
    
    if success:
        print(f"\n🎉 Google Earth Engine authentication successful!")
        print(f"✅ Connected to project: {project_id}")
        print("\n🚀 You can now run your analysis scripts with real data:")
        print("   python ghg_emissions_uzb/ghg_downscaling_uzb.py")
        print("   python urban_heat_gee_standalone.py")
    else:
        print(f"\n⚠️  Google Earth Engine authentication not complete for project: {project_id}")
        provide_project_troubleshooting(project_id)
        print("\n📝 Your analysis scripts will run in simulation mode")
    
    return success

if __name__ == "__main__":
    main()
