#!/usr/bin/env python3
"""
Google Earth Engine Authentication Test and Setup

This script provides multiple authentication methods for Google Earth Engine.
It tests different authentication approaches and helps diagnose connection issues.

Author: AlphaEarth Analysis Team
Date: August 11, 2025
"""

import os
import sys
import json
from pathlib import Path

def test_authentication_method_1():
    """Method 1: Interactive Browser Authentication"""
    print("🔐 Testing Method 1: Interactive Browser Authentication")
    
    try:
        import ee
        
        # Try to authenticate interactively
        print("  Attempting interactive authentication...")
        ee.Authenticate()
        
        # Initialize after authentication
        ee.Initialize()
        
        # Test basic functionality
        print("  Testing basic GEE functionality...")
        image = ee.Image('USGS/SRTMGL1_003')
        scale = image.projection().nominalScale().getInfo()
        print(f"  ✅ Success! SRTM scale: {scale}m")
        return True
        
    except Exception as e:
        print(f"  ❌ Method 1 failed: {e}")
        return False

def test_authentication_method_2():
    """Method 2: Initialize with project specification"""
    print("\n🔐 Testing Method 2: Project-based Authentication")
    
    try:
        import ee
        
        # Try to initialize with explicit project
        print("  Attempting project-based initialization...")
        
        # You can specify your Google Cloud project here
        # ee.Initialize(project='your-project-id')
        ee.Initialize()
        
        # Test basic functionality
        print("  Testing basic GEE functionality...")
        uzbekistan = ee.Geometry.Rectangle([55.9, 37.2, 73.2, 45.6])
        area = uzbekistan.area().getInfo()
        print(f"  ✅ Success! Uzbekistan area: {area/1e6:.0f} km²")
        return True
        
    except Exception as e:
        print(f"  ❌ Method 2 failed: {e}")
        return False

def test_authentication_method_3():
    """Method 3: Service Account Authentication"""
    print("\n🔐 Testing Method 3: Service Account Authentication")
    
    try:
        import ee
        
        # Look for service account credentials
        possible_paths = [
            os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'),
            'service-account-key.json',
            'credentials.json',
            os.path.expanduser('~/.config/earthengine/credentials')
        ]
        
        credentials_found = False
        for path in possible_paths:
            if path and Path(path).exists():
                print(f"  Found credentials at: {path}")
                credentials_found = True
                break
        
        if not credentials_found:
            print("  ℹ️ No service account credentials found")
            print("  To use this method:")
            print("    1. Create a service account in Google Cloud Console")
            print("    2. Download the JSON key file")
            print("    3. Set GOOGLE_APPLICATION_CREDENTIALS environment variable")
            return False
        
        # Initialize with service account
        ee.Initialize()
        
        # Test basic functionality
        print("  Testing basic GEE functionality...")
        modis = ee.ImageCollection('MODIS/061/MOD11A1')
        count = modis.filterDate('2023-01-01', '2023-01-02').size().getInfo()
        print(f"  ✅ Success! MODIS images in test period: {count}")
        return True
        
    except Exception as e:
        print(f"  ❌ Method 3 failed: {e}")
        return False

def test_rest_api_access():
    """Test REST API access directly"""
    print("\n🌐 Testing REST API Access")
    
    try:
        import requests
        
        # Test basic REST API endpoint
        url = "https://earthengine.googleapis.com/v1/projects"
        
        # This would need proper authentication headers
        print("  ℹ️ REST API testing requires additional setup")
        print("  See: https://developers.google.com/earth-engine/reference/rest")
        print("  For now, using Python API is recommended")
        
        return False
        
    except Exception as e:
        print(f"  ❌ REST API test failed: {e}")
        return False

def get_authentication_status():
    """Check current authentication status"""
    print("\n📊 Checking Current Authentication Status")
    
    try:
        import ee
        
        # Check if already initialized
        try:
            # Try a simple operation
            ee.Number(1).getInfo()
            print("  ✅ Google Earth Engine is already authenticated and ready!")
            return True
        except:
            print("  ❌ Google Earth Engine not yet authenticated")
            return False
            
    except ImportError:
        print("  ❌ earthengine-api not installed")
        return False

def setup_authentication_guide():
    """Provide step-by-step authentication guide"""
    print("\n📚 Google Earth Engine Authentication Guide")
    print("=" * 60)
    
    print("\n🔧 Recommended Setup Process:")
    print("\n1. **Enable Earth Engine API**")
    print("   ✅ You've already enabled earthengine.googleapis.com")
    print("   ✅ Service is active at: https://developers.google.com/earth-engine/reference/rest")
    
    print("\n2. **Choose Authentication Method**")
    print("\n   **Option A: Interactive Authentication (Recommended)**")
    print("   ```bash")
    print("   earthengine authenticate")
    print("   ```")
    print("   - Opens browser for Google sign-in")
    print("   - Stores credentials locally")
    print("   - Works for personal/development use")
    
    print("\n   **Option B: Service Account (For Production)**")
    print("   - Create service account in Google Cloud Console")
    print("   - Download JSON credentials file")
    print("   - Set environment variable:")
    print("   ```bash")
    print("   export GOOGLE_APPLICATION_CREDENTIALS='path/to/credentials.json'")
    print("   ```")
    
    print("\n   **Option C: Application Default Credentials**")
    print("   ```bash")
    print("   gcloud auth application-default login")
    print("   ```")
    
    print("\n3. **Test Authentication**")
    print("   ```python")
    print("   import ee")
    print("   ee.Initialize()")
    print("   print(ee.Number(1).getInfo())  # Should print: 1")
    print("   ```")
    
    print("\n🔍 **Troubleshooting Common Issues:**")
    print("\n   • **Error: 'Please make sure you have initialized the Earth Engine library'**")
    print("     → Run authentication first, then ee.Initialize()")
    print("\n   • **Error: 'Project not found'**")
    print("     → Specify project: ee.Initialize(project='your-project-id')")
    print("\n   • **Error: 'Authentication required'**")
    print("     → Check that EE API is enabled in Google Cloud Console")
    print("\n   • **Error: 'Quota exceeded'**")
    print("     → Reduce data requests or contact Google for quota increase")

def manual_authentication_test():
    """Manual step-by-step authentication test"""
    print("\n🔧 Manual Authentication Test")
    print("=" * 40)
    
    try:
        import ee
        print("✅ earthengine-api imported successfully")
        
        # Try different initialization methods
        methods = [
            ("Default initialization", lambda: ee.Initialize()),
            ("High-volume endpoint", lambda: ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')),
            ("With project (if you have one)", lambda: ee.Initialize(project=None)),  # User can modify
        ]
        
        for method_name, init_func in methods:
            try:
                print(f"\n🧪 Testing: {method_name}")
                init_func()
                
                # Test basic operation
                result = ee.Number(42).getInfo()
                print(f"   ✅ Success! Test result: {result}")
                
                # Test image access
                image = ee.Image('USGS/SRTMGL1_003')
                projection = image.projection().getInfo()
                print(f"   ✅ Image access works! Projection: {projection['crs']}")
                
                return True
                
            except Exception as e:
                print(f"   ❌ Failed: {e}")
                continue
        
        print("\n❌ All authentication methods failed")
        return False
        
    except ImportError:
        print("❌ Cannot import earthengine-api")
        return False

def main():
    """Main authentication testing function"""
    
    print("🌍 Google Earth Engine Authentication Tester")
    print("=" * 60)
    print()
    
    # Check current status
    if get_authentication_status():
        print("🎉 Google Earth Engine is ready to use!")
        
        # Run a quick test
        try:
            import ee
            uzbekistan = ee.Geometry.Rectangle([55.9, 37.2, 73.2, 45.6])
            area = uzbekistan.area().divide(1e6).getInfo()
            print(f"✅ Test successful! Uzbekistan area: {area:.0f} km²")
        except Exception as e:
            print(f"⚠️ Basic test failed: {e}")
        
        return True
    
    # Try different authentication methods
    success = False
    
    # Method 1: Interactive
    if not success:
        success = test_authentication_method_1()
    
    # Method 2: Project-based
    if not success:
        success = test_authentication_method_2()
    
    # Method 3: Service account
    if not success:
        success = test_authentication_method_3()
    
    # Manual testing
    if not success:
        success = manual_authentication_test()
    
    # Provide guidance
    if not success:
        setup_authentication_guide()
        
        print("\n🔧 **Quick Fix Attempt:**")
        print("Try running these commands in sequence:")
        print()
        print("```bash")
        print("# Method 1: Command line authentication")
        print("earthengine authenticate")
        print()
        print("# Method 2: Application default credentials")
        print("gcloud auth application-default login")
        print()
        print("# Method 3: Test the setup")
        print("python -c \"import ee; ee.Initialize(); print('Success!')\"")
        print("```")
    
    return success

if __name__ == "__main__":
    main()
