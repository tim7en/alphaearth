#!/usr/bin/env python3
"""
Modern Google Earth Engine Authentication

This script uses the updated authentication method that works with 
Google's current security requirements.

Author: AlphaEarth Analysis Team
Date: August 15, 2025
"""

import os
import sys
import json
from pathlib import Path

def authenticate_gee_modern():
    """Authenticate Google Earth Engine using modern method"""
    
    print("🔐 Modern Google Earth Engine Authentication")
    print("=" * 50)
    
    try:
        import ee
        print("✅ Earth Engine API imported successfully")
        
        # Check if already authenticated
        try:
            ee.Initialize()
            print("✅ Already authenticated and ready!")
            test_connection()
            return True
        except:
            print("🔄 Need to authenticate...")
        
        print("\n🌐 Starting authentication process...")
        print("This will open a browser window for authentication.")
        print("Please follow the instructions in the browser.")
        
        # Use the modern authentication method
        try:
            # This should open browser and handle the OAuth flow automatically
            ee.Authenticate()
            print("✅ Authentication completed!")
            
            # Test initialization
            ee.Initialize()
            print("✅ Earth Engine initialized successfully!")
            
            # Test connection
            return test_connection()
            
        except Exception as e:
            print(f"❌ Modern authentication failed: {e}")
            return try_alternative_methods()
            
    except ImportError:
        print("❌ Earth Engine API not available")
        print("Please install with: pip install earthengine-api")
        return False

def try_alternative_methods():
    """Try alternative authentication methods"""
    
    print("\n🔄 Trying alternative methods...")
    
    try:
        import ee
        
        # Method 1: Try with project specification
        print("1. Trying with project specification...")
        try:
            ee.Initialize(project='earthengine-legacy')
            print("✅ Project-based initialization successful!")
            return test_connection()
        except Exception as e:
            print(f"   Failed: {e}")
        
        # Method 2: Try service account detection
        print("2. Checking for service account credentials...")
        service_account_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
        if service_account_path and Path(service_account_path).exists():
            try:
                ee.Initialize()
                print("✅ Service account authentication successful!")
                return test_connection()
            except Exception as e:
                print(f"   Failed: {e}")
        
        # Method 3: Manual token-based authentication
        print("3. Trying manual authentication...")
        try:
            # This uses any existing credentials in the system
            ee.Initialize()
            print("✅ Manual authentication successful!")
            return test_connection()
        except Exception as e:
            print(f"   Failed: {e}")
        
        print("⚠️ All authentication methods failed")
        return False
        
    except Exception as e:
        print(f"❌ Alternative methods failed: {e}")
        return False

def test_connection():
    """Test Earth Engine connection"""
    
    print("\n🧪 Testing Google Earth Engine connection...")
    
    try:
        import ee
        
        # Test basic computation
        test_number = ee.Number(2025).getInfo()
        print(f"✅ Basic computation test: {test_number}")
        
        # Test image access
        image = ee.Image('USGS/SRTMGL1_003')
        scale = image.projection().nominalScale().getInfo()
        print(f"✅ Image access test successful! SRTM scale: {scale}m")
        
        # Test Uzbekistan-specific data
        uzbekistan = ee.Geometry.Rectangle([55.9, 37.2, 73.2, 45.6])
        modis_collection = ee.ImageCollection('MODIS/061/MOD11A1') \
            .filterDate('2023-01-01', '2023-01-31') \
            .filterBounds(uzbekistan)
        
        count = modis_collection.size().getInfo()
        print(f"✅ Uzbekistan data test: Found {count} MODIS images")
        
        return True
        
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False

def provide_manual_instructions():
    """Provide manual authentication instructions"""
    
    print("\n📋 Manual Authentication Instructions")
    print("=" * 40)
    
    print("\n**Option 1: Command Line Authentication (if earthengine CLI is available)**")
    print("  1. Install earthengine CLI: pip install earthengine-api[cli]")
    print("  2. Run: earthengine authenticate")
    print("  3. Follow browser instructions")
    
    print("\n**Option 2: Google Colab Style Authentication**")
    print("  1. Go to: https://code.earthengine.google.com/")
    print("  2. Sign in with your Google account")
    print("  3. Run a simple script to verify access")
    print("  4. Copy your authentication token")
    
    print("\n**Option 3: Service Account (for production use)**")
    print("  1. Go to Google Cloud Console")
    print("  2. Create a new project or select existing")
    print("  3. Enable Earth Engine API")
    print("  4. Create service account and download JSON key")
    print("  5. Set environment variable:")
    print("     $env:GOOGLE_APPLICATION_CREDENTIALS='path\\to\\key.json'")
    
    print("\n**Option 4: Use existing Google Cloud credentials**")
    print("  1. Install Google Cloud SDK")
    print("  2. Run: gcloud auth application-default login")
    print("  3. Follow browser instructions")

def save_auth_status(authenticated=False):
    """Save authentication status"""
    
    status_file = Path.cwd() / ".gee_auth_status.json"
    
    status = {
        "authenticated": authenticated,
        "timestamp": str(Path(__file__).stat().st_mtime),
        "method": "modern_auth" if authenticated else "simulation"
    }
    
    with open(status_file, 'w') as f:
        json.dump(status, f, indent=2)
    
    print(f"💾 Authentication status saved to: {status_file}")

def main():
    """Main authentication function"""
    
    success = authenticate_gee_modern()
    save_auth_status(success)
    
    if success:
        print("\n🎉 Google Earth Engine authentication successful!")
        print("✅ You can now run your analysis scripts with real data")
        print("\n🚀 Try running:")
        print("   python ghg_emissions_uzb/ghg_downscaling_uzb.py")
        print("   python urban_heat_gee_standalone.py")
    else:
        print("\n⚠️  Google Earth Engine authentication not complete")
        provide_manual_instructions()
        print("\n📝 Your analysis scripts will run in simulation mode")
        print("   This will demonstrate the workflow with synthetic data")
    
    return success

if __name__ == "__main__":
    main()
