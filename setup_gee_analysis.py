#!/usr/bin/env python3
"""
Setup script for Google Earth Engine Urban Heat Analysis

This script helps set up the environment for GEE-based urban heat analysis.
It handles package installation, authentication, and initial testing.

Usage:
    python setup_gee_analysis.py

Author: AlphaEarth Analysis Team
Date: August 11, 2025
"""

import subprocess
import sys
import os
from pathlib import Path

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def check_gee_auth():
    """Check if Google Earth Engine is authenticated"""
    try:
        import ee
        ee.Initialize()
        print("✅ Google Earth Engine is authenticated and ready")
        return True
    except Exception as e:
        print(f"❌ GEE Authentication issue: {e}")
        return False

def main():
    """Main setup function"""
    
    print("🌍 Setting up Google Earth Engine Urban Heat Analysis")
    print("=" * 60)
    
    # 1. Install required packages
    print("\n📦 Installing required packages...")
    
    required_packages = [
        "earthengine-api>=0.1.384",
        "geopandas>=0.14.0", 
        "rasterio>=1.3.9",
        "pyproj>=3.6.1"
    ]
    
    for package in required_packages:
        print(f"  Installing {package}...")
        if install_package(package):
            print(f"    ✅ {package} installed successfully")
        else:
            print(f"    ❌ Failed to install {package}")
    
    # 2. Try to import and test GEE
    print("\n🔑 Testing Google Earth Engine...")
    
    try:
        import ee
        print("  ✅ earthengine-api imported successfully")
        
        # Try to initialize
        if check_gee_auth():
            print("  ✅ GEE authentication successful")
            
            # Test basic functionality
            print("  🧪 Testing basic GEE functionality...")
            try:
                # Simple test: get image count
                uzbekistan = ee.Geometry.Rectangle([55.9, 37.2, 73.2, 45.6])
                modis_count = ee.ImageCollection('MODIS/061/MOD11A1') \
                    .filterDate('2023-01-01', '2023-01-31') \
                    .filterBounds(uzbekistan) \
                    .size().getInfo()
                
                print(f"    ✅ Found {modis_count} MODIS images for test query")
                print("  🎉 GEE setup complete and functional!")
                
            except Exception as e:
                print(f"    ⚠️ GEE test query failed: {e}")
                
        else:
            print("\n🔐 Google Earth Engine Authentication Required")
            print("  Please run the following command to authenticate:")
            print("  earthengine authenticate")
            print("\n  Or use service account authentication for automated workflows")
            print("  See: https://developers.google.com/earth-engine/guides/auth")
            
    except ImportError:
        print("  ❌ earthengine-api not available")
        print("  Please install with: pip install earthengine-api")
    
    # 3. Check workspace structure
    print("\n📁 Checking workspace structure...")
    
    expected_files = [
        "urban_heat_gee_standalone.py",
        "requirements_gee.txt",
        "utils.py"
    ]
    
    for file in expected_files:
        if Path(file).exists():
            print(f"  ✅ {file} found")
        else:
            print(f"  ❌ {file} missing")
    
    # 4. Create output directory
    output_dir = Path("gee_urban_heat_analysis")
    output_dir.mkdir(exist_ok=True)
    print(f"  ✅ Output directory created: {output_dir}")
    
    print("\n🚀 Setup Summary:")
    print("  1. Install packages: pip install -r requirements_gee.txt")
    print("  2. Authenticate GEE: earthengine authenticate")
    print("  3. Run analysis: python urban_heat_gee_standalone.py")
    print("\n  📚 Documentation: https://developers.google.com/earth-engine")

if __name__ == "__main__":
    main()
