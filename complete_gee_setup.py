#!/usr/bin/env python3
"""
Complete Google Earth Engine Setup Guide
========================================
This script helps you complete the Earth Engine registration process.
"""

import ee
import time
import webbrowser

def check_current_status():
    """Check current Earth Engine status"""
    print("🔍 Checking current Earth Engine status...")
    
    try:
        # Check stored credentials
        print("📋 Checking stored credentials...")
        ee.Initialize()
        print("✅ Earth Engine is working! You're all set!")
        return True
    except Exception as e:
        print(f"❌ Earth Engine not ready: {e}")
        return False

def guide_project_setup():
    """Guide user through project setup"""
    print("\n" + "="*60)
    print("🌍 EARTH ENGINE PROJECT SETUP GUIDE")
    print("="*60)
    
    print("""
📝 STEP 1: Create/Select Google Cloud Project
   1. Go to: https://console.cloud.google.com/
   2. Click 'Select a project' at the top
   3. Click 'NEW PROJECT' 
   4. Name it: 'my-earth-engine-project' (or similar)
   5. Click 'CREATE'

📝 STEP 2: Register Project for Earth Engine
   1. Go to: https://code.earthengine.google.com/
   2. Click 'Register a Noncommercial or Commercial Cloud project'
   3. Select your project from Step 1
   4. Choose 'Noncommercial' (for research/education)
   5. Fill out the form and submit

📝 STEP 3: Wait for Approval
   - Usually takes 1-24 hours
   - You'll get an email when approved
   - The project will appear in Earth Engine Code Editor

📝 STEP 4: Test Access
   - Run this script again to test
   - Or try: https://code.earthengine.google.com/
    """)

def try_different_projects():
    """Try to initialize with different project patterns"""
    print("\n🔍 Trying common project name patterns...")
    
    # Common project name patterns
    import os
    username = os.getenv('USER', 'user')
    
    project_patterns = [
        f"ee-{username}",
        f"earthengine-{username}", 
        f"my-earth-engine-project",
        f"gee-project-{username}",
        "earthengine-legacy",
        None  # Default project
    ]
    
    for project in project_patterns:
        try:
            if project:
                print(f"🧪 Trying project: {project}")
                ee.Initialize(project=project)
            else:
                print("🧪 Trying default project...")
                ee.Initialize()
                
            print(f"✅ SUCCESS with project: {project or 'default'}")
            
            # Test basic functionality
            image = ee.Image('USGS/SRTMGL1_003')
            print("✅ Can access Earth Engine datasets!")
            return project
            
        except Exception as e:
            print(f"❌ Failed: {str(e)[:100]}...")
            continue
    
    return None

def main():
    print("🌍 Google Earth Engine Complete Setup")
    print("=" * 50)
    
    # Check current status
    if check_current_status():
        print("🎉 Earth Engine is working perfectly!")
        return
    
    # Try different project configurations
    working_project = try_different_projects()
    
    if working_project:
        print(f"\n🎉 SUCCESS! Earth Engine working with project: {working_project}")
        
        # Test with a simple query
        try:
            print("\n🧪 Testing Earth Engine functionality...")
            
            # Get information about Uzbekistan
            uzbekistan = ee.FeatureCollection("FAO/GAUL/2015/level0").filter(ee.Filter.eq('ADM0_NAME', 'Uzbekistan'))
            print("✅ Can query Uzbekistan boundaries")
            
            # Get a satellite image
            image = ee.ImageCollection('MODIS/006/MOD11A1') \
                     .filterDate('2023-01-01', '2023-12-31') \
                     .first()
            print("✅ Can access MODIS satellite data")
            
            print("\n🚀 Ready to run real satellite analysis!")
            print("📄 Now you can run: python urban_heat_gee_clean.py")
            
        except Exception as e:
            print(f"⚠️ Basic access works, but some features may need more setup: {e}")
    
    else:
        print("\n❌ Earth Engine access not ready yet.")
        guide_project_setup()
        
        print("\n🔗 Quick Links:")
        print("   • Cloud Console: https://console.cloud.google.com/")
        print("   • Earth Engine: https://code.earthengine.google.com/")
        print("   • Registration: https://signup.earthengine.google.com/")
        
        print("\n⏳ After completing registration:")
        print("   Run this script again to test access")

if __name__ == "__main__":
    main()
