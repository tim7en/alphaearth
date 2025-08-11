#!/usr/bin/env python3
"""
Google Earth Engine Authentication Handler

This script provides robust authentication handling for Google Earth Engine,
including manual code entry when the callback server fails.

Author: AlphaEarth Analysis Team
Date: August 11, 2025
"""

import os
import sys
import json
import webbrowser
from pathlib import Path

def manual_authenticate_gee():
    """Manual Google Earth Engine authentication with manual code entry"""
    
    print("🔐 Google Earth Engine Manual Authentication")
    print("=" * 50)
    
    try:
        import ee
        
        # Check if already authenticated
        try:
            ee.Initialize()
            print("✅ Already authenticated and ready!")
            return True
        except:
            pass
        
        print("\n📋 Manual Authentication Process:")
        print("1. We'll open a browser window for you to authenticate")
        print("2. After authorization, you'll be redirected to a localhost page")
        print("3. Copy the 'code' parameter from the URL")
        print("4. Paste it here when prompted")
        
        input("\nPress Enter to continue...")
        
        # Generate the authentication URL manually
        client_id = "517222506229-vsmmajv00ul0bs7p89v5m89qs8eb9359.apps.googleusercontent.com"
        scope = "https://www.googleapis.com/auth/earthengine https://www.googleapis.com/auth/cloud-platform https://www.googleapis.com/auth/drive https://www.googleapis.com/auth/devstorage.full_control"
        redirect_uri = "http://localhost:8085"
        
        auth_url = f"https://accounts.google.com/o/oauth2/auth?client_id={client_id}&scope={scope}&redirect_uri={redirect_uri}&response_type=code&access_type=offline"
        
        print(f"\n🌐 Opening authentication URL...")
        print(f"URL: {auth_url}")
        
        try:
            webbrowser.open(auth_url)
        except:
            print("Could not open browser automatically. Please copy the URL above.")
        
        print("\n⚠️  After you authorize in your browser:")
        print("1. You'll be redirected to a page like: http://localhost:8085/?code=XXXXX")
        print("2. Copy the long code after 'code=' from the URL")
        print("3. Paste it below")
        
        # Get the authorization code from user
        auth_code = input("\n📝 Paste the authorization code here: ").strip()
        
        if not auth_code:
            print("❌ No code provided")
            return False
        
        # Clean up the code (remove any URL parameters)
        if '&' in auth_code:
            auth_code = auth_code.split('&')[0]
        
        print(f"\n🔑 Using authorization code: {auth_code[:20]}...")
        
        # Create credentials directory if it doesn't exist
        config_dir = Path.home() / '.config' / 'earthengine'
        config_dir.mkdir(parents=True, exist_ok=True)
        
        # Try to complete the authentication using the code
        try:
            # We'll use ee.Authenticate() but need to handle the code manually
            print("🔄 Completing authentication...")
            
            # This is a workaround - we'll try to authenticate and let the user know the steps
            ee.Authenticate(code_verifier=None, authorization_code=auth_code)
            
        except Exception as e:
            print(f"⚠️  Standard authentication failed: {e}")
            print("\n🔧 Trying alternative method...")
            
            # Alternative: Try to authenticate normally and hope it works
            try:
                ee.Authenticate()
                print("✅ Authentication completed!")
            except Exception as e2:
                print(f"❌ Alternative method also failed: {e2}")
                return False
        
        # Test the authentication
        try:
            ee.Initialize()
            result = ee.Number(42).getInfo()
            print(f"✅ Authentication successful! Test result: {result}")
            return True
            
        except Exception as e:
            print(f"❌ Authentication test failed: {e}")
            return False
            
    except ImportError:
        print("❌ Google Earth Engine not available. Install with: pip install earthengine-api")
        return False

def service_account_auth():
    """Set up service account authentication"""
    
    print("\n🔧 Service Account Authentication Setup")
    print("=" * 40)
    
    print("For automated/production use, you can use a service account:")
    print("\n1. Go to Google Cloud Console")
    print("2. Create a new service account or use existing one")
    print("3. Download the JSON key file")
    print("4. Save it securely and set environment variable:")
    print("\n   export GOOGLE_APPLICATION_CREDENTIALS='/path/to/service-account-key.json'")
    
    # Check if service account is already set up
    creds_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    if creds_path and Path(creds_path).exists():
        print(f"\n✅ Service account credentials found: {creds_path}")
        
        try:
            import ee
            ee.Initialize()
            result = ee.Number(1).getInfo()
            print(f"✅ Service account authentication working! Test: {result}")
            return True
        except Exception as e:
            print(f"❌ Service account test failed: {e}")
            return False
    else:
        print("\n❌ No service account credentials found")
        return False

def gcloud_auth():
    """Try Google Cloud SDK authentication"""
    
    print("\n☁️  Google Cloud SDK Authentication")
    print("=" * 35)
    
    print("If you have gcloud installed, try:")
    print("  gcloud auth application-default login")
    
    try:
        import subprocess
        result = subprocess.run(['gcloud', 'auth', 'list'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ gcloud is installed")
            print("Active accounts:")
            print(result.stdout)
            
            # Try to set up application default credentials
            setup = input("\n🔧 Set up application default credentials? (y/n): ")
            if setup.lower() == 'y':
                subprocess.run(['gcloud', 'auth', 'application-default', 'login'])
                
                # Test it
                try:
                    import ee
                    ee.Initialize()
                    result = ee.Number(123).getInfo()
                    print(f"✅ gcloud authentication working! Test: {result}")
                    return True
                except Exception as e:
                    print(f"❌ gcloud auth test failed: {e}")
                    return False
        else:
            print("❌ gcloud not installed or not configured")
            return False
            
    except FileNotFoundError:
        print("❌ gcloud command not found")
        return False

def check_current_auth():
    """Check if Google Earth Engine is currently authenticated"""
    
    try:
        import ee
        ee.Initialize()
        
        # Test basic functionality
        test_result = ee.Number(2025).getInfo()
        print(f"✅ Google Earth Engine is authenticated and working!")
        print(f"   Test result: {test_result}")
        
        # Test image access
        image = ee.Image('USGS/SRTMGL1_003')
        scale = image.projection().nominalScale().getInfo()
        print(f"   Image access test: SRTM scale = {scale}m")
        
        return True
        
    except Exception as e:
        print(f"❌ Authentication check failed: {e}")
        return False

def main():
    """Main authentication setup"""
    
    print("🌍 Google Earth Engine Authentication Setup")
    print("=" * 50)
    
    # First check if already authenticated
    print("\n🔍 Checking current authentication status...")
    if check_current_auth():
        print("\n🎉 Google Earth Engine is ready to use!")
        return True
    
    print("\n🔧 Authentication required. Choose a method:")
    print("\n1. 📱 Manual browser authentication (Recommended)")
    print("2. 🔑 Service account authentication")
    print("3. ☁️  Google Cloud SDK authentication")
    print("4. 🚫 Skip authentication")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == '1':
        return manual_authenticate_gee()
    elif choice == '2':
        return service_account_auth()
    elif choice == '3':
        return gcloud_auth()
    elif choice == '4':
        print("⚠️  Skipping authentication - GEE analysis will not work")
        return False
    else:
        print("❌ Invalid choice")
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n✅ Authentication setup complete!")
        print("🚀 You can now run: python urban_heat_gee_standalone.py")
    else:
        print("\n❌ Authentication setup failed")
        print("📚 See GEE_URBAN_HEAT_GUIDE.md for detailed instructions")
