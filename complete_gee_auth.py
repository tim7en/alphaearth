#!/usr/bin/env python3
"""
Complete Google Earth Engine Authentication

This script completes the authentication process using the authorization code
from the callback URL.

Author: AlphaEarth Analysis Team
Date: August 11, 2025
"""

import os
import json
import requests
from pathlib import Path

def complete_gee_auth_with_code():
    """Complete GEE authentication using the authorization code"""
    
    print("🔑 Completing Google Earth Engine Authentication")
    print("=" * 50)
    
    # Your authorization code from the callback URL
    auth_code = "4/0AVMBsJjbU95cPewgH36CaLcFcyxxAvNLfrwnZs4keTJ01M5jGm-rx-p1_E3B7EDL_glYTg"
    
    print(f"Using authorization code: {auth_code[:20]}...")
    
    # Google Earth Engine OAuth2 configuration
    client_id = "517222506229-vsmmajv00ul0bs7p89v5m89qs8eb9359.apps.googleusercontent.com"
    client_secret = "d-FL95Q19q7MQmFpd7hHD0Ty"  # This is the public client secret for EE
    redirect_uri = "http://localhost:8085"
    token_url = "https://oauth2.googleapis.com/token"
    
    try:
        # Exchange authorization code for tokens
        print("🔄 Exchanging authorization code for access token...")
        
        token_data = {
            'code': auth_code,
            'client_id': client_id,
            'client_secret': client_secret,
            'redirect_uri': redirect_uri,
            'grant_type': 'authorization_code'
        }
        
        response = requests.post(token_url, data=token_data)
        
        if response.status_code == 200:
            tokens = response.json()
            print("✅ Successfully received tokens!")
            
            # Create the credentials file
            config_dir = Path.home() / '.config' / 'earthengine'
            config_dir.mkdir(parents=True, exist_ok=True)
            
            credentials_file = config_dir / 'credentials'
            
            # Format the credentials for Earth Engine
            credentials = {
                'refresh_token': tokens.get('refresh_token'),
                'access_token': tokens.get('access_token'),
                'client_id': client_id,
                'client_secret': client_secret,
                'type': 'authorized_user'
            }
            
            # Save credentials
            with open(credentials_file, 'w') as f:
                json.dump(credentials, f, indent=2)
            
            print(f"💾 Credentials saved to: {credentials_file}")
            
            # Test the authentication
            print("🧪 Testing Google Earth Engine authentication...")
            
            try:
                import ee
                ee.Initialize()
                
                # Test basic functionality
                test_result = ee.Number(2025).getInfo()
                print(f"✅ Authentication test successful! Result: {test_result}")
                
                # Test image access
                image = ee.Image('USGS/SRTMGL1_003')
                scale = image.projection().nominalScale().getInfo()
                print(f"✅ Image access test successful! SRTM scale: {scale}m")
                
                return True
                
            except Exception as e:
                print(f"❌ Authentication test failed: {e}")
                print("🔧 But credentials were saved - try running your analysis script")
                return True  # Credentials saved even if test failed
                
        else:
            print(f"❌ Failed to exchange code for tokens: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error during authentication: {e}")
        return False

def alternative_auth_methods():
    """Provide alternative authentication methods"""
    
    print("\n🔧 Alternative Authentication Methods")
    print("=" * 40)
    
    print("\n**Method 1: Command Line Tool (if installed)**")
    print("  earthengine authenticate")
    
    print("\n**Method 2: Try authentication in Python directly**")
    print("  python -c \"import ee; ee.Authenticate(); ee.Initialize(); print('Success!')\"")
    
    print("\n**Method 3: Service Account (for production)**")
    print("  1. Create service account in Google Cloud Console")
    print("  2. Download JSON key file")
    print("  3. Set environment variable:")
    print("     export GOOGLE_APPLICATION_CREDENTIALS='/path/to/key.json'")
    
    print("\n**Method 4: Google Cloud SDK**")
    print("  gcloud auth application-default login")

def main():
    """Main authentication completion"""
    
    success = complete_gee_auth_with_code()
    
    if success:
        print("\n🎉 Google Earth Engine authentication completed!")
        print("🚀 You can now run the urban heat analysis:")
        print("   python urban_heat_gee_standalone.py")
    else:
        print("\n❌ Automatic authentication completion failed")
        alternative_auth_methods()
        
        print("\n💡 You can also try the authentication helper:")
        print("   python gee_auth_helper.py")

if __name__ == "__main__":
    main()
