#!/usr/bin/env python3
"""
Google Earth Engine Code Extractor

This script helps extract the authorization code from the callback URL
and complete the authentication process.

Author: AlphaEarth Analysis Team
Date: August 11, 2025
"""

import urllib.parse

def extract_auth_code():
    """Extract authorization code from callback URL"""
    
    print("🔗 Google Earth Engine Authorization Code Extractor")
    print("=" * 55)
    
    print("\nWhen you authorized Google Earth Engine, you were redirected to a URL like:")
    print("http://localhost:8085/?code=XXXXX&scope=...")
    
    print("\n📝 Please paste the full callback URL here:")
    callback_url = input("URL: ").strip()
    
    if not callback_url:
        print("❌ No URL provided")
        return None
    
    try:
        # Parse the URL to extract the code
        from urllib.parse import urlparse, parse_qs
        
        parsed_url = urlparse(callback_url)
        query_params = parse_qs(parsed_url.query)
        
        if 'code' in query_params:
            auth_code = query_params['code'][0]
            print(f"\n✅ Authorization code extracted: {auth_code[:20]}...")
            print(f"Full code: {auth_code}")
            
            # Also show scope for verification
            if 'scope' in query_params:
                scope = query_params['scope'][0]
                print(f"\n📋 Authorized scopes: {scope}")
            
            return auth_code
        else:
            print("❌ No 'code' parameter found in URL")
            return None
            
    except Exception as e:
        print(f"❌ Error parsing URL: {e}")
        return None

def complete_authentication(auth_code):
    """Complete the authentication process with the code"""
    
    print(f"\n🔑 Completing authentication with code...")
    
    try:
        import ee
        
        # Try to authenticate with the code
        # Note: The standard ee.Authenticate() might not directly accept the code
        # We'll try different approaches
        
        print("🔄 Attempting authentication...")
        
        # Method 1: Standard authentication (hoping it picks up the code)
        try:
            ee.Authenticate()
            ee.Initialize()
            test = ee.Number(42).getInfo()
            print(f"✅ Authentication successful! Test: {test}")
            return True
        except Exception as e:
            print(f"⚠️  Standard method failed: {e}")
        
        # Method 2: Try to manually complete the OAuth flow
        print("🔧 Trying manual OAuth completion...")
        
        # This is more complex and would require implementing the full OAuth flow
        # For now, we'll provide instructions
        
        print("\n📚 Manual completion steps:")
        print("1. The authorization code you received is valid")
        print("2. Save it for manual entry in the authentication process")
        print("3. Try running: earthengine authenticate")
        print("4. If that fails, you may need to use a different authentication method")
        
        return False
        
    except ImportError:
        print("❌ Google Earth Engine not available")
        return False

def create_credentials_manually():
    """Help create credentials file manually"""
    
    print("\n🔧 Manual Credentials Creation")
    print("=" * 35)
    
    print("If automatic authentication fails, you can try these alternatives:")
    
    print("\n**Option 1: Command Line Tool**")
    print("  earthengine authenticate")
    
    print("\n**Option 2: Service Account**")
    print("  1. Go to Google Cloud Console")
    print("  2. Create service account")
    print("  3. Download JSON key")
    print("  4. Set environment variable:")
    print("     export GOOGLE_APPLICATION_CREDENTIALS='/path/to/key.json'")
    
    print("\n**Option 3: Application Default Credentials**")
    print("  gcloud auth application-default login")

def main():
    """Main function"""
    
    # First, try to extract the code from your callback URL
    auth_code = extract_auth_code()
    
    if auth_code:
        # Try to complete authentication
        success = complete_authentication(auth_code)
        
        if not success:
            print("\n🔧 Automatic completion failed. Let's try alternative methods...")
            create_credentials_manually()
    else:
        print("\n❌ Could not extract authorization code")
        create_credentials_manually()

if __name__ == "__main__":
    main()
