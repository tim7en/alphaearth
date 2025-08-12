#!/usr/bin/env python3
"""
Test script to verify all modules import correctly
"""

import sys
import traceback

def test_imports():
    """Test all critical imports"""
    
    print("🧪 Testing module imports...")
    
    tests = [
        ("contextily", "import contextily as ctx"),
        ("geopandas", "import geopandas as gpd"),
        ("folium", "import folium"),
        ("soil_moisture_enhanced", "sys.path.append('alphaearth-uz/src'); from aeuz.soil_moisture_enhanced import load_agricultural_enhanced_data"),
        ("utils", "sys.path.append('alphaearth-uz/src'); from aeuz.utils import load_config"),
        ("matplotlib", "import matplotlib.pyplot as plt"),
        ("seaborn", "import seaborn as sns"),
        ("pandas", "import pandas as pd"),
        ("numpy", "import numpy as np"),
        ("sklearn", "from sklearn.ensemble import RandomForestRegressor"),
    ]
    
    passed = 0
    failed = 0
    
    for name, import_statement in tests:
        try:
            exec(import_statement)
            print(f"   ✅ {name}")
            passed += 1
        except Exception as e:
            print(f"   ❌ {name}: {str(e)}")
            failed += 1
    
    print(f"\n📊 Test Results:")
    print(f"   ✅ Passed: {passed}")
    print(f"   ❌ Failed: {failed}")
    print(f"   📈 Success Rate: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        print("\n🎉 All imports working correctly!")
    else:
        print(f"\n⚠️ {failed} import(s) need attention")
    
    return failed == 0

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
