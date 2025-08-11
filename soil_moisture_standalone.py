#!/usr/bin/env python3
"""
Standalone Soil Moisture Analysis using AlphaEarth Satellite Embeddings

This script performs comprehensive soil moisture analysis for Uzbekistan using
real AlphaEarth satellite embeddings. It can be run independently to generate
soil moisture maps, stress hotspots, and model predictions.

Usage:
    python soil_moisture_standalone.py
    
Requirements:
    - AlphaEarth Satellite Embedding V1 dataset
    - Python 3.10+
    - Dependencies from requirements.txt
"""

import sys
import os
from pathlib import Path

# Add alphaearth-uz source to path
project_root = Path(__file__).parent / 'alphaearth-uz'
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))

# Change to project directory for relative paths
os.chdir(project_root)

def main():
    """Run standalone soil moisture analysis"""
    print("🌊 AlphaEarth Soil Moisture Analysis - Standalone")
    print("=" * 60)
    print()
    
    try:
        # Import and run soil moisture analysis
        from aeuz import soil_moisture
        
        print("Starting soil moisture analysis with real satellite data...")
        results = soil_moisture.run()
        
        print("\n✅ Soil Moisture Analysis Complete!")
        print(f"📊 Results summary: {results}")
        
        print("\n📁 Generated outputs:")
        print("  - Soil moisture maps: alphaearth-uz/figs/")
        print("  - Analysis tables: alphaearth-uz/tables/")
        print("  - Summary reports: alphaearth-uz/reports/")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure you're running from the correct directory with required dependencies.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Analysis error: {e}")
        print("Please check AlphaEarth satellite data availability and format.")
        sys.exit(1)

if __name__ == "__main__":
    main()