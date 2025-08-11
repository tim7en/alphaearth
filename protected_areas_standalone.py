#!/usr/bin/env python3
"""
Standalone Protected Areas Monitoring using AlphaEarth Satellite Embeddings

This script performs comprehensive protected area monitoring for Uzbekistan 
using real AlphaEarth satellite embeddings. It detects conservation status, 
identifies incidents, and assesses management effectiveness.

Usage:
    python protected_areas_standalone.py
    
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
    """Run standalone protected areas monitoring analysis"""
    print("🏛️ AlphaEarth Protected Areas Monitoring - Standalone")
    print("=" * 60)
    print()
    
    try:
        # Import and run protected areas analysis
        from aeuz import protected_areas
        
        print("Starting protected areas monitoring with real satellite data...")
        results = protected_areas.run()
        
        print("\n✅ Protected Areas Analysis Complete!")
        print(f"📊 Results summary: {results}")
        
        print("\n📁 Generated outputs:")
        print("  - Conservation maps: alphaearth-uz/figs/")
        print("  - Incident reports: alphaearth-uz/data_final/protected_area_incidents.geojson")
        print("  - Management assessment: alphaearth-uz/tables/")
        
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