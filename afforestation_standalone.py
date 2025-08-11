#!/usr/bin/env python3
"""
Standalone Afforestation Analysis using AlphaEarth Satellite Embeddings

This script performs comprehensive afforestation suitability analysis for Uzbekistan 
using real AlphaEarth satellite embeddings. It generates site suitability maps, 
species recommendations, and candidate locations for reforestation.

Usage:
    python afforestation_standalone.py
    
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
    """Run standalone afforestation analysis"""
    print("🌳 AlphaEarth Afforestation Analysis - Standalone")
    print("=" * 60)
    print()
    
    try:
        # Import and run afforestation analysis
        from aeuz import afforestation
        
        print("Starting afforestation suitability analysis with real satellite data...")
        results = afforestation.run()
        
        print("\n✅ Afforestation Analysis Complete!")
        print(f"📊 Results summary: {results}")
        
        print("\n📁 Generated outputs:")
        print("  - Suitability maps: alphaearth-uz/figs/")
        print("  - Candidate sites: alphaearth-uz/data_final/afforestation_candidates.geojson")
        print("  - Species recommendations: alphaearth-uz/tables/")
        
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