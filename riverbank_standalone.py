#!/usr/bin/env python3
"""
Standalone Riverbank Disturbance Analysis using AlphaEarth Satellite Embeddings

This script performs comprehensive riverbank disturbance monitoring for Uzbekistan 
using real AlphaEarth satellite embeddings. It detects buffer zone changes, 
identifies disturbance patterns, and flags priority areas.

Usage:
    python riverbank_standalone.py
    
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
    """Run standalone riverbank disturbance analysis"""
    print("🌊 AlphaEarth Riverbank Disturbance Analysis - Standalone")
    print("=" * 60)
    print()
    
    try:
        # Import and run riverbank analysis
        from aeuz import riverbank
        
        print("Starting riverbank disturbance assessment with real satellite data...")
        results = riverbank.run()
        
        print("\n✅ Riverbank Disturbance Analysis Complete!")
        print(f"📊 Results summary: {results}")
        
        print("\n📁 Generated outputs:")
        print("  - Disturbance maps: alphaearth-uz/figs/")
        print("  - Flagged areas: alphaearth-uz/data_final/riverbank_flags.geojson")
        print("  - Analysis tables: alphaearth-uz/tables/")
        
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