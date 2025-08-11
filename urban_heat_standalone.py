#!/usr/bin/env python3
"""
Standalone Urban Heat Analysis using AlphaEarth Satellite Embeddings

This script performs comprehensive urban heat island analysis for Uzbekistan 
using real AlphaEarth satellite embeddings. It models heat patterns, 
identifies mitigation strategies, and assesses temperature risks.

Usage:
    python urban_heat_standalone.py
    
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
    """Run standalone urban heat analysis"""
    print("🌡️ AlphaEarth Urban Heat Analysis - Standalone")
    print("=" * 60)
    print()
    
    try:
        # Import and run urban heat analysis
        from aeuz import urban_heat
        
        print("Starting urban heat assessment with real satellite data...")
        results = urban_heat.run()
        
        print("\n✅ Urban Heat Analysis Complete!")
        print(f"📊 Results summary: {results}")
        
        print("\n📁 Generated outputs:")
        print("  - Temperature maps: alphaearth-uz/figs/")
        print("  - Heat risk assessment: alphaearth-uz/tables/")
        print("  - Mitigation strategies: alphaearth-uz/tables/")
        
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