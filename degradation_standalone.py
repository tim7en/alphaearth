#!/usr/bin/env python3
"""
Standalone Land Degradation Analysis using AlphaEarth Satellite Embeddings

This script performs comprehensive land degradation assessment for Uzbekistan 
using real AlphaEarth satellite embeddings. It identifies degradation hotspots, 
analyzes trends, and prioritizes intervention areas.

Usage:
    python degradation_standalone.py
    
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
    """Run standalone land degradation analysis"""
    print("🏜️ AlphaEarth Land Degradation Analysis - Standalone")
    print("=" * 60)
    print()
    
    try:
        # Import and run degradation analysis
        from aeuz import degradation
        
        print("Starting land degradation assessment with real satellite data...")
        results = degradation.run()
        
        print("\n✅ Land Degradation Analysis Complete!")
        print(f"📊 Results summary: {results}")
        
        print("\n📁 Generated outputs:")
        print("  - Degradation maps: alphaearth-uz/figs/")
        print("  - Hotspot analysis: alphaearth-uz/tables/")
        print("  - Intervention priorities: alphaearth-uz/tables/")
        
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