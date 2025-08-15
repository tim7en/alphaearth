#!/usr/bin/env python3
"""
Standalone Drought & Vegetation Anomaly Analysis using Mock GEE Data

This script performs comprehensive drought and vegetation anomaly analysis for 
Uzbekistan's agro-districts to answer: "Where and when have agro-districts 
experienced the deepest vegetation deficits since 2000? How did 2021 & 2023 compare?"

Analysis Components:
- Mock MOD13Q1 NDVI/EVI data (replacing real GEE data)
- Mock CHIRPS precipitation data
- NDVI/EVI z-scores vs 2001–2020 baseline
- SPI (Standardized Precipitation Index) calculation
- Pixel-wise Mann-Kendall trend analysis
- District drought atlas, anomaly time-series, hotspot maps

Usage:
    python drought_vegetation_standalone.py
    
Requirements:
    - Python 3.10+
    - Dependencies from requirements.txt
    - Mock GEE datasets (auto-generated)
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
    """Run standalone drought and vegetation anomaly analysis"""
    print("🌾 AlphaEarth Drought & Vegetation Anomaly Analysis - Standalone")
    print("=" * 70)
    print()
    print("Research Question: Where and when have agro-districts experienced")
    print("the deepest vegetation deficits since 2000? How did 2021 & 2023 compare?")
    print()
    
    try:
        # Import and run drought vegetation analysis
        from aeuz import drought_vegetation_analysis
        
        print("Starting drought & vegetation analysis with mock GEE data...")
        print("(Mock data simulates MOD13Q1 NDVI/EVI and CHIRPS precipitation)")
        print()
        
        results = drought_vegetation_analysis.run()
        
        print("\n✅ Drought & Vegetation Anomaly Analysis Complete!")
        print("\n🔍 Key Research Findings:")
        print(f"  📍 Most severely affected district: {results['most_affected_district']}")
        print(f"  📊 Severity score: {results['highest_severity_score']:.1f}")
        print(f"  📉 Districts with declining vegetation: {results['districts_with_decreasing_vegetation_trend']}")
        print(f"  📈 Percentage declining: {results['percent_districts_declining']}")
        print(f"  🔄 2023 vs 2021 comparison: {results['districts_worse_drought_2023_vs_2021']} districts worse in 2023")
        
        print("\n📁 Generated outputs:")
        print("  - Drought atlas: alphaearth-uz/figs/drought_atlas_comprehensive.png")
        print("  - Time series: alphaearth-uz/figs/drought_timeseries_analysis.png")
        print("  - Hotspot ranking: alphaearth-uz/tables/drought_hotspots_ranking.csv")
        print("  - 2021 vs 2023 comparison: alphaearth-uz/tables/drought_2021_vs_2023_comparison.csv")
        print("  - Trend analysis: alphaearth-uz/tables/vegetation_trend_analysis.csv")
        print("  - Summary report: alphaearth-uz/tables/drought_analysis_summary.json")
        
        print("\n🔬 Methodology Summary:")
        print("  • Data: Mock MOD13Q1 NDVI/EVI (250m, 16-day) + CHIRPS precipitation (5.5km, daily)")
        print("  • Period: 2000-2023 with 2001-2020 baseline for z-scores")
        print("  • Analysis: Z-scores, SPI (90-day), Mann-Kendall trends")
        print("  • Focus: Agricultural districts, growing season (Apr-Sep)")
        print("  • Comparison: 2021 vs 2023 drought conditions")
        
        print("\n📋 Production Features:")
        print("  ✓ Memory-efficient processing")
        print("  ✓ Mock data replaces GEE authentication requirements")
        print("  ✓ Realistic data patterns and drought signatures")
        print("  ✓ Comprehensive output products for policy/research")
        print("  ✓ Ready for real GEE data replacement")
        
        return results
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        print("\nTroubleshooting:")
        print("  1. Ensure you're in the correct directory")
        print("  2. Check that all dependencies are installed: pip install -r requirements.txt")
        print("  3. Verify Python version is 3.10+")
        return None

if __name__ == "__main__":
    results = main()
    if results:
        print(f"\n🎯 Analysis completed successfully!")
        print(f"   Visit the generated files to explore drought patterns in Uzbekistan's agro-districts.")
    else:
        sys.exit(1)