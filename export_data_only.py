#!/usr/bin/env python3
"""
Data Export Only Script for Uzbekistan Urban Expansion Analysis
===============================================================
This script exports only the original satellite data and analysis results
without running the full visualization pipeline.
"""

import ee
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import the main analysis functions
from urban_expansion_impact_analysis import (
    UZBEKISTAN_CITIES, 
    authenticate_gee, 
    analyze_urban_expansion_impacts,
    calculate_expansion_impacts,
    export_original_data
)

def main():
    """Run data collection and export only"""
    print("💾 UZBEKISTAN URBAN EXPANSION DATA EXPORT")
    print("="*60)
    print("Collecting and exporting original satellite data...")
    print("="*60)
    
    try:
        # Initialize GEE
        if not authenticate_gee():
            return
        
        # Analyze urban expansion impacts (data collection only)
        print("\n📡 Phase 1: Collecting urban expansion data...")
        expansion_data = analyze_urban_expansion_impacts()
        
        if not expansion_data or len(expansion_data) < 1:
            print("❌ Insufficient expansion data collected. Exiting...")
            return
        
        # Calculate impacts 
        print("\n📊 Phase 2: Calculating expansion impacts...")
        impacts_df, regional_impacts = calculate_expansion_impacts(expansion_data)
        
        # Export all data
        print("\n💾 Phase 3: Exporting original data...")
        export_info = export_original_data(expansion_data, impacts_df, regional_impacts)
        
        print("\n" + "="*60)
        print("✅ Data Export Complete!")
        print(f"📁 Files: {export_info['files_generated']} generated")
        print(f"📊 Data points: {export_info['data_points']:,} satellite observations")
        print(f"💾 Location: {export_info['output_dir']}")
        print(f"🆔 Dataset ID: {export_info['timestamp']}")
        print(f"📋 Summary: {export_info['summary_file']}")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error in data export: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
