#!/usr/bin/env python3
"""
Test script for Comprehensive Agricultural Soil Moisture Analysis

This script validates the agricultural soil moisture analysis results
and ensures production readiness.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

def test_agricultural_analysis():
    """Test the agricultural soil moisture analysis components"""
    
    print("🧪 Testing Agricultural Soil Moisture Analysis Components")
    print("=" * 60)
    
    # Test 1: Check if output files exist
    print("📁 Test 1: Checking output files...")
    
    expected_files = [
        "data_final/agricultural_soil_moisture_comprehensive.csv",
        "reports/agricultural_soil_moisture_comprehensive_report.md",
        "tables/agricultural_regional_summary.csv",
        "tables/irrigation_system_analysis.csv", 
        "tables/critical_water_stressed_areas.csv",
        "figs/agricultural_irrigation_analysis.png",
        "figs/agricultural_spatial_analysis.png"
    ]
    
    missing_files = []
    for file_path in expected_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All expected output files present")
    
    # Test 2: Validate data quality
    print("\n📊 Test 2: Validating data quality...")
    
    try:
        df = pd.read_csv("data_final/agricultural_soil_moisture_comprehensive.csv")
        
        # Check basic data properties
        assert len(df) > 0, "Dataset is empty"
        assert df['soil_moisture_volumetric'].between(0, 1).all(), "Soil moisture values out of range"
        assert df['irrigation_efficiency'].between(0, 1).all(), "Irrigation efficiency values out of range"
        assert df['comprehensive_water_stress'].between(0, 1).all(), "Water stress values out of range"
        
        # Check for required columns
        required_cols = [
            'region', 'latitude', 'longitude', 'primary_crop', 'irrigation_system',
            'soil_moisture_volumetric', 'irrigation_efficiency', 'estimated_crop_yield',
            'comprehensive_water_stress', 'predicted_soil_moisture'
        ]
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        assert len(missing_cols) == 0, f"Missing required columns: {missing_cols}"
        
        print(f"✅ Data validation passed ({len(df)} samples)")
        
    except Exception as e:
        print(f"❌ Data validation failed: {e}")
        return False
    
    # Test 3: Check agricultural focus
    print("\n🌾 Test 3: Validating agricultural focus...")
    
    try:
        # Check crop types
        crop_types = df['primary_crop'].unique()
        expected_crops = ['cotton', 'wheat', 'rice', 'vegetables', 'fruits']
        has_agricultural_crops = any(crop in crop_types for crop in expected_crops)
        assert has_agricultural_crops, "No major agricultural crops found"
        
        # Check irrigation systems
        irrigation_systems = df['irrigation_system'].unique()
        expected_systems = ['drip', 'furrow', 'sprinkler']
        has_irrigation_systems = any(system in irrigation_systems for system in expected_systems)
        assert has_irrigation_systems, "No major irrigation systems found"
        
        # Check agricultural variables
        agricultural_vars = [
            'crop_water_requirement', 'irrigation_water_applied', 'water_use_efficiency',
            'crop_yield_potential', 'estimated_crop_yield', 'biomass_productivity'
        ]
        
        missing_ag_vars = [var for var in agricultural_vars if var not in df.columns]
        assert len(missing_ag_vars) == 0, f"Missing agricultural variables: {missing_ag_vars}"
        
        print(f"✅ Agricultural focus validated")
        print(f"   - Crop types: {', '.join(crop_types)}")
        print(f"   - Irrigation systems: {', '.join(irrigation_systems)}")
        
    except Exception as e:
        print(f"❌ Agricultural focus validation failed: {e}")
        return False
    
    # Test 4: Check analysis results
    print("\n📈 Test 4: Validating analysis results...")
    
    try:
        # Check correlations
        moisture_yield_corr = df['soil_moisture_volumetric'].corr(df['estimated_crop_yield'])
        assert abs(moisture_yield_corr) > 0.1, "Weak moisture-yield correlation"
        
        # Check model predictions
        if 'predicted_soil_moisture' in df.columns:
            prediction_corr = df['soil_moisture_volumetric'].corr(df['predicted_soil_moisture'])
            assert prediction_corr > 0.8, "Poor model predictions"
        
        # Check regional variation
        regional_stats = df.groupby('region')['soil_moisture_volumetric'].mean()
        assert regional_stats.std() > 0.01, "Insufficient regional variation"
        
        print(f"✅ Analysis results validated")
        print(f"   - Moisture-yield correlation: {moisture_yield_corr:.3f}")
        if 'predicted_soil_moisture' in df.columns:
            print(f"   - Model prediction accuracy: {prediction_corr:.3f}")
        
    except Exception as e:
        print(f"❌ Analysis results validation failed: {e}")
        return False
    
    # Test 5: Check report quality
    print("\n📄 Test 5: Validating report quality...")
    
    try:
        report_path = "reports/agricultural_soil_moisture_comprehensive_report.md"
        with open(report_path, 'r') as f:
            report_content = f.read()
        
        # Check for key sections
        required_sections = [
            "Executive Summary", "Key Findings", "Implementation Roadmap",
            "Agricultural", "Irrigation", "Water"
        ]
        
        missing_sections = [section for section in required_sections 
                          if section.lower() not in report_content.lower()]
        
        assert len(missing_sections) == 0, f"Missing report sections: {missing_sections}"
        assert len(report_content) > 1000, "Report too short"
        
        print(f"✅ Report quality validated ({len(report_content)} characters)")
        
    except Exception as e:
        print(f"❌ Report validation failed: {e}")
        return False
    
    print("\n🎯 All Tests Passed! Agricultural Analysis Ready for Production")
    return True


def generate_production_summary():
    """Generate production deployment summary"""
    
    print("\n🚀 Production Deployment Summary")
    print("=" * 50)
    
    try:
        df = pd.read_csv("data_final/agricultural_soil_moisture_comprehensive.csv")
        
        # Key statistics
        total_samples = len(df)
        regions = df['region'].nunique()
        crop_types = df['primary_crop'].nunique()
        irrigation_systems = df['irrigation_system'].nunique()
        
        avg_moisture = df['soil_moisture_volumetric'].mean()
        avg_efficiency = df['irrigation_efficiency'].mean()
        high_stress_areas = len(df[df['comprehensive_water_stress'] > 0.6])
        
        print(f"📊 Dataset Coverage:")
        print(f"   • Total samples: {total_samples:,}")
        print(f"   • Regions covered: {regions}")
        print(f"   • Crop types: {crop_types}")
        print(f"   • Irrigation systems: {irrigation_systems}")
        
        print(f"\n🌾 Agricultural Insights:")
        print(f"   • Average soil moisture: {avg_moisture:.1%}")
        print(f"   • Average irrigation efficiency: {avg_efficiency:.1%}")
        print(f"   • High water stress areas: {high_stress_areas}")
        
        print(f"\n📁 Generated Outputs:")
        print(f"   • Enhanced dataset: data_final/agricultural_soil_moisture_comprehensive.csv")
        print(f"   • Comprehensive report: reports/agricultural_soil_moisture_comprehensive_report.md")
        print(f"   • Regional analysis: tables/agricultural_regional_summary.csv")
        print(f"   • Irrigation analysis: tables/irrigation_system_analysis.csv")
        print(f"   • Critical areas: tables/critical_water_stressed_areas.csv")
        print(f"   • Visualizations: figs/agricultural_*.png")
        
        print(f"\n✅ Status: READY FOR PRODUCTION DEPLOYMENT")
        print(f"   ✓ Agricultural focus implemented")
        print(f"   ✓ Irrigation efficiency analysis complete")
        print(f"   ✓ Water stress identification done")
        print(f"   ✓ Crop yield correlations established")
        print(f"   ✓ Machine learning models validated")
        print(f"   ✓ Comprehensive reporting generated")
        
    except Exception as e:
        print(f"❌ Error generating summary: {e}")


if __name__ == "__main__":
    print("🧪 Agricultural Soil Moisture Analysis - Test & Validation")
    print("🎯 Validating production readiness for Uzbekistan analysis")
    print()
    
    # Run tests
    test_passed = test_agricultural_analysis()
    
    if test_passed:
        # Generate production summary
        generate_production_summary()
        
        print(f"\n🏆 SUCCESS: Agricultural soil moisture analysis is production-ready!")
        print(f"🌾 Focus on agricultural heartlands: ✅")
        print(f"🚿 Irrigation efficiency evaluation: ✅") 
        print(f"💧 Water-stressed areas identification: ✅")
        print(f"📈 Crop yield correlations: ✅")
        print(f"🤖 Volumetric water content models: ✅")
        print(f"📊 Comprehensive reporting: ✅")
        
        sys.exit(0)
    else:
        print(f"\n❌ FAILED: Analysis not ready for production")
        sys.exit(1)