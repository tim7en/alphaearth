#!/usr/bin/env python3
"""
Comprehensive Test for Enhanced Scientific Land Degradation Analysis

This script validates the enhanced land degradation analysis implementation
with Google Earth Engine integration and advanced scientific methods.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import os

def test_enhanced_degradation_analysis():
    """Test the enhanced scientific land degradation analysis"""
    
    print("🧪 Enhanced Scientific Land Degradation Analysis Testing")
    print("=" * 70)
    print("🎯 Testing comprehensive scientific implementation")
    print()
    
    validation_results = {}
    
    # Test 1: Enhanced module import and functionality
    print("🔬 Test 1: Enhanced module functionality...")
    try:
        # Change to project directory for relative imports
        project_root = Path(__file__).parent / 'alphaearth-uz'
        src_path = project_root / 'src'
        sys.path.insert(0, str(src_path))
        os.chdir(project_root)
        
        from aeuz import degradation
        
        # Test enhanced functions exist
        enhanced_functions = [
            'initialize_gee',
            'calculate_spectral_indices', 
            'load_enhanced_satellite_data',
            'calculate_advanced_degradation_indicators',
            'perform_advanced_trend_analysis',
            'perform_spatial_autocorrelation_analysis'
        ]
        
        functions_available = all(hasattr(degradation, func) for func in enhanced_functions)
        validation_results['enhanced_functions'] = functions_available
        print(f"   ✅ Enhanced functions available: {functions_available}")
        
    except Exception as e:
        validation_results['enhanced_functions'] = False
        print(f"   ❌ Enhanced module import failed: {e}")
    
    # Test 2: Google Earth Engine integration
    print("\n🛰️  Test 2: Google Earth Engine integration...")
    try:
        # Test GEE availability detection
        gee_module_available = hasattr(degradation, 'GEE_AVAILABLE')
        
        # Test GEE functions
        gee_functions = [
            'initialize_gee',
            'get_uzbekistan_boundaries',
            'calculate_spectral_indices'
        ]
        
        gee_functions_available = all(hasattr(degradation, func) for func in gee_functions)
        validation_results['gee_integration'] = gee_module_available and gee_functions_available
        print(f"   ✅ GEE integration implemented: {validation_results['gee_integration']}")
        
    except Exception as e:
        validation_results['gee_integration'] = False
        print(f"   ❌ GEE integration test failed: {e}")
    
    # Test 3: Advanced scientific methods
    print("\n📊 Test 3: Advanced scientific methods...")
    try:
        # Run the enhanced analysis
        results = degradation.run()
        
        # Check for scientific rigor indicators
        scientific_indicators = [
            results.get('status') == 'success',
            'spatial_autocorrelation_morans_i' in results.get('summary_stats', {}),
            'methodology' in results,
            len(results.get('methodology', {}).get('statistical_methods', [])) >= 4,
            'quality_assurance' in results.get('methodology', {}),
            results.get('analysis_type') == 'comprehensive_scientific_land_degradation'
        ]
        
        validation_results['scientific_methods'] = all(scientific_indicators)
        print(f"   ✅ Scientific methods implemented: {validation_results['scientific_methods']}")
        print(f"   📈 Statistical methods: {results.get('methodology', {}).get('statistical_methods', [])}")
        print(f"   🗺️  Spatial autocorrelation: {results.get('summary_stats', {}).get('spatial_autocorrelation_morans_i', 'N/A')}")
        
    except Exception as e:
        validation_results['scientific_methods'] = False
        print(f"   ❌ Scientific methods test failed: {e}")
    
    # Test 4: Output quality and completeness
    print("\n📁 Test 4: Output quality and completeness...")
    try:
        # Check for enhanced outputs
        expected_files = [
            "alphaearth-uz/tables/comprehensive_degradation_analysis.csv",
            "alphaearth-uz/figs/comprehensive_degradation_analysis_scientific.png"
        ]
        
        files_exist = all(Path(f).exists() for f in expected_files)
        
        # Check data quality
        if Path("alphaearth-uz/tables/comprehensive_degradation_analysis.csv").exists():
            df = pd.read_csv("alphaearth-uz/tables/comprehensive_degradation_analysis.csv")
            
            # Check for enhanced columns
            enhanced_columns = [
                'land_degradation_index',
                'vegetation_degradation',
                'soil_degradation', 
                'climate_stress',
                'pressure_index',
                'comprehensive_risk_score',
                'spatial_pattern',
                'local_morans_i'
            ]
            
            columns_available = all(col in df.columns for col in enhanced_columns)
            data_quality = len(df) > 200 and df.isnull().sum().sum() < len(df) * 0.1
            
            validation_results['output_quality'] = files_exist and columns_available and data_quality
            print(f"   ✅ Enhanced outputs generated: {files_exist}")
            print(f"   ✅ Enhanced columns available: {columns_available}")
            print(f"   ✅ Data quality acceptable: {data_quality}")
            print(f"   📊 Dataset size: {len(df)} records")
            
        else:
            validation_results['output_quality'] = False
            print("   ❌ Output files not found")
            
    except Exception as e:
        validation_results['output_quality'] = False
        print(f"   ❌ Output quality test failed: {e}")
    
    # Test 5: Scientific reporting features
    print("\n📋 Test 5: Scientific reporting features...")
    try:
        # Test methodology documentation
        methodology_complete = (
            'satellite_data_source' in results.get('methodology', {}) and
            'statistical_methods' in results.get('methodology', {}) and
            'degradation_indices' in results.get('methodology', {}) and
            'quality_assurance' in results.get('methodology', {})
        )
        
        # Test comprehensive summary stats
        summary_complete = (
            'total_areas_assessed' in results.get('summary_stats', {}) and
            'avg_land_degradation_index' in results.get('summary_stats', {}) and
            'spatial_autocorrelation_morans_i' in results.get('summary_stats', {}) and
            'data_quality_score' in results.get('summary_stats', {})
        )
        
        validation_results['scientific_reporting'] = methodology_complete and summary_complete
        print(f"   ✅ Methodology documentation: {methodology_complete}")
        print(f"   ✅ Comprehensive statistics: {summary_complete}")
        print(f"   📊 Quality score: {results.get('summary_stats', {}).get('data_quality_score', 'N/A')}%")
        
    except Exception as e:
        validation_results['scientific_reporting'] = False
        print(f"   ❌ Scientific reporting test failed: {e}")
    
    # Summary of validation
    print("\n" + "=" * 70)
    print("🎯 VALIDATION SUMMARY")
    print("=" * 70)
    
    total_tests = len(validation_results)
    passed_tests = sum(validation_results.values())
    
    for test_name, result in validation_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name.replace('_', ' ').title()}")
    
    print(f"\nOverall Result: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED! Enhanced scientific land degradation analysis is fully implemented.")
        return True
    else:
        print("⚠️  Some tests failed. Review implementation for missing features.")
        return False

if __name__ == "__main__":
    success = test_enhanced_degradation_analysis()
    sys.exit(0 if success else 1)