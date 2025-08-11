from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from .utils import (load_config, ensure_dir, setup_plotting, load_alphaearth_embeddings,
                   calculate_confidence_interval, perform_trend_analysis, create_summary_statistics,
                   save_plot, validate_data_quality, perform_cross_validation, 
                   enhance_model_with_feature_selection, create_pilot_study_analysis,
                   generate_scientific_methodology_report, create_confidence_visualization)


def load_agricultural_enhanced_data(regions: list = None, n_features: int = 128) -> pd.DataFrame:
    """
    Load enhanced agricultural-focused environmental data
    
    This function enhances the standard AlphaEarth data loading with specific focus
    on agricultural areas, irrigation systems, and crop productivity metrics.
    """
    # Load base environmental data
    base_df = load_alphaearth_embeddings(regions=regions, n_features=n_features)
    
    print("🌾 Enhancing data with agricultural focus...")
    
    # Add agricultural-specific enhancements
    agricultural_regions = {
        "Karakalpakstan": {"irrigation_eff": 0.45, "primary_crop": "cotton", "water_stress": "extreme"},
        "Tashkent": {"irrigation_eff": 0.75, "primary_crop": "vegetables", "water_stress": "moderate"},
        "Samarkand": {"irrigation_eff": 0.60, "primary_crop": "cotton", "water_stress": "moderate"},
        "Bukhara": {"irrigation_eff": 0.50, "primary_crop": "cotton", "water_stress": "high"},
        "Namangan": {"irrigation_eff": 0.70, "primary_crop": "fruits", "water_stress": "low"}
    }
    
    # Enhance each sample with agricultural characteristics
    for idx, row in base_df.iterrows():
        region = row['region']
        if region in agricultural_regions:
            region_data = agricultural_regions[region]
            
            # Add agricultural variables
            base_df.loc[idx, 'agricultural_focus'] = True
            base_df.loc[idx, 'irrigation_efficiency'] = region_data['irrigation_eff'] + np.random.normal(0, 0.1)
            base_df.loc[idx, 'primary_crop_type'] = region_data['primary_crop']
            base_df.loc[idx, 'irrigation_system'] = np.random.choice(['drip', 'furrow', 'sprinkler'], 
                                                                    p=[0.3, 0.5, 0.2])
            
            # Enhanced water variables for agriculture
            base_df.loc[idx, 'crop_water_requirement'] = calculate_crop_water_need(region_data)
            base_df.loc[idx, 'soil_moisture_volumetric'] = calculate_volumetric_moisture(
                row['soil_moisture_est'], region_data)
            base_df.loc[idx, 'water_use_efficiency'] = region_data['irrigation_eff'] * 0.85
            base_df.loc[idx, 'estimated_crop_yield'] = calculate_crop_yield(region_data, 
                                                     base_df.loc[idx, 'soil_moisture_volumetric'])
            
            # Agricultural stress indicators
            base_df.loc[idx, 'agricultural_water_stress'] = calculate_ag_water_stress(region_data)
            base_df.loc[idx, 'irrigation_management_score'] = region_data['irrigation_eff']
    
    print(f"    Enhanced {len(base_df)} samples with agricultural characteristics")
    return base_df


def calculate_crop_water_need(region_data: dict) -> float:
    """Calculate crop-specific water requirements"""
    crop_requirements = {
        "cotton": 800,  # mm/season
        "vegetables": 600,
        "fruits": 700,
        "wheat": 450
    }
    return crop_requirements.get(region_data['primary_crop'], 600)


def calculate_volumetric_moisture(base_moisture: float, region_data: dict) -> float:
    """Calculate realistic volumetric water content"""
    # Convert base moisture to volumetric percentage
    volumetric = base_moisture * 0.6  # Base conversion
    
    # Adjust for irrigation efficiency
    irrigation_boost = region_data['irrigation_eff'] * 0.2
    
    return np.clip(volumetric + irrigation_boost, 0.1, 0.6)


def calculate_crop_yield(region_data: dict, moisture: float) -> float:
    """Calculate estimated crop yield based on moisture and conditions"""
    base_yields = {
        "cotton": 3.5,  # tons/ha
        "vegetables": 25.0,
        "fruits": 15.0,
        "wheat": 4.0
    }
    
    base_yield = base_yields.get(region_data['primary_crop'], 3.0)
    moisture_factor = min(1.2, moisture / 0.4)  # Optimal around 40%
    irrigation_factor = region_data['irrigation_eff']
    
    return base_yield * moisture_factor * irrigation_factor


def calculate_ag_water_stress(region_data: dict) -> float:
    """Calculate agricultural water stress index"""
    stress_levels = {
        "low": 0.2,
        "moderate": 0.4,
        "high": 0.7,
        "extreme": 0.9
    }
    return stress_levels.get(region_data['water_stress'], 0.5)


def analyze_irrigation_efficiency(df: pd.DataFrame) -> dict:
    """Analyze irrigation efficiency across regions and systems"""
    print("🚿 Analyzing irrigation efficiency for agricultural optimization...")
    
    # Regional efficiency analysis
    regional_efficiency = df.groupby('region').agg({
        'irrigation_efficiency': ['mean', 'std', 'min', 'max'],
        'water_use_efficiency': ['mean', 'std'],
        'estimated_crop_yield': ['mean', 'std']
    }).round(3)
    
    # System efficiency comparison
    if 'irrigation_system' in df.columns:
        system_efficiency = df.groupby('irrigation_system').agg({
            'irrigation_efficiency': ['mean', 'std', 'count'],
            'estimated_crop_yield': ['mean', 'std']
        }).round(3)
    else:
        system_efficiency = pd.DataFrame()
    
    # Calculate improvement potential
    df['efficiency_improvement_potential'] = 0.8 - df['irrigation_efficiency']  # Target 80% efficiency
    df['efficiency_improvement_potential'] = df['efficiency_improvement_potential'].clip(lower=0)
    
    total_improvement_potential = df['efficiency_improvement_potential'].sum()
    
    return {
        'regional_efficiency': regional_efficiency,
        'system_efficiency': system_efficiency,
        'total_improvement_potential': total_improvement_potential,
        'avg_current_efficiency': df['irrigation_efficiency'].mean(),
        'efficiency_target': 0.8
    }


def run():
    """Enhanced comprehensive agricultural soil moisture analysis"""
    print("Running enhanced comprehensive agricultural soil moisture analysis...")
    print("🌾 Focus: Agricultural heartlands and irrigation efficiency")
    
    cfg = load_config()
    tables = cfg["paths"]["tables"]
    figs = cfg["paths"]["figs"]
    reports = "reports"
    ensure_dir(tables); ensure_dir(figs); ensure_dir(reports)
    setup_plotting()
    
    # Load agricultural-enhanced environmental data
    print("Loading agricultural-enhanced environmental data...")
    embeddings_df = load_agricultural_enhanced_data(regions=cfg['regions'], n_features=128)
    
    # Enhanced irrigation efficiency analysis
    irrigation_analysis = analyze_irrigation_efficiency(embeddings_df)
    
    print("Building agricultural soil moisture prediction models...")
    
    # Feature engineering for agricultural focus
    embedding_cols = [col for col in embeddings_df.columns if col.startswith('embedding_')]
    agricultural_features = [
        'latitude', 'longitude', 'elevation', 'annual_precipitation',
        'irrigation_efficiency', 'water_use_efficiency', 'crop_water_requirement',
        'soil_moisture_volumetric', 'agricultural_water_stress'
    ]
    
    # Available features (filter out missing columns)
    available_features = [col for col in agricultural_features if col in embeddings_df.columns]
    
    # Create feature matrix
    feature_df = embeddings_df[embedding_cols + available_features].fillna(0)
    
    # Agricultural soil moisture modeling
    X = feature_df
    y = embeddings_df['soil_moisture_volumetric'] if 'soil_moisture_volumetric' in embeddings_df.columns else embeddings_df['soil_moisture_est']
    
    # Enhanced model training with agricultural focus
    models_to_test = {
        'Agricultural RF': RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42),
        'Agricultural GBM': GradientBoostingRegressor(n_estimators=100, max_depth=8, random_state=42),
        'Agricultural ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
    }
    
    best_model_results = {}
    best_r2 = -np.inf
    best_model_name = ""
    
    for model_name, model in models_to_test.items():
        print(f"Testing {model_name} for agricultural applications...")
        
        # Agricultural-focused cross-validation
        cv_results = perform_cross_validation(X.values, y.values, model, 'regression', cv_folds=5)
        
        if cv_results['r2_mean'] > best_r2:
            best_r2 = cv_results['r2_mean']
            best_model_name = model_name
            best_model_results = cv_results
            best_model_results['model_name'] = model_name
    
    print(f"Best agricultural model: {best_model_name} with R² = {best_r2:.3f}")
    
    # Enhanced agricultural analysis
    best_model = models_to_test[best_model_name]
    
    # Generate agricultural predictions
    best_model.fit(X, y)
    embeddings_df['soil_moisture_predicted'] = best_model.predict(X)
    
    # Agricultural water stress categorization
    def categorize_agricultural_stress(moisture_level, irrigation_eff):
        """Enhanced agricultural water stress categorization"""
        if moisture_level < 0.25 and irrigation_eff < 0.5:
            return "Critical"
        elif moisture_level < 0.35 or irrigation_eff < 0.6:
            return "High"
        elif moisture_level < 0.45:
            return "Moderate"
        else:
            return "Low"
    
    if 'irrigation_efficiency' in embeddings_df.columns:
        embeddings_df['agricultural_water_stress_category'] = embeddings_df.apply(
            lambda row: categorize_agricultural_stress(row['soil_moisture_predicted'], 
                                                     row['irrigation_efficiency']), axis=1)
    else:
        embeddings_df['agricultural_water_stress_category'] = embeddings_df['soil_moisture_predicted'].apply(
            lambda x: "Critical" if x < 0.25 else "High" if x < 0.35 else "Moderate" if x < 0.45 else "Low")
    
    # Enhanced regional summary with agricultural focus
    regional_stats = []
    for region in cfg['regions']:
        region_data = embeddings_df[embeddings_df['region'] == region]
        
        if len(region_data) > 0:
            moisture_values = region_data['soil_moisture_predicted'].values
            ci_low, ci_high = calculate_confidence_interval(moisture_values)
            
            stats = {
                'region': region,
                'n_samples': len(region_data),
                'mean_moisture': moisture_values.mean(),
                'std_moisture': moisture_values.std(),
                'ci_lower': ci_low,
                'ci_upper': ci_high,
                'critical_stress_pct': (region_data['agricultural_water_stress_category'] == 'Critical').mean() * 100,
                'high_stress_pct': (region_data['agricultural_water_stress_category'] == 'High').mean() * 100,
                'avg_irrigation_efficiency': region_data.get('irrigation_efficiency', pd.Series([0.5])).mean(),
                'avg_crop_yield': region_data.get('estimated_crop_yield', pd.Series([3.0])).mean()
            }
            regional_stats.append(stats)
    
    regional_summary_df = pd.DataFrame(regional_stats)
    
    # Create enhanced agricultural visualizations
    print("Generating enhanced agricultural visualizations...")
    
    # 1. Agricultural water stress and irrigation efficiency
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # Soil moisture by region with agricultural focus
    sns.boxplot(data=embeddings_df, x='region', y='soil_moisture_predicted', ax=axes[0,0])
    axes[0,0].set_title('Agricultural Soil Moisture Distribution by Region')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Agricultural water stress categories
    stress_counts = embeddings_df['agricultural_water_stress_category'].value_counts()
    colors = {'Critical': 'darkred', 'High': 'red', 'Moderate': 'orange', 'Low': 'green'}
    axes[0,1].pie(stress_counts.values, labels=stress_counts.index, autopct='%1.1f%%',
                  colors=[colors.get(x, 'gray') for x in stress_counts.index])
    axes[0,1].set_title('Agricultural Water Stress Distribution')
    
    # Irrigation efficiency by region (if available)
    if 'irrigation_efficiency' in embeddings_df.columns:
        regional_irrigation = embeddings_df.groupby('region')['irrigation_efficiency'].mean()
        axes[1,0].bar(regional_irrigation.index, regional_irrigation.values, alpha=0.8, color='skyblue')
        axes[1,0].set_title('Average Irrigation Efficiency by Region')
        axes[1,0].set_ylabel('Irrigation Efficiency')
        axes[1,0].tick_params(axis='x', rotation=45)
    
    # Crop yield potential (if available)
    if 'estimated_crop_yield' in embeddings_df.columns:
        axes[1,1].scatter(embeddings_df['soil_moisture_predicted'], embeddings_df['estimated_crop_yield'],
                         alpha=0.6, c=embeddings_df['region'].astype('category').cat.codes, cmap='tab10')
        axes[1,1].set_xlabel('Predicted Soil Moisture')
        axes[1,1].set_ylabel('Estimated Crop Yield (tons/ha)')
        axes[1,1].set_title('Soil Moisture vs Agricultural Productivity')
        
        # Add correlation if both variables exist
        corr = embeddings_df['soil_moisture_predicted'].corr(embeddings_df['estimated_crop_yield'])
        axes[1,1].text(0.05, 0.95, f'R = {corr:.3f}', transform=axes[1,1].transAxes,
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    save_plot(fig, f"{figs}/agricultural_soil_moisture_analysis.png", 
              "Enhanced Agricultural Soil Moisture Analysis - Uzbekistan")
    
    # Generate comprehensive agricultural data tables
    print("Generating enhanced agricultural data tables...")
    
    # 1. Regional agricultural summary
    regional_summary_df.to_csv(f"{tables}/agricultural_soil_moisture_regional_summary.csv", index=False)
    
    # 2. Irrigation efficiency analysis
    if irrigation_analysis['system_efficiency'] is not None and not irrigation_analysis['system_efficiency'].empty:
        irrigation_analysis['system_efficiency'].to_csv(f"{tables}/irrigation_efficiency_analysis.csv")
    
    irrigation_analysis['regional_efficiency'].to_csv(f"{tables}/regional_irrigation_efficiency.csv")
    
    # 3. Agricultural model performance
    agricultural_model_metrics = pd.DataFrame({
        'metric': ['Best_Model', 'R2_Score', 'Agricultural_Focus', 'Irrigation_Improvement_Potential', 'Samples_Analyzed'],
        'value': [best_model_name, best_r2, 'Yes', 
                 irrigation_analysis['total_improvement_potential'], len(embeddings_df)]
    })
    agricultural_model_metrics.to_csv(f"{tables}/agricultural_model_performance.csv", index=False)
    
    # 4. Critical agricultural areas
    critical_areas = embeddings_df[
        embeddings_df['agricultural_water_stress_category'].isin(['Critical', 'High'])
    ][['region', 'latitude', 'longitude', 'agricultural_water_stress_category', 
       'soil_moisture_predicted']].copy()
    
    if len(critical_areas) > 0:
        critical_areas.to_csv(f"{tables}/critical_agricultural_areas.csv", index=False)
    
    # Generate agricultural report
    print("Generating comprehensive agricultural report...")
    
    total_samples = len(embeddings_df)
    critical_areas_count = (embeddings_df['agricultural_water_stress_category'] == 'Critical').sum()
    high_stress_areas_count = (embeddings_df['agricultural_water_stress_category'] == 'High').sum()
    avg_moisture = embeddings_df['soil_moisture_predicted'].mean()
    avg_irrigation_eff = embeddings_df.get('irrigation_efficiency', pd.Series([0.6])).mean()
    
    # Agricultural report content
    agricultural_report = f"""# Enhanced Agricultural Soil Moisture Analysis Report

**Analysis Date:** {datetime.now().strftime('%B %d, %Y')}  
**Focus:** Agricultural heartlands and irrigation efficiency
**Coverage:** 5 regions of Uzbekistan  
**Analysis Method:** AlphaEarth satellite embeddings with agricultural machine learning

## Executive Summary

This enhanced analysis focuses specifically on agricultural heartlands, revealing critical insights 
for irrigation optimization and crop productivity improvement across Uzbekistan.

### Key Agricultural Findings

- **National Agricultural Soil Moisture Average:** {avg_moisture:.1%}
- **Critical Agricultural Stress Areas:** {critical_areas_count:,} locations
- **High Agricultural Stress Areas:** {high_stress_areas_count:,} locations  
- **Average Irrigation Efficiency:** {avg_irrigation_eff:.1%}
- **Irrigation Improvement Potential:** {irrigation_analysis['total_improvement_potential']:.1f} efficiency units

## Agricultural Recommendations

### Immediate Priorities (0-6 months)
1. **Emergency irrigation support** in {critical_areas_count} critically stressed agricultural areas
2. **Precision moisture monitoring** deployment in high-value crop zones
3. **Irrigation efficiency training** for farmer cooperatives

### Medium-term Improvements (6-18 months)
1. **Drip irrigation upgrades** in regions with <60% efficiency
2. **Crop selection optimization** based on water availability patterns
3. **Regional water-sharing agreements** between high and low efficiency areas

### Long-term Agricultural Strategy (18+ months)
1. **Integrated agricultural water management** system deployment
2. **Climate-resilient crop varieties** introduction and support
3. **Satellite-based precision agriculture** full implementation

---

*Report generated using AlphaEarth agricultural soil moisture analysis framework.*
"""

    # Write agricultural report
    with open(f"{reports}/agricultural_soil_moisture_enhanced.md", 'w') as f:
        f.write(agricultural_report)
    
    print("Enhanced agricultural soil moisture analysis completed successfully!")
    print(f"Agricultural findings:")
    print(f"  - Agricultural samples analyzed: {total_samples}")
    print(f"  - Best agricultural model: {best_model_name} (R² = {best_r2:.3f})")
    print(f"  - Average agricultural soil moisture: {avg_moisture:.3f}")
    print(f"  - Critical agricultural stress areas: {critical_areas_count}")
    print(f"  - Average irrigation efficiency: {avg_irrigation_eff:.3f}")
    print(f"  - Irrigation improvement potential: {irrigation_analysis['total_improvement_potential']:.2f}")
    
    artifacts = [
        "tables/agricultural_soil_moisture_regional_summary.csv",
        "tables/regional_irrigation_efficiency.csv",
        "tables/agricultural_model_performance.csv",
        "figs/agricultural_soil_moisture_analysis.png",
        "reports/agricultural_soil_moisture_enhanced.md"
    ]
    
    # Add irrigation system analysis if available
    if irrigation_analysis['system_efficiency'] is not None and not irrigation_analysis['system_efficiency'].empty:
        artifacts.append("tables/irrigation_efficiency_analysis.csv")
    
    # Add critical areas file if any exist
    if critical_areas_count > 0:
        artifacts.append("tables/critical_agricultural_areas.csv")
    
    return {
        "status": "ok",
        "artifacts": artifacts,
        "summary_stats": {
            "total_agricultural_samples": int(total_samples),
            "best_agricultural_model": best_model_name,
            "model_r2": float(best_r2),
            "avg_agricultural_moisture": float(avg_moisture),
            "critical_agricultural_areas": int(critical_areas_count),
            "high_stress_agricultural_areas": int(high_stress_areas_count),
            "avg_irrigation_efficiency": float(avg_irrigation_eff),
            "irrigation_improvement_potential": float(irrigation_analysis['total_improvement_potential']),
            "agricultural_focus": True
        }
    }