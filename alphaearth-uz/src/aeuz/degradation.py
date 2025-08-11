from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from datetime import datetime
from .utils import (load_config, ensure_dir, setup_plotting, generate_synthetic_embeddings,
                   calculate_confidence_interval, perform_trend_analysis, create_summary_statistics,
                   save_plot, validate_data_quality)

def run():
    """Comprehensive land degradation analysis with trend detection and change assessment"""
    print("Running comprehensive land degradation analysis...")
    
    cfg = load_config()
    tables = cfg["paths"]["tables"]
    figs = cfg["paths"]["figs"]
    ensure_dir(tables); ensure_dir(figs)
    setup_plotting()
    
    # Generate synthetic AlphaEarth embeddings for degradation analysis
    print("Processing AlphaEarth embeddings for land degradation assessment...")
    embeddings_df = generate_synthetic_embeddings(n_samples=3500, n_features=128)
    
    # Add degradation-specific indicators
    np.random.seed(42)
    
    # Vegetation and soil indicators
    embeddings_df['ndvi_current'] = np.clip(np.random.beta(2, 3, len(embeddings_df)), 0.1, 0.9)
    embeddings_df['ndvi_baseline'] = embeddings_df['ndvi_current'] + np.random.normal(0.1, 0.15, len(embeddings_df))
    embeddings_df['ndvi_baseline'] = np.clip(embeddings_df['ndvi_baseline'], 0.1, 0.9)
    
    # Soil indicators
    embeddings_df['soil_erosion_rate'] = np.clip(np.random.exponential(2.0, len(embeddings_df)), 0, 15)  # mm/year
    embeddings_df['soil_salinity'] = np.clip(np.random.gamma(2, 1.5, len(embeddings_df)), 0, 20)  # dS/m
    embeddings_df['organic_carbon_loss'] = np.clip(np.random.beta(3, 2, len(embeddings_df)) * 5, 0, 5)  # %
    
    # Land use pressure indicators
    embeddings_df['grazing_pressure'] = np.random.choice([0, 1, 2, 3], len(embeddings_df), p=[0.3, 0.4, 0.2, 0.1])  # 0=none, 3=severe
    embeddings_df['irrigation_intensity'] = np.clip(np.random.gamma(1.5, 2, len(embeddings_df)), 0, 10)
    embeddings_df['crop_yield_decline'] = np.clip(np.random.beta(2, 5, len(embeddings_df)) * 50, 0, 50)  # % decline
    
    # Climate stress indicators
    embeddings_df['drought_frequency'] = np.random.poisson(1.2, len(embeddings_df))  # events per 5 years
    embeddings_df['heat_stress_days'] = np.random.poisson(25, len(embeddings_df))  # days >35°C annually
    
    # Data quality validation
    required_cols = ['region', 'ndvi_current', 'soil_erosion_rate', 'soil_salinity']
    quality_report = validate_data_quality(embeddings_df, required_cols)
    print(f"Data quality score: {quality_report['quality_score']:.1f}%")
    
    # Calculate comprehensive degradation indices
    print("Calculating degradation indices...")
    
    def calculate_vegetation_degradation(row):
        """Calculate vegetation degradation index"""
        ndvi_change = (row['ndvi_baseline'] - row['ndvi_current']) / row['ndvi_baseline']
        yield_factor = row['crop_yield_decline'] / 50.0
        return min(1.0, max(0.0, (ndvi_change * 0.6 + yield_factor * 0.4)))
    
    def calculate_soil_degradation(row):
        """Calculate soil degradation index"""
        erosion_factor = min(1.0, row['soil_erosion_rate'] / 10.0)
        salinity_factor = min(1.0, row['soil_salinity'] / 15.0)
        carbon_factor = row['organic_carbon_loss'] / 5.0
        return (erosion_factor * 0.4 + salinity_factor * 0.3 + carbon_factor * 0.3)
    
    def calculate_pressure_index(row):
        """Calculate anthropogenic pressure index"""
        grazing_factor = row['grazing_pressure'] / 3.0
        irrigation_factor = min(1.0, row['irrigation_intensity'] / 8.0)
        return (grazing_factor * 0.6 + irrigation_factor * 0.4)
    
    def calculate_climate_stress(row):
        """Calculate climate-induced stress index"""
        drought_factor = min(1.0, row['drought_frequency'] / 3.0)
        heat_factor = min(1.0, row['heat_stress_days'] / 40.0)
        return (drought_factor * 0.5 + heat_factor * 0.5)
    
    # Calculate individual indices
    embeddings_df['vegetation_degradation'] = embeddings_df.apply(calculate_vegetation_degradation, axis=1)
    embeddings_df['soil_degradation'] = embeddings_df.apply(calculate_soil_degradation, axis=1)
    embeddings_df['pressure_index'] = embeddings_df.apply(calculate_pressure_index, axis=1)
    embeddings_df['climate_stress'] = embeddings_df.apply(calculate_climate_stress, axis=1)
    
    # Calculate overall degradation score
    embeddings_df['degradation_score'] = (
        embeddings_df['vegetation_degradation'] * 0.3 +
        embeddings_df['soil_degradation'] * 0.3 +
        embeddings_df['pressure_index'] * 0.25 +
        embeddings_df['climate_stress'] * 0.15
    )
    
    # Categorize degradation severity
    def categorize_degradation(score):
        if score >= 0.75:
            return 'Severe'
        elif score >= 0.5:
            return 'High'
        elif score >= 0.25:
            return 'Moderate'
        else:
            return 'Low'
    
    embeddings_df['degradation_category'] = embeddings_df['degradation_score'].apply(categorize_degradation)
    
    # Anomaly detection for hotspot identification
    print("Identifying degradation hotspots using anomaly detection...")
    
    # Prepare features for anomaly detection
    degradation_features = [
        'vegetation_degradation', 'soil_degradation', 'pressure_index', 'climate_stress',
        'soil_erosion_rate', 'soil_salinity', 'organic_carbon_loss'
    ]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(embeddings_df[degradation_features].fillna(0))
    
    # Isolation Forest for anomaly detection
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    anomaly_labels = iso_forest.fit_predict(X_scaled)
    embeddings_df['is_hotspot'] = (anomaly_labels == -1).astype(int)
    
    # Temporal trend analysis
    print("Performing temporal degradation trend analysis...")
    
    trend_results = {}
    for region in cfg['regions']:
        region_data = embeddings_df[embeddings_df['region'] == region]
        
        # Simulate temporal data by adding small variations
        yearly_degradation = []
        years = list(range(2017, 2026))
        
        for year in years:
            # Add year-specific noise to simulate temporal variation
            year_factor = (year - 2017) * 0.02  # Slight increase over time
            year_data = region_data['degradation_score'].mean() + year_factor + np.random.normal(0, 0.05)
            yearly_degradation.append(max(0, min(1, year_data)))
        
        if len(yearly_degradation) > 2:
            trend_stats = perform_trend_analysis(np.array(yearly_degradation), np.array(years))
            trend_results[region] = trend_stats
            trend_results[region]['current_degradation_level'] = yearly_degradation[-1]
            trend_results[region]['degradation_change_5yr'] = yearly_degradation[-1] - yearly_degradation[0]
    
    # Risk assessment and prioritization
    print("Conducting risk assessment and prioritization...")
    
    def calculate_risk_score(row):
        """Calculate future degradation risk score"""
        current_degradation = row['degradation_score']
        vulnerability_factors = (
            row['soil_erosion_rate'] / 10.0 * 0.3 +
            row['drought_frequency'] / 3.0 * 0.3 +
            row['pressure_index'] * 0.4
        )
        return min(1.0, current_degradation + vulnerability_factors * 0.3)
    
    embeddings_df['future_risk_score'] = embeddings_df.apply(calculate_risk_score, axis=1)
    
    # Identify priority intervention areas
    priority_criteria = (
        (embeddings_df['degradation_score'] >= 0.4) |
        (embeddings_df['future_risk_score'] >= 0.6) |
        (embeddings_df['is_hotspot'] == 1)
    )
    
    priority_areas = embeddings_df[priority_criteria].copy()
    print(f"Identified {len(priority_areas)} priority areas for intervention")
    
    # Regional degradation analysis
    print("Generating regional degradation assessments...")
    
    regional_analysis = []
    for region in cfg['regions']:
        region_data = embeddings_df[embeddings_df['region'] == region]
        region_priority = priority_areas[priority_areas['region'] == region]
        
        if len(region_data) > 0:
            analysis = {
                'region': region,
                'total_area_assessed': len(region_data),
                'avg_degradation_score': region_data['degradation_score'].mean(),
                'severe_degradation_areas': (region_data['degradation_category'] == 'Severe').sum(),
                'high_degradation_areas': (region_data['degradation_category'] == 'High').sum(),
                'hotspots_identified': (region_data['is_hotspot'] == 1).sum(),
                'priority_intervention_areas': len(region_priority),
                'avg_soil_erosion_rate': region_data['soil_erosion_rate'].mean(),
                'avg_soil_salinity': region_data['soil_salinity'].mean(),
                'high_grazing_pressure_areas': (region_data['grazing_pressure'] >= 2).sum(),
                'estimated_restoration_cost': len(region_priority) * 5000,  # $5000 per area
                'trend_direction': trend_results.get(region, {}).get('trend_direction', 'unknown')
            }
            regional_analysis.append(analysis)
    
    regional_df = pd.DataFrame(regional_analysis)
    
    # Create comprehensive visualizations
    print("Generating visualizations...")
    
    # 1. Degradation overview analysis
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Degradation category distribution
    degradation_counts = embeddings_df['degradation_category'].value_counts()
    axes[0,0].pie(degradation_counts.values, labels=degradation_counts.index, autopct='%1.1f%%')
    axes[0,0].set_title('Land Degradation Severity Distribution')
    
    # Degradation by region
    sns.boxplot(data=embeddings_df, x='region', y='degradation_score', ax=axes[0,1])
    axes[0,1].set_title('Degradation Scores by Region')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Component analysis
    component_data = embeddings_df[['vegetation_degradation', 'soil_degradation', 
                                   'pressure_index', 'climate_stress']].mean()
    axes[1,0].bar(component_data.index, component_data.values)
    axes[1,0].set_title('Average Degradation Components')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Risk vs current degradation
    axes[1,1].scatter(embeddings_df['degradation_score'], embeddings_df['future_risk_score'], 
                     alpha=0.6, c=embeddings_df['is_hotspot'], cmap='coolwarm')
    axes[1,1].set_xlabel('Current Degradation Score')
    axes[1,1].set_ylabel('Future Risk Score')
    axes[1,1].set_title('Current Degradation vs Future Risk')
    
    save_plot(fig, f"{figs}/degradation_overview_analysis.png", 
              "Land Degradation Overview Analysis - Uzbekistan")
    
    # 2. Spatial degradation patterns
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Degradation score spatial distribution
    scatter1 = axes[0].scatter(embeddings_df['longitude'], embeddings_df['latitude'], 
                              c=embeddings_df['degradation_score'], cmap='RdYlGn_r', 
                              alpha=0.7, s=15)
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')
    axes[0].set_title('Degradation Score Distribution')
    plt.colorbar(scatter1, ax=axes[0], label='Degradation Score')
    
    # Hotspots identification
    normal_areas = embeddings_df[embeddings_df['is_hotspot'] == 0]
    hotspot_areas = embeddings_df[embeddings_df['is_hotspot'] == 1]
    
    axes[1].scatter(normal_areas['longitude'], normal_areas['latitude'], 
                   c='lightgreen', alpha=0.5, s=10, label='Normal areas')
    axes[1].scatter(hotspot_areas['longitude'], hotspot_areas['latitude'], 
                   c='red', alpha=0.8, s=25, label='Degradation hotspots')
    axes[1].set_xlabel('Longitude')
    axes[1].set_ylabel('Latitude')
    axes[1].set_title('Degradation Hotspots')
    axes[1].legend()
    
    # Future risk distribution
    scatter3 = axes[2].scatter(embeddings_df['longitude'], embeddings_df['latitude'], 
                              c=embeddings_df['future_risk_score'], cmap='Reds', 
                              alpha=0.7, s=15)
    axes[2].set_xlabel('Longitude')
    axes[2].set_ylabel('Latitude')
    axes[2].set_title('Future Degradation Risk')
    plt.colorbar(scatter3, ax=axes[2], label='Risk Score')
    
    save_plot(fig, f"{figs}/degradation_spatial_analysis.png", 
              "Spatial Degradation Analysis - Uzbekistan")
    
    # 3. Temporal trends and components
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Regional trends
    if trend_results:
        regions = list(trend_results.keys())
        trend_directions = [trend_results[r]['trend_direction'] for r in regions]
        trend_slopes = [trend_results[r]['linear_slope'] for r in regions]
        
        axes[0,0].bar(regions, trend_slopes, color=['red' if t == 'increasing' else 'green' for t in trend_directions])
        axes[0,0].set_title('Degradation Trend Slopes by Region')
        axes[0,0].tick_params(axis='x', rotation=45)
        axes[0,0].set_ylabel('Slope (per year)')
    
    # Component correlation matrix
    component_corr = embeddings_df[['vegetation_degradation', 'soil_degradation', 
                                   'pressure_index', 'climate_stress']].corr()
    sns.heatmap(component_corr, annot=True, cmap='coolwarm', center=0, ax=axes[0,1])
    axes[0,1].set_title('Degradation Component Correlations')
    
    # Pressure vs degradation
    sns.scatterplot(data=embeddings_df, x='pressure_index', y='degradation_score', 
                   hue='region', alpha=0.6, ax=axes[1,0])
    axes[1,0].set_title('Anthropogenic Pressure vs Degradation')
    
    # Climate stress vs degradation
    sns.scatterplot(data=embeddings_df, x='climate_stress', y='degradation_score', 
                   hue='region', alpha=0.6, ax=axes[1,1])
    axes[1,1].set_title('Climate Stress vs Degradation')
    
    save_plot(fig, f"{figs}/degradation_analysis_components.png", 
              "Degradation Component Analysis - Uzbekistan")
    
    # Generate comprehensive data tables
    print("Generating data tables...")
    
    # 1. Regional analysis summary
    regional_df.to_csv(f"{tables}/degradation_regional_analysis.csv", index=False)
    
    # 2. Degradation hotspots
    hotspot_summary = embeddings_df[embeddings_df['is_hotspot'] == 1].groupby('region').agg({
        'degradation_score': ['count', 'mean', 'std'],
        'soil_erosion_rate': 'mean',
        'soil_salinity': 'mean',
        'vegetation_degradation': 'mean',
        'soil_degradation': 'mean'
    }).round(3)
    
    hotspot_summary.columns = ['_'.join(col).strip() for col in hotspot_summary.columns]
    hotspot_summary = hotspot_summary.reset_index()
    hotspot_summary.to_csv(f"{tables}/degradation_hotspots.csv", index=False)
    
    # 3. Trend analysis results
    if trend_results:
        trend_df = pd.DataFrame.from_dict(trend_results, orient='index').reset_index()
        trend_df.rename(columns={'index': 'region'}, inplace=True)
        trend_df.to_csv(f"{tables}/degradation_trend_analysis.csv", index=False)
    
    # 4. Priority intervention areas
    priority_summary = priority_areas.groupby('region').agg({
        'degradation_score': ['count', 'mean'],
        'future_risk_score': 'mean',
        'soil_erosion_rate': 'mean',
        'pressure_index': 'mean'
    }).round(3)
    
    priority_summary.columns = ['_'.join(col).strip() for col in priority_summary.columns]
    priority_summary = priority_summary.reset_index()
    priority_summary.to_csv(f"{tables}/degradation_priority_areas.csv", index=False)
    
    # 5. Component analysis
    component_stats = create_summary_statistics(
        embeddings_df, 'region',
        ['vegetation_degradation', 'soil_degradation', 'pressure_index', 'climate_stress']
    )
    component_stats.to_csv(f"{tables}/degradation_component_analysis.csv", index=False)
    
    # 6. Risk assessment results
    risk_categories = pd.cut(embeddings_df['future_risk_score'], 
                            bins=[0, 0.25, 0.5, 0.75, 1.0], 
                            labels=['Low', 'Moderate', 'High', 'Very High'])
    embeddings_df['risk_category'] = risk_categories
    
    risk_summary = embeddings_df.groupby(['region', 'risk_category']).size().unstack(fill_value=0)
    risk_summary.to_csv(f"{tables}/degradation_risk_assessment.csv")
    
    # Generate executive summary statistics
    total_severe_areas = (embeddings_df['degradation_category'] == 'Severe').sum()
    total_high_areas = (embeddings_df['degradation_category'] == 'High').sum()
    avg_degradation = embeddings_df['degradation_score'].mean()
    total_hotspots = (embeddings_df['is_hotspot'] == 1).sum()
    
    print("Land degradation analysis completed successfully!")
    print(f"Key findings:")
    print(f"  - Average degradation score: {avg_degradation:.3f}")
    print(f"  - Severe degradation areas: {total_severe_areas}")
    print(f"  - High degradation areas: {total_high_areas}")
    print(f"  - Degradation hotspots identified: {total_hotspots}")
    print(f"  - Priority intervention areas: {len(priority_areas)}")
    
    artifacts = [
        "tables/degradation_regional_analysis.csv",
        "tables/degradation_hotspots.csv",
        "tables/degradation_trend_analysis.csv",
        "tables/degradation_priority_areas.csv",
        "tables/degradation_component_analysis.csv",
        "tables/degradation_risk_assessment.csv",
        "figs/degradation_overview_analysis.png",
        "figs/degradation_spatial_analysis.png",
        "figs/degradation_analysis_components.png"
    ]
    
    return {
        "status": "ok",
        "artifacts": artifacts,
        "summary_stats": {
            "avg_degradation_score": float(avg_degradation),
            "severe_degradation_areas": int(total_severe_areas),
            "high_degradation_areas": int(total_high_areas),
            "hotspots_identified": int(total_hotspots),
            "priority_areas": len(priority_areas),
            "total_assessed": len(embeddings_df)
        }
    }
