from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from datetime import datetime
from .utils import (load_config, ensure_dir, setup_plotting, load_alphaearth_embeddings,
                   calculate_confidence_interval, perform_trend_analysis, create_summary_statistics,
                   save_plot, validate_data_quality, perform_cross_validation, 
                   enhance_model_with_feature_selection, create_pilot_study_analysis,
                   generate_scientific_methodology_report, create_confidence_visualization)

def run():
    """Comprehensive urban heat island analysis with LST modeling and mitigation strategies"""
    print("Running comprehensive urban heat island analysis...")
    
    cfg = load_config()
    tables = cfg["paths"]["tables"]
    figs = cfg["paths"]["figs"]
    ensure_dir(tables); ensure_dir(figs)
    setup_plotting()
    
    # Load AlphaEarth satellite embeddings for urban heat analysis
    print("Loading AlphaEarth satellite embeddings for urban heat assessment...")
    embeddings_df = load_alphaearth_embeddings(regions=cfg['regions'], n_features=96)
    
    # Add urban heat-specific variables
    np.random.seed(42)
    
    # Land use/land cover characteristics
    embeddings_df['built_up_density'] = np.clip(np.random.beta(2, 5, len(embeddings_df)), 0, 1)
    embeddings_df['green_space_ratio'] = np.clip(1 - embeddings_df['built_up_density'] + 
                                                 np.random.normal(0, 0.2, len(embeddings_df)), 0, 1)
    embeddings_df['impervious_surface_pct'] = embeddings_df['built_up_density'] * 90 + np.random.normal(0, 10, len(embeddings_df))
    embeddings_df['impervious_surface_pct'] = np.clip(embeddings_df['impervious_surface_pct'], 0, 100)
    
    # Urban characteristics  
    embeddings_df['building_height_avg'] = np.clip(np.random.gamma(2, 3, len(embeddings_df)), 1, 50)  # meters
    embeddings_df['population_density'] = np.clip(np.random.exponential(5000, len(embeddings_df)), 50, 25000)  # people/km²
    embeddings_df['distance_to_center_km'] = np.clip(np.random.exponential(8, len(embeddings_df)), 0.1, 40)
    
    # Environmental factors
    embeddings_df['albedo'] = np.clip(0.15 + embeddings_df['green_space_ratio'] * 0.15 + 
                                     np.random.normal(0, 0.05, len(embeddings_df)), 0.1, 0.6)
    embeddings_df['water_body_distance_km'] = np.clip(np.random.exponential(5, len(embeddings_df)), 0.1, 25)
    embeddings_df['elevation_urban_m'] = 500 + np.random.normal(0, 200, len(embeddings_df))
    
    # Simulate Land Surface Temperature (LST)
    def calculate_lst(row):
        """Calculate Land Surface Temperature based on urban factors"""
        base_temp = 25.0  # Base temperature (°C)
        
        # Urban heat contributions
        built_up_effect = row['built_up_density'] * 8.0  # Up to 8°C increase
        impervious_effect = (row['impervious_surface_pct'] / 100.0) * 6.0  # Up to 6°C
        density_effect = min(row['population_density'] / 10000.0, 1.0) * 4.0  # Up to 4°C
        
        # Cooling effects
        green_cooling = row['green_space_ratio'] * -5.0  # Up to 5°C cooling
        water_cooling = max(0, (5 - row['water_body_distance_km']) / 5.0) * -3.0  # Up to 3°C cooling
        albedo_cooling = (row['albedo'] - 0.15) * -10.0  # Albedo effect
        
        # Distance from center (heat island core)
        center_effect = max(0, (10 - row['distance_to_center_km']) / 10.0) * 3.0
        
        # Add some random variation
        random_variation = np.random.normal(0, 1.5)
        
        total_lst = (base_temp + built_up_effect + impervious_effect + density_effect + 
                    green_cooling + water_cooling + albedo_cooling + center_effect + random_variation)
        
        return max(15.0, total_lst)  # Minimum realistic temperature
    
    embeddings_df['lst_celsius'] = embeddings_df.apply(calculate_lst, axis=1)
    
    # Calculate heat island intensity (difference from rural baseline)
    rural_baseline = embeddings_df[embeddings_df['built_up_density'] < 0.1]['lst_celsius'].mean()
    embeddings_df['uhi_intensity'] = embeddings_df['lst_celsius'] - rural_baseline
    
    # Data quality validation
    required_cols = ['region', 'built_up_density', 'green_space_ratio', 'lst_celsius']
    quality_report = validate_data_quality(embeddings_df, required_cols)
    print(f"Data quality score: {quality_report['quality_score']:.1f}%")
    
    # Heat risk assessment
    print("Conducting heat risk assessment...")
    
    def calculate_heat_risk(row):
        """Calculate comprehensive heat risk score"""
        # Temperature risk (40% weight)
        temp_risk = min(1.0, max(0, (row['lst_celsius'] - 30) / 15))  # Risk above 30°C
        
        # Vulnerability factors (30% weight)
        pop_vulnerability = min(1.0, row['population_density'] / 15000)
        built_vulnerability = row['built_up_density']
        vulnerability = (pop_vulnerability + built_vulnerability) / 2
        
        # Cooling capacity (20% weight) - inverted
        cooling_capacity = 1 - row['green_space_ratio']
        
        # Exposure factors (10% weight)
        impervious_exposure = row['impervious_surface_pct'] / 100
        
        risk_score = (temp_risk * 0.4 + vulnerability * 0.3 + 
                     cooling_capacity * 0.2 + impervious_exposure * 0.1)
        
        return min(1.0, risk_score)
    
    embeddings_df['heat_risk_score'] = embeddings_df.apply(calculate_heat_risk, axis=1)
    
    # Categorize heat risk
    def categorize_heat_risk(score):
        if score >= 0.75:
            return 'Very High'
        elif score >= 0.5:
            return 'High'
        elif score >= 0.25:
            return 'Moderate'
        else:
            return 'Low'
    
    embeddings_df['heat_risk_category'] = embeddings_df['heat_risk_score'].apply(categorize_heat_risk)
    
    # Machine learning model for LST prediction
    print("Building LST prediction model...")
    
    # Prepare features
    embedding_cols = [col for col in embeddings_df.columns if col.startswith('embed_')]
    physical_features = [
        'built_up_density', 'green_space_ratio', 'impervious_surface_pct',
        'building_height_avg', 'population_density', 'distance_to_center_km',
        'albedo', 'water_body_distance_km', 'elevation_urban_m',
        'latitude', 'longitude'
    ]
    
    all_features = embedding_cols + physical_features
    
    X = embeddings_df[all_features].fillna(embeddings_df[all_features].mean())
    y = embeddings_df['lst_celsius']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Random Forest model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=12)
    rf_model.fit(X_train, y_train)
    
    # Model evaluation
    y_pred = rf_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"LST prediction model - R²: {r2:.3f}, RMSE: {rmse:.3f}°C")
    
    # Generate predictions
    embeddings_df['lst_predicted'] = rf_model.predict(X)
    
    # Urban cooling potential analysis
    print("Analyzing urban cooling potential...")
    
    def calculate_cooling_potential(row):
        """Calculate potential temperature reduction from green infrastructure"""
        current_green = row['green_space_ratio']
        max_feasible_green = min(0.6, current_green + (1 - row['built_up_density']) * 0.3)
        
        # Potential cooling from additional green space
        green_cooling_potential = (max_feasible_green - current_green) * 4.0  # 4°C per unit green ratio
        
        # Potential cooling from albedo improvement
        current_albedo = row['albedo']
        max_albedo = min(0.45, current_albedo + 0.15)  # Realistic albedo increase
        albedo_cooling_potential = (max_albedo - current_albedo) * 8.0
        
        return min(8.0, green_cooling_potential + albedo_cooling_potential)
    
    embeddings_df['cooling_potential'] = embeddings_df.apply(calculate_cooling_potential, axis=1)
    
    # Mitigation strategy assessment
    print("Developing mitigation strategies...")
    
    # Strategy 1: Green infrastructure expansion
    high_priority_green = embeddings_df[
        (embeddings_df['heat_risk_score'] >= 0.5) & 
        (embeddings_df['green_space_ratio'] < 0.3) &
        (embeddings_df['built_up_density'] < 0.8)
    ]
    
    # Strategy 2: Cool roof/pavement implementation
    high_priority_albedo = embeddings_df[
        (embeddings_df['heat_risk_score'] >= 0.6) & 
        (embeddings_df['impervious_surface_pct'] > 50) &
        (embeddings_df['albedo'] < 0.25)
    ]
    
    # Strategy 3: Water feature installation
    high_priority_water = embeddings_df[
        (embeddings_df['heat_risk_score'] >= 0.7) & 
        (embeddings_df['water_body_distance_km'] > 2.0)
    ]
    
    # Regional analysis
    print("Generating regional heat analysis...")
    
    regional_analysis = []
    for region in cfg['regions']:
        region_data = embeddings_df[embeddings_df['region'] == region]
        
        if len(region_data) > 0:
            # Urban areas (high built-up density)
            urban_areas = region_data[region_data['built_up_density'] > 0.3]
            
            analysis = {
                'region': region,
                'total_areas_assessed': len(region_data),
                'urban_areas': len(urban_areas),
                'avg_lst': region_data['lst_celsius'].mean(),
                'max_lst': region_data['lst_celsius'].max(),
                'avg_uhi_intensity': region_data['uhi_intensity'].mean(),
                'high_risk_areas': (region_data['heat_risk_category'].isin(['High', 'Very High'])).sum(),
                'very_high_risk_areas': (region_data['heat_risk_category'] == 'Very High').sum(),
                'avg_green_space_ratio': region_data['green_space_ratio'].mean(),
                'avg_cooling_potential': region_data['cooling_potential'].mean(),
                'priority_green_sites': len(high_priority_green[high_priority_green['region'] == region]),
                'priority_albedo_sites': len(high_priority_albedo[high_priority_albedo['region'] == region]),
                'priority_water_sites': len(high_priority_water[high_priority_water['region'] == region]),
                'estimated_affected_population': (region_data['population_density'] * 
                                                 (region_data['heat_risk_score'] > 0.5)).sum() / 1000  # thousands
            }
            regional_analysis.append(analysis)
    
    regional_df = pd.DataFrame(regional_analysis)
    
    # Temporal trend analysis (simulate multi-year data)
    print("Performing temporal heat trend analysis...")
    
    trend_results = {}
    for region in cfg['regions']:
        region_data = embeddings_df[embeddings_df['region'] == region]
        
        if len(region_data) > 0:
            # Simulate yearly temperature trends (general warming)
            yearly_temps = []
            years = list(range(2017, 2026))
            base_temp = region_data['lst_celsius'].mean()
            
            for year in years:
                # Add warming trend + random variation
                year_effect = (year - 2017) * 0.15  # 0.15°C per year warming
                temp = base_temp + year_effect + np.random.normal(0, 0.8)
                yearly_temps.append(temp)
            
            trend_stats = perform_trend_analysis(np.array(yearly_temps), np.array(years))
            trend_results[region] = trend_stats
            trend_results[region]['temperature_increase_9yr'] = yearly_temps[-1] - yearly_temps[0]
    
    # Create comprehensive visualizations
    print("Generating visualizations...")
    
    # 1. Urban heat overview
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # LST distribution by region
    sns.boxplot(data=embeddings_df, x='region', y='lst_celsius', ax=axes[0,0])
    axes[0,0].set_title('Land Surface Temperature by Region')
    axes[0,0].tick_params(axis='x', rotation=45)
    axes[0,0].set_ylabel('Temperature (°C)')
    
    # Heat risk distribution
    risk_counts = embeddings_df['heat_risk_category'].value_counts()
    axes[0,1].pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%')
    axes[0,1].set_title('Heat Risk Category Distribution')
    
    # UHI intensity vs built-up density
    axes[1,0].scatter(embeddings_df['built_up_density'], embeddings_df['uhi_intensity'], 
                     alpha=0.6, c=embeddings_df['green_space_ratio'], cmap='RdYlGn')
    axes[1,0].set_xlabel('Built-up Density')
    axes[1,0].set_ylabel('UHI Intensity (°C)')
    axes[1,0].set_title('UHI Intensity vs Urban Development')
    
    # Cooling potential analysis
    sns.scatterplot(data=embeddings_df, x='heat_risk_score', y='cooling_potential', 
                   hue='region', alpha=0.7, ax=axes[1,1])
    axes[1,1].set_xlabel('Heat Risk Score')
    axes[1,1].set_ylabel('Cooling Potential (°C)')
    axes[1,1].set_title('Heat Risk vs Cooling Potential')
    
    save_plot(fig, f"{figs}/urban_heat_overview_analysis.png", 
              "Urban Heat Island Overview Analysis - Uzbekistan")
    
    # 2. Spatial heat patterns
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # LST spatial distribution
    scatter1 = axes[0].scatter(embeddings_df['longitude'], embeddings_df['latitude'], 
                              c=embeddings_df['lst_celsius'], cmap='coolwarm', alpha=0.7, s=20)
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')
    axes[0].set_title('Land Surface Temperature Distribution')
    plt.colorbar(scatter1, ax=axes[0], label='Temperature (°C)')
    
    # Heat risk spatial distribution
    scatter2 = axes[1].scatter(embeddings_df['longitude'], embeddings_df['latitude'], 
                              c=embeddings_df['heat_risk_score'], cmap='Reds', alpha=0.7, s=20)
    axes[1].set_xlabel('Longitude')
    axes[1].set_ylabel('Latitude')
    axes[1].set_title('Heat Risk Score Distribution')
    plt.colorbar(scatter2, ax=axes[1], label='Risk Score')
    
    # Green space ratio
    scatter3 = axes[2].scatter(embeddings_df['longitude'], embeddings_df['latitude'], 
                              c=embeddings_df['green_space_ratio'], cmap='RdYlGn', alpha=0.7, s=20)
    axes[2].set_xlabel('Longitude')
    axes[2].set_ylabel('Latitude')
    axes[2].set_title('Green Space Ratio Distribution')
    plt.colorbar(scatter3, ax=axes[2], label='Green Space Ratio')
    
    save_plot(fig, f"{figs}/urban_heat_spatial_patterns.png", 
              "Urban Heat Spatial Patterns - Uzbekistan")
    
    # 3. Mitigation strategies visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Priority areas for different strategies
    strategy_data = pd.DataFrame({
        'Strategy': ['Green Infrastructure', 'Cool Surfaces', 'Water Features'],
        'Priority_Sites': [len(high_priority_green), len(high_priority_albedo), len(high_priority_water)]
    })
    
    sns.barplot(data=strategy_data, x='Strategy', y='Priority_Sites', ax=axes[0,0])
    axes[0,0].set_title('Priority Sites by Mitigation Strategy')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Feature importance for LST prediction
    feature_importance = pd.DataFrame({
        'feature': all_features,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False).head(10)
    
    sns.barplot(data=feature_importance, x='importance', y='feature', ax=axes[0,1])
    axes[0,1].set_title('Top 10 Features for LST Prediction')
    
    # Green space vs temperature
    axes[1,0].scatter(embeddings_df['green_space_ratio'], embeddings_df['lst_celsius'], 
                     alpha=0.6, c=embeddings_df['built_up_density'], cmap='viridis')
    axes[1,0].set_xlabel('Green Space Ratio')
    axes[1,0].set_ylabel('Land Surface Temperature (°C)')
    axes[1,0].set_title('Green Space vs Temperature')
    
    # Model performance
    axes[1,1].scatter(y_test, y_pred, alpha=0.6)
    axes[1,1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[1,1].set_xlabel('Actual Temperature (°C)')
    axes[1,1].set_ylabel('Predicted Temperature (°C)')
    axes[1,1].set_title(f'Model Performance (R² = {r2:.3f})')
    
    save_plot(fig, f"{figs}/urban_heat_mitigation_analysis.png", 
              "Urban Heat Mitigation Analysis - Uzbekistan")
    
    # Generate comprehensive data tables
    print("Generating data tables...")
    
    # 1. Regional analysis
    regional_df.to_csv(f"{tables}/urban_heat_regional_analysis.csv", index=False)
    
    # 2. Heat risk assessment
    heat_risk_summary = embeddings_df.groupby(['region', 'heat_risk_category']).size().unstack(fill_value=0)
    heat_risk_summary.to_csv(f"{tables}/urban_heat_risk_assessment.csv")
    
    # 3. Model performance
    model_performance = pd.DataFrame({
        'metric': ['R²', 'RMSE', 'Mean_Temperature', 'Std_Temperature'],
        'value': [r2, rmse, embeddings_df['lst_celsius'].mean(), embeddings_df['lst_celsius'].std()]
    })
    model_performance.to_csv(f"{tables}/urban_heat_model_performance.csv", index=False)
    
    # 4. Mitigation priorities
    mitigation_summary = pd.DataFrame({
        'strategy': ['Green Infrastructure', 'Cool Surfaces', 'Water Features'],
        'priority_sites': [len(high_priority_green), len(high_priority_albedo), len(high_priority_water)],
        'avg_current_risk': [
            high_priority_green['heat_risk_score'].mean() if len(high_priority_green) > 0 else 0,
            high_priority_albedo['heat_risk_score'].mean() if len(high_priority_albedo) > 0 else 0,
            high_priority_water['heat_risk_score'].mean() if len(high_priority_water) > 0 else 0
        ],
        'avg_cooling_potential': [
            high_priority_green['cooling_potential'].mean() if len(high_priority_green) > 0 else 0,
            high_priority_albedo['cooling_potential'].mean() if len(high_priority_albedo) > 0 else 0,
            high_priority_water['cooling_potential'].mean() if len(high_priority_water) > 0 else 0
        ]
    })
    mitigation_summary.to_csv(f"{tables}/urban_heat_mitigation_priorities.csv", index=False)
    
    # 5. Temperature trends
    if trend_results:
        trend_df = pd.DataFrame.from_dict(trend_results, orient='index').reset_index()
        trend_df.rename(columns={'index': 'region'}, inplace=True)
        trend_df.to_csv(f"{tables}/urban_heat_trend_analysis.csv", index=False)
    
    # 6. Detailed heat scores
    heat_scores = embeddings_df.groupby('region').agg({
        'lst_celsius': ['mean', 'max', 'std'],
        'uhi_intensity': ['mean', 'max'],
        'heat_risk_score': ['mean', 'std'],
        'cooling_potential': ['mean', 'max'],
        'green_space_ratio': 'mean',
        'population_density': 'mean'
    }).round(2)
    
    heat_scores.columns = ['_'.join(col).strip() for col in heat_scores.columns]
    heat_scores = heat_scores.reset_index()
    heat_scores.to_csv(f"{tables}/urban_heat_scores.csv", index=False)
    
    # 7. Feature importance
    feature_importance.to_csv(f"{tables}/urban_heat_feature_importance.csv", index=False)
    
    # Generate executive summary statistics
    avg_lst = embeddings_df['lst_celsius'].mean()
    max_uhi = embeddings_df['uhi_intensity'].max()
    high_risk_areas = (embeddings_df['heat_risk_category'].isin(['High', 'Very High'])).sum()
    total_cooling_potential = embeddings_df['cooling_potential'].sum()
    
    # Focus on Tashkent as primary urban center
    tashkent_data = embeddings_df[embeddings_df['region'] == 'Tashkent']
    tashkent_avg_temp = tashkent_data['lst_celsius'].mean() if len(tashkent_data) > 0 else avg_lst
    
    print("Urban heat island analysis completed successfully!")
    print(f"Key findings:")
    print(f"  - Average land surface temperature: {avg_lst:.1f}°C")
    print(f"  - Maximum UHI intensity: {max_uhi:.1f}°C")
    print(f"  - High/very high risk areas: {high_risk_areas}")
    print(f"  - Total cooling potential: {total_cooling_potential:.1f}°C")
    print(f"  - Tashkent average temperature: {tashkent_avg_temp:.1f}°C")
    
    artifacts = [
        "tables/urban_heat_regional_analysis.csv",
        "tables/urban_heat_risk_assessment.csv",
        "tables/urban_heat_model_performance.csv",
        "tables/urban_heat_mitigation_priorities.csv",
        "tables/urban_heat_trend_analysis.csv",
        "tables/urban_heat_scores.csv",
        "tables/urban_heat_feature_importance.csv",
        "figs/urban_heat_overview_analysis.png",
        "figs/urban_heat_spatial_patterns.png",
        "figs/urban_heat_mitigation_analysis.png"
    ]
    
    return {
        "status": "ok",
        "artifacts": artifacts,
        "summary_stats": {
            "avg_lst": float(avg_lst),
            "max_uhi_intensity": float(max_uhi),
            "high_risk_areas": int(high_risk_areas),
            "total_cooling_potential": float(total_cooling_potential),
            "tashkent_avg_temp": float(tashkent_avg_temp),
            "model_r2": float(r2),
            "model_rmse": float(rmse)
        }
    }
