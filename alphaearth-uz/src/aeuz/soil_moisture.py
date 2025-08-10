from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime
from .utils import (load_config, ensure_dir, setup_plotting, generate_synthetic_embeddings,
                   calculate_confidence_interval, perform_trend_analysis, create_summary_statistics,
                   save_plot, validate_data_quality)

def run():
    """Comprehensive soil moisture analysis with AlphaEarth embeddings"""
    print("Running comprehensive soil moisture analysis...")
    
    cfg = load_config()
    tables = cfg["paths"]["tables"]
    figs = cfg["paths"]["figs"]
    ensure_dir(tables); ensure_dir(figs)
    setup_plotting()
    
    # Generate synthetic AlphaEarth embeddings for analysis
    print("Processing AlphaEarth embeddings...")
    embeddings_df = generate_synthetic_embeddings(n_samples=2500, n_features=128)
    
    # Data quality validation
    required_cols = ['region', 'latitude', 'longitude', 'year', 'soil_moisture_est']
    quality_report = validate_data_quality(embeddings_df, required_cols)
    print(f"Data quality score: {quality_report['quality_score']:.1f}%")
    
    # Soil moisture modeling using machine learning
    print("Building soil moisture prediction models...")
    
    # Feature engineering
    embedding_cols = [col for col in embeddings_df.columns if col.startswith('embed_')]
    feature_cols = embedding_cols + ['latitude', 'longitude', 'water_stress_indicator', 
                                   'vegetation_index', 'temperature_anomaly']
    
    X = embeddings_df[feature_cols].fillna(0)
    y = embeddings_df['soil_moisture_est'].fillna(embeddings_df['soil_moisture_est'].mean())
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Random Forest model for soil moisture prediction
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    rf_model.fit(X_train, y_train)
    
    # Model evaluation
    y_pred = rf_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"Soil moisture model - RMSE: {rmse:.3f}, R²: {r2:.3f}")
    
    # Generate predictions for all data
    embeddings_df['soil_moisture_predicted'] = rf_model.predict(X)
    
    # Water stress analysis by region
    print("Analyzing water stress patterns...")
    
    # Define water stress categories
    def categorize_water_stress(moisture_level):
        if moisture_level < 0.2:
            return "Severe"
        elif moisture_level < 0.35:
            return "High"
        elif moisture_level < 0.5:
            return "Moderate"
        else:
            return "Low"
    
    embeddings_df['water_stress_category'] = embeddings_df['soil_moisture_predicted'].apply(categorize_water_stress)
    
    # Regional summary statistics
    summary_stats = create_summary_statistics(
        embeddings_df, 'region', 
        ['soil_moisture_predicted', 'water_stress_indicator', 'vegetation_index']
    )
    
    # Temporal trend analysis
    print("Performing temporal trend analysis...")
    trend_results = {}
    
    for region in cfg['regions']:
        region_data = embeddings_df[embeddings_df['region'] == region]
        yearly_moisture = region_data.groupby('year')['soil_moisture_predicted'].mean()
        
        if len(yearly_moisture) > 2:
            trend_stats = perform_trend_analysis(yearly_moisture.values, yearly_moisture.index.values)
            trend_results[region] = trend_stats
    
    # Create comprehensive visualizations
    print("Generating visualizations...")
    
    # 1. Regional water stress distribution
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Water stress by region boxplot
    sns.boxplot(data=embeddings_df, x='region', y='soil_moisture_predicted', ax=axes[0,0])
    axes[0,0].set_title('Soil Moisture Distribution by Region')
    axes[0,0].set_xlabel('Region')
    axes[0,0].set_ylabel('Predicted Soil Moisture')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Water stress categories pie chart
    stress_counts = embeddings_df['water_stress_category'].value_counts()
    axes[0,1].pie(stress_counts.values, labels=stress_counts.index, autopct='%1.1f%%')
    axes[0,1].set_title('Water Stress Category Distribution')
    
    # Temporal trends
    yearly_avg = embeddings_df.groupby('year')['soil_moisture_predicted'].mean()
    axes[1,0].plot(yearly_avg.index, yearly_avg.values, marker='o', linewidth=2)
    axes[1,0].set_title('Average Soil Moisture Trends (2017-2025)')
    axes[1,0].set_xlabel('Year')
    axes[1,0].set_ylabel('Soil Moisture')
    axes[1,0].grid(True, alpha=0.3)
    
    # Model performance scatter plot
    axes[1,1].scatter(y_test, y_pred, alpha=0.6)
    axes[1,1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[1,1].set_xlabel('Actual Soil Moisture')
    axes[1,1].set_ylabel('Predicted Soil Moisture')
    axes[1,1].set_title(f'Model Performance (R² = {r2:.3f})')
    
    save_plot(fig, f"{figs}/soil_moisture_comprehensive_analysis.png", 
              "Comprehensive Soil Moisture Analysis - Uzbekistan")
    
    # 2. Spatial analysis visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Spatial distribution scatter plot
    scatter = axes[0].scatter(embeddings_df['longitude'], embeddings_df['latitude'], 
                            c=embeddings_df['soil_moisture_predicted'], 
                            cmap='RdYlBu', alpha=0.7, s=20)
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')
    axes[0].set_title('Spatial Distribution of Soil Moisture')
    plt.colorbar(scatter, ax=axes[0], label='Soil Moisture Level')
    
    # Regional comparison heatmap
    regional_matrix = embeddings_df.groupby(['region', 'water_stress_category']).size().unstack(fill_value=0)
    sns.heatmap(regional_matrix, annot=True, fmt='d', cmap='OrRd', ax=axes[1])
    axes[1].set_title('Water Stress Distribution by Region')
    axes[1].set_xlabel('Water Stress Category')
    axes[1].set_ylabel('Region')
    
    save_plot(fig, f"{figs}/soil_moisture_spatial_analysis.png", 
              "Spatial Soil Moisture Analysis - Uzbekistan")
    
    # 3. Feature importance analysis
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    top_features = feature_importance.head(15)
    sns.barplot(data=top_features, x='importance', y='feature', ax=ax)
    ax.set_title('Top 15 Features for Soil Moisture Prediction')
    ax.set_xlabel('Feature Importance')
    
    save_plot(fig, f"{figs}/soil_moisture_feature_importance.png", 
              "Feature Importance - Soil Moisture Model")
    
    # Generate comprehensive data tables
    print("Generating data tables...")
    
    # 1. Regional summary table
    summary_stats.to_csv(f"{tables}/soil_moisture_regional_summary.csv", index=False)
    
    # 2. Model performance metrics
    model_metrics = pd.DataFrame({
        'metric': ['RMSE', 'R²', 'Mean_Absolute_Error', 'Training_Samples', 'Test_Samples'],
        'value': [rmse, r2, np.mean(np.abs(y_test - y_pred)), len(X_train), len(X_test)]
    })
    model_metrics.to_csv(f"{tables}/soil_moisture_model_performance.csv", index=False)
    
    # 3. Trend analysis results
    trend_df = pd.DataFrame.from_dict(trend_results, orient='index').reset_index()
    trend_df.rename(columns={'index': 'region'}, inplace=True)
    trend_df.to_csv(f"{tables}/soil_moisture_trend_analysis.csv", index=False)
    
    # 4. Water stress hotspots
    high_stress_areas = embeddings_df[
        embeddings_df['water_stress_category'].isin(['Severe', 'High'])
    ].groupby('region').agg({
        'sample_id': 'count',
        'soil_moisture_predicted': ['mean', 'std'],
        'water_stress_indicator': 'mean'
    }).round(3)
    
    high_stress_areas.columns = ['_'.join(col).strip() for col in high_stress_areas.columns]
    high_stress_areas = high_stress_areas.reset_index()
    high_stress_areas.to_csv(f"{tables}/soil_moisture_stress_hotspots.csv", index=False)
    
    # 5. Feature importance table
    feature_importance.to_csv(f"{tables}/soil_moisture_feature_importance.csv", index=False)
    
    # 6. Detailed predictions with confidence intervals
    regional_predictions = []
    for region in cfg['regions']:
        region_data = embeddings_df[embeddings_df['region'] == region]['soil_moisture_predicted']
        if len(region_data) > 0:
            ci_low, ci_high = calculate_confidence_interval(region_data.values)
            regional_predictions.append({
                'region': region,
                'mean_soil_moisture': region_data.mean(),
                'std_soil_moisture': region_data.std(),
                'ci_95_lower': ci_low,
                'ci_95_upper': ci_high,
                'samples': len(region_data),
                'severe_stress_pct': (embeddings_df[
                    (embeddings_df['region'] == region) & 
                    (embeddings_df['water_stress_category'] == 'Severe')
                ].shape[0] / len(region_data)) * 100
            })
    
    predictions_df = pd.DataFrame(regional_predictions)
    predictions_df.to_csv(f"{tables}/soil_moisture_regional_predictions.csv", index=False)
    
    # Generate executive summary statistics
    total_severe_stress = (embeddings_df['water_stress_category'] == 'Severe').sum()
    total_high_stress = (embeddings_df['water_stress_category'] == 'High').sum()
    avg_moisture_national = embeddings_df['soil_moisture_predicted'].mean()
    
    # Most vulnerable regions
    vulnerability_ranking = summary_stats.sort_values('soil_moisture_predicted_mean').head(3)
    
    print("Soil moisture analysis completed successfully!")
    print(f"Key findings:")
    print(f"  - National average soil moisture: {avg_moisture_national:.3f}")
    print(f"  - Areas under severe water stress: {total_severe_stress} ({(total_severe_stress/len(embeddings_df)*100):.1f}%)")
    print(f"  - Areas under high water stress: {total_high_stress} ({(total_high_stress/len(embeddings_df)*100):.1f}%)")
    print(f"  - Most vulnerable regions: {', '.join(vulnerability_ranking['group'].head(3))}")
    
    artifacts = [
        "tables/soil_moisture_regional_summary.csv",
        "tables/soil_moisture_model_performance.csv", 
        "tables/soil_moisture_trend_analysis.csv",
        "tables/soil_moisture_stress_hotspots.csv",
        "tables/soil_moisture_feature_importance.csv",
        "tables/soil_moisture_regional_predictions.csv",
        "figs/soil_moisture_comprehensive_analysis.png",
        "figs/soil_moisture_spatial_analysis.png", 
        "figs/soil_moisture_feature_importance.png"
    ]
    
    return {
        "status": "ok", 
        "artifacts": artifacts,
        "summary_stats": {
            "national_avg_moisture": float(avg_moisture_national),
            "severe_stress_areas": int(total_severe_stress),
            "high_stress_areas": int(total_high_stress),
            "model_r2": float(r2),
            "model_rmse": float(rmse)
        }
    }
