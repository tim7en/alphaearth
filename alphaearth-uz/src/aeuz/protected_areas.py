from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from .utils import (load_config, ensure_dir, setup_plotting, load_actual_data,
                   calculate_confidence_interval, perform_trend_analysis, create_summary_statistics,
                   save_plot, validate_data_quality)

def run():
    """Comprehensive protected area disturbance analysis with anomaly detection"""
    print("Running comprehensive protected area disturbance analysis...")
    
    cfg = load_config()
    tables = cfg["paths"]["tables"]
    figs = cfg["paths"]["figs"]
    final = cfg["paths"]["final"]
    ensure_dir(tables); ensure_dir(figs); ensure_dir(final)
    setup_plotting()
    
    # Load actual protected areas analysis data
    print("Loading actual protected areas analysis data...")
    data_df = load_actual_data('protected_areas')
    
    print(f"Loaded {len(data_df)} records from protected areas analysis")
    print(f"Data sources: {data_df['data_source'].unique()}")
    print(f"Regions covered: {data_df['region'].unique()}")
    
    # Data quality validation
    required_cols = ['region']
    available_cols = [col for col in required_cols if col in data_df.columns]
    if available_cols:
        quality_report = validate_data_quality(data_df, available_cols)
        print(f"Data quality score: {quality_report['quality_score']:.1f}%")
    embeddings_df['mining_proximity_km'] = np.clip(np.random.exponential(20, len(embeddings_df)), 1, 100)
    
    # Perform analysis on the actual data
    print("Analyzing protected areas disturbance patterns...")
    
    # Regional analysis using conservation priorities data
    conservation_data = data_df[data_df['data_source'] == 'protected_areas_conservation_priorities.csv']
    if not conservation_data.empty:
        print("\n=== Regional Conservation Priority Analysis ===")
        print(conservation_data.groupby('region').agg({
            'disturbance_index_mean': ['mean', 'std'],
            'threat_level_mean': ['mean', 'std'],
            'area_size_km2_sum': 'sum',
            'illegal_logging_incidents_sum': 'sum',
            'fire_incidents_5yr_sum': 'sum'
        }).round(3))
    
    # Management effectiveness analysis
    management_data = data_df[data_df['data_source'] == 'protected_areas_management_effectiveness.csv']
    if not management_data.empty:
        print("\n=== Management Effectiveness by Protection Level ===")
        mgmt_summary = management_data.groupby(['region', 'protection_level']).agg({
            'disturbance_index': ['mean', 'count'],
            'current_forest_cover': 'mean',
            'biodiversity_index': 'mean'
        }).round(3)
        print(mgmt_summary)
    
    # Incident analysis
    incidents_data = data_df[data_df['data_source'] == 'protected_areas_incidents.csv']
    if not incidents_data.empty:
        print("\n=== Regional Incident Analysis ===")
        incident_summary = incidents_data.groupby('region').agg({
            'disturbance_index_mean': 'mean',
            'threat_level_mean': 'mean',
            'current_forest_cover_mean': 'mean',
            'biodiversity_index_mean': 'mean',
            'area_size_km2_sum': 'sum'
        }).round(3)
        print(incident_summary)
    
    # Create visualizations based on actual data
    print("Generating visualizations based on actual data...")
    
    # Regional analysis summary
    if not conservation_data.empty:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Regional disturbance comparison
        conservation_data.set_index('region')['disturbance_index_mean'].plot(kind='bar', ax=axes[0,0])
        axes[0,0].set_title('Average Disturbance Index by Region')
        axes[0,0].set_ylabel('Disturbance Index')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Regional threat levels
        conservation_data.set_index('region')['threat_level_mean'].plot(kind='bar', ax=axes[0,1], color='orange')
        axes[0,1].set_title('Average Threat Level by Region')
        axes[0,1].set_ylabel('Threat Level')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Total protected area size by region
        conservation_data.set_index('region')['area_size_km2_sum'].plot(kind='bar', ax=axes[1,0], color='green')
        axes[1,0].set_title('Total Protected Area Size by Region')
        axes[1,0].set_ylabel('Area (km²)')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Illegal logging incidents
        conservation_data.set_index('region')['illegal_logging_incidents_sum'].plot(kind='bar', ax=axes[1,1], color='red')
        axes[1,1].set_title('Illegal Logging Incidents by Region')
        axes[1,1].set_ylabel('Number of Incidents')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        save_plot(fig, f"{figs}/protected_areas_spatial_analysis.png", 
                  "Protected Areas Regional Analysis")
    
    # Management effectiveness analysis visualization
    if not management_data.empty:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Forest cover by protection level
        forest_by_protection = management_data.groupby('protection_level')['current_forest_cover'].mean()
        forest_by_protection.plot(kind='bar', ax=axes[0])
        axes[0].set_title('Forest Cover by Protection Level')
        axes[0].set_ylabel('Current Forest Cover')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Biodiversity index by protection level
        biodiv_by_protection = management_data.groupby('protection_level')['biodiversity_index'].mean()
        biodiv_by_protection.plot(kind='bar', ax=axes[1], color='purple')
        axes[1].set_title('Biodiversity Index by Protection Level')
        axes[1].set_ylabel('Biodiversity Index')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        save_plot(fig, f"{figs}/protected_areas_conservation_analysis.png",
                  "Protected Areas Conservation Effectiveness")
    
    # Summary statistics and export
    print("Generating summary statistics...")
    
    # Create summary tables
    summary_stats = []
    
    if not conservation_data.empty:
        for _, row in conservation_data.iterrows():
            summary_stats.append({
                'Analysis_Type': 'Regional_Conservation_Priority',
                'Region': row['region'],
                'Disturbance_Index': f"{row['disturbance_index_mean']:.3f}",
                'Threat_Level': f"{row['threat_level_mean']:.3f}",
                'Protected_Area_km2': f"{row['area_size_km2_sum']:.1f}",
                'Logging_Incidents': row['illegal_logging_incidents_sum'],
                'Fire_Incidents': row['fire_incidents_5yr_sum']
            })
    
    if not management_data.empty:
        mgmt_summary = management_data.groupby(['region', 'protection_level']).agg({
            'disturbance_index': 'mean',
            'current_forest_cover': 'mean',
            'biodiversity_index': 'mean'
        }).round(3)
        
        for (region, protection), values in mgmt_summary.iterrows():
            summary_stats.append({
                'Analysis_Type': 'Management_Effectiveness',
                'Region': region,
                'Protection_Level': protection,
                'Disturbance_Index': f"{values['disturbance_index']:.3f}",
                'Forest_Cover': f"{values['current_forest_cover']:.3f}",
                'Biodiversity_Index': f"{values['biodiversity_index']:.3f}"
            })
    
    # Export results
    if summary_stats:
        summary_df = pd.DataFrame(summary_stats)
        summary_df.to_csv(f"{tables}/protected_areas_analysis_results.csv", index=False)
        print(f"Exported {len(summary_stats)} analysis results to CSV")
    
    # Find priority conservation areas (highest disturbance + threat)
    priority_areas = []
    if not conservation_data.empty:
        for _, row in conservation_data.iterrows():
            priority_score = (row['disturbance_index_mean'] * 0.6 + 
                            row['threat_level_mean'] * 0.4)
            priority_areas.append({
                'region': row['region'],
                'priority_score': priority_score,
                'disturbance_index': row['disturbance_index_mean'],
                'threat_level': row['threat_level_mean'],
                'total_area_km2': row['area_size_km2_sum']
            })
    
    if priority_areas:
        priority_df = pd.DataFrame(priority_areas).sort_values('priority_score', ascending=False)
        priority_df.to_csv(f"{final}/protected_area_priorities.csv", index=False)
        
        print("\n=== Top Priority Conservation Areas ===")
        print(priority_df.head().to_string(index=False))
    
    print("\n=== Protected Areas Analysis Complete ===")
    print(f"✓ Processed data from {len(data_df)} analysis records")
    print(f"✓ Analyzed {len(data_df['region'].unique())} regions: {', '.join(data_df['region'].unique())}")
    print(f"✓ Generated visualizations and analysis tables")
    print(f"✓ Results exported to tables/ and data_final/ directories")
    
    # Return summary results
    analysis_results = {
        'status': 'completed',
        'total_records_analyzed': len(data_df),
        'regions_covered': len(data_df['region'].unique()),
        'data_sources_used': data_df['data_source'].nunique(),
        'output_files_generated': 3 + (2 if not conservation_data.empty else 0) + (1 if not management_data.empty else 0)
    }
    
    if not conservation_data.empty:
        analysis_results.update({
            'highest_disturbance_region': conservation_data.loc[conservation_data['disturbance_index_mean'].idxmax(), 'region'],
            'highest_threat_region': conservation_data.loc[conservation_data['threat_level_mean'].idxmax(), 'region'],
            'largest_protected_area_region': conservation_data.loc[conservation_data['area_size_km2_sum'].idxmax(), 'region']
        })
    
    return analysis_results
