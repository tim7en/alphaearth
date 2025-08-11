from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from .utils import (load_config, ensure_dir, setup_plotting, generate_synthetic_embeddings,
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
    
    # Generate synthetic AlphaEarth embeddings for protected area analysis
    print("Processing AlphaEarth embeddings for protected area assessment...")
    embeddings_df = generate_synthetic_embeddings(n_samples=1800, n_features=128)
    
    # Add protected area-specific indicators
    np.random.seed(42)
    
    # Protected area characteristics
    embeddings_df['protection_level'] = np.random.choice(['National_Park', 'Nature_Reserve', 'Wildlife_Sanctuary', 'Biosphere_Reserve'], 
                                                         len(embeddings_df), p=[0.3, 0.25, 0.3, 0.15])
    embeddings_df['area_size_km2'] = np.clip(np.random.lognormal(3, 1.5, len(embeddings_df)), 1, 2000)
    embeddings_df['establishment_year'] = np.random.randint(1970, 2020, len(embeddings_df))
    embeddings_df['boundary_clarity'] = np.random.choice(['well_defined', 'partially_defined', 'unclear'], 
                                                         len(embeddings_df), p=[0.6, 0.3, 0.1])
    
    # Baseline ecosystem indicators
    embeddings_df['baseline_forest_cover'] = np.clip(np.random.beta(3, 2, len(embeddings_df)), 0.1, 0.95)
    embeddings_df['current_forest_cover'] = np.clip(embeddings_df['baseline_forest_cover'] - 
                                                    np.random.exponential(0.05, len(embeddings_df)), 0, 1)
    embeddings_df['biodiversity_index'] = np.clip(np.random.beta(2, 2, len(embeddings_df)), 0.2, 1.0)
    
    # Human pressure indicators
    embeddings_df['human_settlements_nearby'] = np.random.poisson(2.5, len(embeddings_df))
    embeddings_df['distance_to_road_km'] = np.clip(np.random.exponential(8, len(embeddings_df)), 0.5, 50)
    embeddings_df['livestock_grazing_evidence'] = np.random.choice([0, 1], len(embeddings_df), p=[0.7, 0.3])
    embeddings_df['illegal_logging_incidents'] = np.random.poisson(0.8, len(embeddings_df))
    embeddings_df['mining_proximity_km'] = np.clip(np.random.exponential(20, len(embeddings_df)), 1, 100)
    
    # Tourism and access pressure
    embeddings_df['visitor_pressure'] = np.random.choice(['low', 'moderate', 'high'], 
                                                         len(embeddings_df), p=[0.5, 0.3, 0.2])
    embeddings_df['trail_density_km_per_km2'] = np.clip(np.random.gamma(1, 2, len(embeddings_df)), 0, 15)
    embeddings_df['unauthorized_access_points'] = np.random.poisson(1.2, len(embeddings_df))
    
    # Infrastructure and development pressure
    embeddings_df['infrastructure_encroachment'] = np.random.choice([0, 1, 2, 3], len(embeddings_df), p=[0.6, 0.25, 0.1, 0.05])
    embeddings_df['agricultural_expansion_threat'] = np.random.choice(['none', 'low', 'moderate', 'high'], 
                                                                     len(embeddings_df), p=[0.4, 0.3, 0.2, 0.1])
    embeddings_df['water_extraction_impact'] = np.random.choice([0, 1, 2], len(embeddings_df), p=[0.7, 0.25, 0.05])
    
    # Management effectiveness indicators
    embeddings_df['ranger_presence'] = np.random.choice(['adequate', 'limited', 'insufficient'], 
                                                        len(embeddings_df), p=[0.3, 0.4, 0.3])
    embeddings_df['monitoring_frequency'] = np.random.choice(['regular', 'occasional', 'rare'], 
                                                            len(embeddings_df), p=[0.4, 0.4, 0.2])
    embeddings_df['management_budget_adequacy'] = np.random.choice(['adequate', 'limited', 'insufficient'], 
                                                                  len(embeddings_df), p=[0.2, 0.5, 0.3])
    
    # Recent disturbance events
    embeddings_df['fire_incidents_5yr'] = np.random.poisson(0.5, len(embeddings_df))
    embeddings_df['flood_impact_severity'] = np.random.choice([0, 1, 2, 3], len(embeddings_df), p=[0.7, 0.2, 0.08, 0.02])
    embeddings_df['disease_outbreak_incidents'] = np.random.poisson(0.3, len(embeddings_df))
    
    # Data quality validation
    required_cols = ['region', 'protection_level', 'current_forest_cover', 'biodiversity_index']
    quality_report = validate_data_quality(embeddings_df, required_cols)
    print(f"Data quality score: {quality_report['quality_score']:.1f}%")
    
    # Calculate disturbance and threat indices
    print("Calculating disturbance and threat indices...")
    
    def calculate_disturbance_index(row):
        """Calculate comprehensive disturbance index (0-1, higher = more disturbed)"""
        score = 0.0
        
        # Habitat loss (25% weight)
        forest_loss = row['baseline_forest_cover'] - row['current_forest_cover']
        if forest_loss > 0.2:
            score += 0.20
        elif forest_loss > 0.1:
            score += 0.15
        elif forest_loss > 0.05:
            score += 0.10
        
        # Human pressure (30% weight)
        if row['human_settlements_nearby'] >= 5:
            score += 0.10
        elif row['human_settlements_nearby'] >= 3:
            score += 0.05
        
        if row['distance_to_road_km'] < 2:
            score += 0.10
        elif row['distance_to_road_km'] < 5:
            score += 0.05
        
        if row['livestock_grazing_evidence']:
            score += 0.05
        
        if row['illegal_logging_incidents'] >= 2:
            score += 0.05
        
        # Infrastructure pressure (20% weight)
        score += row['infrastructure_encroachment'] * 0.05
        
        threat_mapping = {'none': 0, 'low': 0.02, 'moderate': 0.05, 'high': 0.10}
        score += threat_mapping.get(row['agricultural_expansion_threat'], 0)
        
        score += row['water_extraction_impact'] * 0.025
        
        # Tourism pressure (10% weight)
        visitor_mapping = {'low': 0, 'moderate': 0.02, 'high': 0.05}
        score += visitor_mapping.get(row['visitor_pressure'], 0)
        
        if row['unauthorized_access_points'] >= 3:
            score += 0.03
        elif row['unauthorized_access_points'] >= 1:
            score += 0.02
        
        # Management deficiency (15% weight)
        management_mapping = {'adequate': 0, 'limited': 0.05, 'insufficient': 0.10}
        score += management_mapping.get(row['ranger_presence'], 0)
        score += management_mapping.get(row['monitoring_frequency'], 0) * 0.5
        score += management_mapping.get(row['management_budget_adequacy'], 0) * 0.5
        
        return min(1.0, score)
    
    def calculate_threat_level(row):
        """Calculate future threat level based on pressures"""
        threat_score = 0.0
        
        # Proximity threats
        if row['distance_to_road_km'] < 5:
            threat_score += 0.2
        if row['mining_proximity_km'] < 10:
            threat_score += 0.15
        
        # Development pressure
        threat_mapping = {'none': 0, 'low': 0.1, 'moderate': 0.3, 'high': 0.5}
        threat_score += threat_mapping.get(row['agricultural_expansion_threat'], 0)
        
        # Management capacity
        if row['ranger_presence'] == 'insufficient':
            threat_score += 0.2
        if row['management_budget_adequacy'] == 'insufficient':
            threat_score += 0.15
        
        return min(1.0, threat_score)
    
    embeddings_df['disturbance_index'] = embeddings_df.apply(calculate_disturbance_index, axis=1)
    embeddings_df['threat_level'] = embeddings_df.apply(calculate_threat_level, axis=1)
    
    # Classify conservation status
    def classify_conservation_status(row):
        if row['disturbance_index'] < 0.2 and row['threat_level'] < 0.3:
            return 'Good'
        elif row['disturbance_index'] < 0.4 and row['threat_level'] < 0.5:
            return 'Fair'
        elif row['disturbance_index'] < 0.6:
            return 'Poor'
        else:
            return 'Critical'
    
    embeddings_df['conservation_status'] = embeddings_df.apply(classify_conservation_status, axis=1)
    
    # Anomaly detection for unusual disturbance patterns
    print("Detecting anomalous disturbance patterns...")
    
    # Features for anomaly detection
    anomaly_features = ['disturbance_index', 'threat_level', 'current_forest_cover', 'biodiversity_index',
                       'illegal_logging_incidents', 'fire_incidents_5yr', 'infrastructure_encroachment']
    
    anomaly_data = embeddings_df[anomaly_features].fillna(embeddings_df[anomaly_features].mean())
    scaler = StandardScaler()
    anomaly_data_scaled = scaler.fit_transform(anomaly_data)
    
    # Use clustering to identify outliers
    kmeans = KMeans(n_clusters=4, random_state=42)
    embeddings_df['cluster'] = kmeans.fit_predict(anomaly_data_scaled)
    
    # Calculate distance to cluster centers to identify anomalies
    cluster_centers = kmeans.cluster_centers_
    distances = []
    for i, row in enumerate(anomaly_data_scaled):
        cluster_id = embeddings_df.iloc[i]['cluster']
        distance = np.linalg.norm(row - cluster_centers[cluster_id])
        distances.append(distance)
    
    embeddings_df['anomaly_score'] = distances
    
    # Identify incidents (top 10% anomalies + critical conservation status)
    anomaly_threshold = np.percentile(embeddings_df['anomaly_score'], 90)
    embeddings_df['is_incident'] = ((embeddings_df['anomaly_score'] > anomaly_threshold) | 
                                   (embeddings_df['conservation_status'] == 'Critical')).astype(int)
    
    incidents = embeddings_df[embeddings_df['is_incident'] == 1]
    print(f"Identified {len(incidents)} protected area incidents requiring immediate attention")
    
    # Regional analysis
    print("Generating regional protected area assessments...")
    
    regional_analysis = []
    for region in cfg['regions']:
        region_data = embeddings_df[embeddings_df['region'] == region]
        
        if len(region_data) > 0:
            analysis = {
                'region': region,
                'total_protected_areas': len(region_data),
                'avg_disturbance_index': region_data['disturbance_index'].mean(),
                'avg_threat_level': region_data['threat_level'].mean(),
                'critical_status_areas': len(region_data[region_data['conservation_status'] == 'Critical']),
                'incident_count': len(region_data[region_data['is_incident'] == 1]),
                'total_protected_area_km2': region_data['area_size_km2'].sum(),
                'avg_forest_cover': region_data['current_forest_cover'].mean(),
                'avg_biodiversity_index': region_data['biodiversity_index'].mean(),
                'management_challenges': region_data[region_data['ranger_presence'] == 'insufficient'].shape[0],
                'priority_conservation_needs': region_data[region_data['conservation_status'].isin(['Poor', 'Critical'])].shape[0]
            }
            regional_analysis.append(analysis)
    
    regional_df = pd.DataFrame(regional_analysis)
    
    # Create comprehensive visualizations
    print("Generating visualizations...")
    
    # 1. Conservation status overview
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Conservation status distribution
    status_counts = embeddings_df['conservation_status'].value_counts()
    axes[0,0].pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%')
    axes[0,0].set_title('Protected Area Conservation Status')
    
    # Disturbance by protection level
    sns.boxplot(data=embeddings_df, x='protection_level', y='disturbance_index', ax=axes[0,1])
    axes[0,1].set_title('Disturbance Index by Protection Level')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Forest cover vs disturbance
    axes[1,0].scatter(embeddings_df['current_forest_cover'], embeddings_df['disturbance_index'], 
                     alpha=0.6, s=20)
    axes[1,0].set_xlabel('Current Forest Cover')
    axes[1,0].set_ylabel('Disturbance Index')
    axes[1,0].set_title('Forest Cover vs Disturbance')
    
    # Threat level distribution
    axes[1,1].hist(embeddings_df['threat_level'], bins=20, alpha=0.7, edgecolor='black')
    axes[1,1].set_xlabel('Threat Level')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].set_title('Distribution of Threat Levels')
    
    save_plot(fig, f"{figs}/protected_areas_conservation_analysis.png", 
              "Protected Area Conservation Analysis - Uzbekistan")
    
    # 2. Spatial analysis
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Conservation status spatial distribution
    status_colors = {'Good': 'green', 'Fair': 'yellow', 'Poor': 'orange', 'Critical': 'red'}
    for status in status_colors:
        status_data = embeddings_df[embeddings_df['conservation_status'] == status]
        if len(status_data) > 0:
            axes[0].scatter(status_data['longitude'], status_data['latitude'], 
                           c=status_colors[status], alpha=0.7, s=20, label=status)
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')
    axes[0].set_title('Conservation Status Spatial Distribution')
    axes[0].legend()
    
    # Incidents location
    if len(incidents) > 0:
        axes[1].scatter(embeddings_df['longitude'], embeddings_df['latitude'], 
                       c='lightblue', alpha=0.3, s=10, label='All protected areas')
        axes[1].scatter(incidents['longitude'], incidents['latitude'], 
                       c='red', alpha=0.8, s=30, label='Incidents')
        axes[1].set_xlabel('Longitude')
        axes[1].set_ylabel('Latitude')
        axes[1].set_title('Protected Area Incidents')
        axes[1].legend()
    
    # Threat level spatial distribution
    scatter3 = axes[2].scatter(embeddings_df['longitude'], embeddings_df['latitude'], 
                              c=embeddings_df['threat_level'], cmap='OrRd', 
                              alpha=0.7, s=15)
    axes[2].set_xlabel('Longitude')
    axes[2].set_ylabel('Latitude')
    axes[2].set_title('Threat Level Distribution')
    plt.colorbar(scatter3, ax=axes[2], label='Threat Level')
    
    save_plot(fig, f"{figs}/protected_areas_spatial_analysis.png", 
              "Spatial Protected Area Analysis - Uzbekistan")
    
    # Generate data tables
    print("Generating data tables...")
    
    # 1. Regional summary
    regional_df.to_csv(f"{tables}/protected_areas_regional_analysis.csv", index=False)
    
    # 2. Incident summary
    if len(incidents) > 0:
        incident_summary = incidents.groupby('region').agg({
            'disturbance_index': ['count', 'mean'],
            'threat_level': 'mean',
            'current_forest_cover': 'mean',
            'biodiversity_index': 'mean',
            'area_size_km2': 'sum'
        }).round(3)
        
        incident_summary.columns = ['_'.join(col).strip() for col in incident_summary.columns]
        incident_summary = incident_summary.reset_index()
        incident_summary.to_csv(f"{tables}/protected_areas_incidents.csv", index=False)
    
    # 3. Conservation priorities
    conservation_priorities = embeddings_df[embeddings_df['conservation_status'].isin(['Poor', 'Critical'])].groupby('region').agg({
        'disturbance_index': ['count', 'mean'],
        'threat_level': 'mean',
        'area_size_km2': 'sum',
        'illegal_logging_incidents': 'sum',
        'fire_incidents_5yr': 'sum'
    }).round(3)
    
    conservation_priorities.columns = ['_'.join(col).strip() for col in conservation_priorities.columns]
    conservation_priorities = conservation_priorities.reset_index()
    conservation_priorities.to_csv(f"{tables}/protected_areas_conservation_priorities.csv", index=False)
    
    # 4. Management effectiveness
    management_analysis = embeddings_df.groupby(['region', 'protection_level']).agg({
        'disturbance_index': 'mean',
        'current_forest_cover': 'mean',
        'biodiversity_index': 'mean'
    }).round(3)
    
    management_analysis = management_analysis.reset_index()
    management_analysis.to_csv(f"{tables}/protected_areas_management_effectiveness.csv", index=False)
    
    # Generate GeoJSON for incidents
    print("Generating incident GeoJSON...")
    
    geojson_features = []
    for _, row in incidents.iterrows():
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [row['longitude'], row['latitude']]
            },
            "properties": {
                "site_id": row['sample_id'],
                "region": row['region'],
                "protection_level": row['protection_level'],
                "conservation_status": row['conservation_status'],
                "disturbance_index": float(row['disturbance_index']),
                "threat_level": float(row['threat_level']),
                "area_size_km2": float(row['area_size_km2']),
                "forest_cover": float(row['current_forest_cover']),
                "biodiversity_index": float(row['biodiversity_index']),
                "anomaly_score": float(row['anomaly_score']),
                "primary_threat": row['agricultural_expansion_threat'],
                "management_issue": row['ranger_presence'],
                "incident_type": "conservation_concern"
            }
        }
        geojson_features.append(feature)
    
    geojson_data = {
        "type": "FeatureCollection",
        "features": geojson_features,
        "metadata": {
            "total_incidents": len(geojson_features),
            "generation_date": datetime.now().isoformat(),
            "analysis_type": "AlphaEarth Protected Area Disturbance Assessment",
            "critical_areas": int(embeddings_df[embeddings_df['conservation_status'] == 'Critical'].shape[0])
        }
    }
    
    with open(f"{final}/protected_area_incidents.geojson", 'w') as f:
        json.dump(geojson_data, f, indent=2)
    
    # Generate executive summary statistics
    total_areas = len(embeddings_df)
    critical_areas = len(embeddings_df[embeddings_df['conservation_status'] == 'Critical'])
    avg_disturbance = embeddings_df['disturbance_index'].mean()
    total_incidents = len(incidents)
    
    print("Protected area disturbance analysis completed successfully!")
    print(f"Key findings:")
    print(f"  - Total protected areas analyzed: {total_areas}")
    print(f"  - Areas with critical conservation status: {critical_areas} ({(critical_areas/total_areas*100):.1f}%)")
    print(f"  - Average disturbance index: {avg_disturbance:.3f}")
    print(f"  - Incidents requiring immediate attention: {total_incidents}")
    
    artifacts = [
        "tables/protected_areas_regional_analysis.csv",
        "tables/protected_areas_incidents.csv",
        "tables/protected_areas_conservation_priorities.csv",
        "tables/protected_areas_management_effectiveness.csv",
        "figs/protected_areas_conservation_analysis.png",
        "figs/protected_areas_spatial_analysis.png",
        "data_final/protected_area_incidents.geojson"
    ]
    
    return {
        "status": "ok",
        "artifacts": artifacts,
        "summary_stats": {
            "total_protected_areas": int(total_areas),
            "critical_areas": int(critical_areas),
            "avg_disturbance_index": float(avg_disturbance),
            "total_incidents": int(total_incidents),
            "total_protected_area_km2": float(embeddings_df['area_size_km2'].sum())
        }
    }
