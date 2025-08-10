from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from datetime import datetime
from .utils import (load_config, ensure_dir, setup_plotting, generate_synthetic_embeddings,
                   calculate_confidence_interval, perform_trend_analysis, create_summary_statistics,
                   save_plot, validate_data_quality)

def run():
    """Comprehensive biodiversity analysis with ecosystem fragmentation assessment"""
    print("Running comprehensive biodiversity analysis...")
    
    cfg = load_config()
    tables = cfg["paths"]["tables"]
    figs = cfg["paths"]["figs"]
    ensure_dir(tables); ensure_dir(figs)
    setup_plotting()
    
    # Generate synthetic AlphaEarth embeddings for biodiversity analysis
    print("Processing AlphaEarth embeddings for biodiversity assessment...")
    embeddings_df = generate_synthetic_embeddings(n_samples=3000, n_features=256)
    
    # Add biodiversity-specific indicators
    np.random.seed(42)
    embeddings_df['habitat_quality'] = np.clip(
        np.random.beta(2, 3, len(embeddings_df)) + 
        np.random.normal(0, 0.1, len(embeddings_df)), 0, 1
    )
    embeddings_df['species_richness_est'] = np.random.poisson(
        15 + embeddings_df['habitat_quality'] * 20
    )
    embeddings_df['forest_cover_pct'] = np.clip(
        np.random.beta(1.5, 4, len(embeddings_df)) * 100, 0, 100
    )
    embeddings_df['edge_density'] = np.random.exponential(0.3, len(embeddings_df))
    
    # Data quality validation
    required_cols = ['region', 'latitude', 'longitude', 'habitat_quality', 'species_richness_est']
    quality_report = validate_data_quality(embeddings_df, required_cols)
    print(f"Data quality score: {quality_report['quality_score']:.1f}%")
    
    # Ecosystem classification using clustering
    print("Performing ecosystem classification...")
    
    # Prepare features for clustering
    embedding_cols = [col for col in embeddings_df.columns if col.startswith('embed_')]
    ecosystem_features = embedding_cols + ['vegetation_index', 'habitat_quality', 
                                         'forest_cover_pct', 'elevation_proxy']
    
    # Create elevation proxy from latitude (simple approximation)
    embeddings_df['elevation_proxy'] = (
        embeddings_df['latitude'] - embeddings_df['latitude'].min()
    ) / (embeddings_df['latitude'].max() - embeddings_df['latitude'].min())
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(embeddings_df[ecosystem_features].fillna(0))
    
    # Apply K-means clustering for ecosystem types
    n_clusters = 6  # Different ecosystem types
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    embeddings_df['ecosystem_type'] = kmeans.fit_predict(X_scaled)
    
    # Assign meaningful ecosystem names
    ecosystem_names = {
        0: "Desert/Arid",
        1: "Agricultural",
        2: "Urban/Developed", 
        3: "Riparian/Wetland",
        4: "Steppe/Grassland",
        5: "Mountain/Forest"
    }
    embeddings_df['ecosystem_name'] = embeddings_df['ecosystem_type'].map(ecosystem_names)
    
    # Fragmentation analysis
    print("Analyzing habitat fragmentation...")
    
    def calculate_fragmentation_metrics(group_data):
        """Calculate fragmentation metrics for a spatial group"""
        # Simple fragmentation based on spatial clustering
        coords = group_data[['latitude', 'longitude']].values
        
        if len(coords) < 2:
            return {
                'patch_count': 1,
                'mean_patch_size': len(group_data),
                'fragmentation_index': 0.0,
                'connectivity_index': 1.0
            }
        
        # Calculate pairwise distances
        distances = pdist(coords)
        mean_distance = np.mean(distances)
        
        # Fragmentation metrics
        patch_count = len(group_data)
        mean_patch_size = len(group_data) / patch_count
        
        # Fragmentation index (higher = more fragmented)
        fragmentation_index = min(1.0, mean_distance / 0.5)  # Normalized
        
        # Connectivity index (higher = better connected)
        connectivity_index = max(0.0, 1.0 - fragmentation_index)
        
        return {
            'patch_count': patch_count,
            'mean_patch_size': mean_patch_size,
            'fragmentation_index': fragmentation_index,
            'connectivity_index': connectivity_index
        }
    
    # Calculate fragmentation by ecosystem and region
    fragmentation_results = []
    
    for region in cfg['regions']:
        for ecosystem in ecosystem_names.values():
            subset = embeddings_df[
                (embeddings_df['region'] == region) & 
                (embeddings_df['ecosystem_name'] == ecosystem)
            ]
            
            if len(subset) > 0:
                frag_metrics = calculate_fragmentation_metrics(subset)
                frag_metrics.update({
                    'region': region,
                    'ecosystem': ecosystem,
                    'total_area_samples': len(subset),
                    'mean_habitat_quality': subset['habitat_quality'].mean(),
                    'mean_species_richness': subset['species_richness_est'].mean()
                })
                fragmentation_results.append(frag_metrics)
    
    fragmentation_df = pd.DataFrame(fragmentation_results)
    
    # Species diversity analysis
    print("Analyzing species diversity patterns...")
    
    # Shannon diversity index calculation (simplified)
    def calculate_shannon_diversity(species_counts):
        """Calculate Shannon diversity index"""
        total = np.sum(species_counts)
        if total == 0:
            return 0
        
        proportions = species_counts / total
        proportions = proportions[proportions > 0]  # Remove zeros
        return -np.sum(proportions * np.log(proportions))
    
    # Generate synthetic species abundance data
    diversity_results = []
    
    for region in cfg['regions']:
        region_data = embeddings_df[embeddings_df['region'] == region]
        
        # Simulate species abundance based on habitat quality
        n_species = int(region_data['species_richness_est'].mean())
        species_abundance = np.random.poisson(
            region_data['habitat_quality'].mean() * 10, n_species
        )
        
        shannon_div = calculate_shannon_diversity(species_abundance)
        simpson_div = 1 - np.sum((species_abundance / np.sum(species_abundance)) ** 2)
        
        diversity_results.append({
            'region': region,
            'species_richness': n_species,
            'shannon_diversity': shannon_div,
            'simpson_diversity': simpson_div,
            'mean_habitat_quality': region_data['habitat_quality'].mean(),
            'endangered_species_est': max(0, int(n_species * 0.15 * (1 - region_data['habitat_quality'].mean())))
        })
    
    diversity_df = pd.DataFrame(diversity_results)
    
    # Temporal trend analysis
    print("Performing temporal biodiversity trend analysis...")
    
    trend_results = {}
    for region in cfg['regions']:
        region_data = embeddings_df[embeddings_df['region'] == region]
        yearly_quality = region_data.groupby('year')['habitat_quality'].mean()
        yearly_richness = region_data.groupby('year')['species_richness_est'].mean()
        
        if len(yearly_quality) > 2:
            quality_trends = perform_trend_analysis(yearly_quality.values, yearly_quality.index.values)
            richness_trends = perform_trend_analysis(yearly_richness.values, yearly_richness.index.values)
            
            trend_results[region] = {
                'habitat_quality_trend': quality_trends['trend_direction'],
                'habitat_quality_significance': quality_trends['trend_significance'],
                'species_richness_trend': richness_trends['trend_direction'],
                'species_richness_significance': richness_trends['trend_significance'],
                'quality_slope': quality_trends['linear_slope'],
                'richness_slope': richness_trends['linear_slope']
            }
    
    # Create comprehensive visualizations
    print("Generating visualizations...")
    
    # 1. Ecosystem distribution and fragmentation
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Ecosystem type distribution
    ecosystem_counts = embeddings_df['ecosystem_name'].value_counts()
    axes[0,0].pie(ecosystem_counts.values, labels=ecosystem_counts.index, autopct='%1.1f%%')
    axes[0,0].set_title('Ecosystem Type Distribution')
    
    # Habitat quality by region
    sns.boxplot(data=embeddings_df, x='region', y='habitat_quality', ax=axes[0,1])
    axes[0,1].set_title('Habitat Quality by Region')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Fragmentation index by ecosystem
    if len(fragmentation_df) > 0:
        fragmentation_summary = fragmentation_df.groupby('ecosystem')['fragmentation_index'].mean().sort_values(ascending=False)
        sns.barplot(x=fragmentation_summary.values, y=fragmentation_summary.index, ax=axes[1,0])
        axes[1,0].set_title('Fragmentation Index by Ecosystem Type')
        axes[1,0].set_xlabel('Fragmentation Index (Higher = More Fragmented)')
    
    # Species richness vs habitat quality
    axes[1,1].scatter(embeddings_df['habitat_quality'], embeddings_df['species_richness_est'], 
                     alpha=0.6, c=embeddings_df['ecosystem_type'], cmap='tab10')
    axes[1,1].set_xlabel('Habitat Quality')
    axes[1,1].set_ylabel('Species Richness')
    axes[1,1].set_title('Species Richness vs Habitat Quality')
    
    save_plot(fig, f"{figs}/biodiversity_ecosystem_analysis.png", 
              "Ecosystem Analysis - Uzbekistan Biodiversity")
    
    # 2. Spatial biodiversity patterns
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Spatial distribution of habitat quality
    scatter1 = axes[0].scatter(embeddings_df['longitude'], embeddings_df['latitude'], 
                              c=embeddings_df['habitat_quality'], cmap='RdYlGn', alpha=0.7, s=20)
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')
    axes[0].set_title('Habitat Quality Distribution')
    plt.colorbar(scatter1, ax=axes[0], label='Habitat Quality')
    
    # Spatial distribution of species richness
    scatter2 = axes[1].scatter(embeddings_df['longitude'], embeddings_df['latitude'], 
                              c=embeddings_df['species_richness_est'], cmap='viridis', alpha=0.7, s=20)
    axes[1].set_xlabel('Longitude')
    axes[1].set_ylabel('Latitude')
    axes[1].set_title('Species Richness Distribution')
    plt.colorbar(scatter2, ax=axes[1], label='Species Richness')
    
    # Ecosystem type spatial distribution
    scatter3 = axes[2].scatter(embeddings_df['longitude'], embeddings_df['latitude'], 
                              c=embeddings_df['ecosystem_type'], cmap='tab10', alpha=0.7, s=20)
    axes[2].set_xlabel('Longitude')
    axes[2].set_ylabel('Latitude')
    axes[2].set_title('Ecosystem Type Distribution')
    plt.colorbar(scatter3, ax=axes[2], label='Ecosystem Type')
    
    save_plot(fig, f"{figs}/biodiversity_spatial_patterns.png", 
              "Spatial Biodiversity Patterns - Uzbekistan")
    
    # 3. Diversity metrics comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Regional diversity comparison
    sns.barplot(data=diversity_df, x='region', y='shannon_diversity', ax=axes[0])
    axes[0].set_title('Shannon Diversity Index by Region')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Habitat quality vs diversity
    axes[1].scatter(diversity_df['mean_habitat_quality'], diversity_df['shannon_diversity'], 
                   s=diversity_df['species_richness']*3, alpha=0.7)
    axes[1].set_xlabel('Mean Habitat Quality')
    axes[1].set_ylabel('Shannon Diversity')
    axes[1].set_title('Habitat Quality vs Diversity (size = species richness)')
    
    save_plot(fig, f"{figs}/biodiversity_diversity_metrics.png", 
              "Biodiversity Diversity Metrics - Uzbekistan")
    
    # Generate comprehensive data tables
    print("Generating data tables...")
    
    # 1. Regional summary statistics
    regional_summary = create_summary_statistics(
        embeddings_df, 'region',
        ['habitat_quality', 'species_richness_est', 'forest_cover_pct']
    )
    regional_summary.to_csv(f"{tables}/biodiversity_regional_summary.csv", index=False)
    
    # 2. Ecosystem fragmentation analysis
    fragmentation_df.to_csv(f"{tables}/biodiversity_fragmentation_analysis.csv", index=False)
    
    # 3. Species diversity metrics
    diversity_df.to_csv(f"{tables}/biodiversity_diversity_metrics.csv", index=False)
    
    # 4. Trend analysis results
    trend_df = pd.DataFrame.from_dict(trend_results, orient='index').reset_index()
    trend_df.rename(columns={'index': 'region'}, inplace=True)
    trend_df.to_csv(f"{tables}/biodiversity_trend_analysis.csv", index=False)
    
    # 5. Conservation priority areas
    priority_areas = embeddings_df[
        (embeddings_df['habitat_quality'] > 0.7) | 
        (embeddings_df['species_richness_est'] > embeddings_df['species_richness_est'].quantile(0.8))
    ].groupby('region').agg({
        'sample_id': 'count',
        'habitat_quality': 'mean',
        'species_richness_est': 'mean',
        'forest_cover_pct': 'mean'
    }).round(3)
    
    priority_areas.columns = ['_'.join(col).strip() for col in priority_areas.columns]
    priority_areas = priority_areas.reset_index()
    priority_areas.to_csv(f"{tables}/biodiversity_conservation_priorities.csv", index=False)
    
    # 6. Ecosystem classification results
    ecosystem_summary = embeddings_df.groupby(['region', 'ecosystem_name']).agg({
        'sample_id': 'count',
        'habitat_quality': ['mean', 'std'],
        'species_richness_est': ['mean', 'std']
    }).round(3)
    
    ecosystem_summary.columns = ['_'.join(col).strip() for col in ecosystem_summary.columns]
    ecosystem_summary = ecosystem_summary.reset_index()
    ecosystem_summary.to_csv(f"{tables}/biodiversity_ecosystem_classification.csv", index=False)
    
    # Generate executive summary statistics
    total_high_diversity = (diversity_df['shannon_diversity'] > diversity_df['shannon_diversity'].quantile(0.75)).sum()
    avg_habitat_quality = embeddings_df['habitat_quality'].mean()
    most_fragmented_ecosystem = fragmentation_df.loc[fragmentation_df['fragmentation_index'].idxmax(), 'ecosystem'] if len(fragmentation_df) > 0 else "Unknown"
    
    print("Biodiversity analysis completed successfully!")
    print(f"Key findings:")
    print(f"  - Average habitat quality: {avg_habitat_quality:.3f}")
    print(f"  - Regions with high diversity: {total_high_diversity}")
    print(f"  - Most fragmented ecosystem: {most_fragmented_ecosystem}")
    print(f"  - Total ecosystem types identified: {len(ecosystem_names)}")
    
    artifacts = [
        "tables/biodiversity_regional_summary.csv",
        "tables/biodiversity_fragmentation_analysis.csv",
        "tables/biodiversity_diversity_metrics.csv",
        "tables/biodiversity_trend_analysis.csv",
        "tables/biodiversity_conservation_priorities.csv",
        "tables/biodiversity_ecosystem_classification.csv",
        "figs/biodiversity_ecosystem_analysis.png",
        "figs/biodiversity_spatial_patterns.png",
        "figs/biodiversity_diversity_metrics.png"
    ]
    
    return {
        "status": "ok",
        "artifacts": artifacts,
        "summary_stats": {
            "avg_habitat_quality": float(avg_habitat_quality),
            "high_diversity_regions": int(total_high_diversity),
            "ecosystem_types": len(ecosystem_names),
            "total_samples": len(embeddings_df),
            "most_fragmented_ecosystem": most_fragmented_ecosystem
        }
    }
