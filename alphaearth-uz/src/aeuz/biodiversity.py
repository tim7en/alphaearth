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
from .utils import (load_config, ensure_dir, setup_plotting, load_alphaearth_embeddings,
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
    
    # Load real environmental data for biodiversity analysis
    print("Loading real environmental data for biodiversity assessment...")
    embeddings_df = load_alphaearth_embeddings(regions=cfg['regions'], n_features=256)
    
    # Calculate biodiversity indicators from real environmental data
    print("Calculating biodiversity indicators from environmental characteristics...")
    
    def calculate_habitat_quality(row):
        """Calculate habitat quality from environmental factors"""
        quality = 0.5  # Base quality
        
        # NDVI contribution (higher vegetation = better habitat)
        quality += row['ndvi_calculated'] * 0.3
        
        # Water availability (lower water stress = better habitat)
        quality += (1 - row['water_stress_level']) * 0.2
        
        # Distance to water (closer = better)
        water_factor = max(0, 1 - row['distance_to_water'] / 30.0)
        quality += water_factor * 0.1
        
        # Elevation diversity (moderate elevation = better)
        if 200 <= row['elevation'] <= 1000:
            quality += 0.1
        elif row['elevation'] > 1500:
            quality -= 0.1  # Too high
        
        # Human disturbance (farther from urban = better for wildlife)
        disturbance_factor = min(0.1, row['distance_to_urban'] / 50.0)
        quality += disturbance_factor
        
        return np.clip(quality, 0, 1)
    
    embeddings_df['habitat_quality'] = embeddings_df.apply(calculate_habitat_quality, axis=1)
    
    # Calculate species richness from habitat quality and environmental diversity
    def calculate_species_richness(row):
        """Estimate species richness from habitat characteristics"""
        base_richness = 5  # Minimum species
        
        # Habitat quality factor
        habitat_bonus = row['habitat_quality'] * 25
        
        # Vegetation diversity (NDVI indicates plant diversity)
        vegetation_bonus = row['ndvi_calculated'] * 15
        
        # Water availability (aquatic and semi-aquatic species)
        water_bonus = (1 - row['water_stress_level']) * 10
        
        # Regional diversity factors
        regional_factors = {
            "Karakalpakstan": 0.7,  # Arid, lower diversity
            "Tashkent": 1.1,        # Urban edge diversity
            "Samarkand": 1.0,       # Moderate diversity
            "Bukhara": 0.8,         # Oasis ecosystem
            "Namangan": 1.3         # Mountain diversity
        }
        
        regional_multiplier = regional_factors.get(row['region'], 1.0)
        
        total_richness = (base_richness + habitat_bonus + vegetation_bonus + water_bonus) * regional_multiplier
        
        return int(np.clip(total_richness, 3, 60))  # Realistic range
    
    embeddings_df['species_richness_est'] = embeddings_df.apply(calculate_species_richness, axis=1)
    
    # Calculate forest cover from NDVI and land use
    def calculate_forest_cover(row):
        """Estimate forest cover percentage from vegetation indices"""
        if row['ndvi_calculated'] < 0.3:
            return 0  # No forest in low vegetation areas
        elif row['ndvi_calculated'] < 0.5:
            return row['ndvi_calculated'] * 20  # Sparse woodland
        else:
            # Adjust by region (some regions naturally have less forest)
            forest_potential = {
                "Karakalpakstan": 5,   # Desert region
                "Tashkent": 15,        # Some urban forestry
                "Samarkand": 25,       # Agricultural with trees
                "Bukhara": 10,         # Oasis vegetation
                "Namangan": 45         # Mountain forests
            }
            max_forest = forest_potential.get(row['region'], 20)
            return min(max_forest, row['ndvi_calculated'] * 60)
    
    embeddings_df['forest_cover_pct'] = embeddings_df.apply(calculate_forest_cover, axis=1)
    
    # Calculate edge density from habitat fragmentation
    def calculate_edge_density(row):
        """Calculate habitat edge density from fragmentation indicators"""
        base_edge = 0.1
        
        # Higher edge density near urban areas (fragmentation)
        urban_effect = max(0, 1 - row['distance_to_urban'] / 30.0) * 0.4
        
        # Agricultural areas have moderate edge density
        if row['ndvi_calculated'] > 0.3 and row['distance_to_urban'] < 20:
            agricultural_effect = 0.2
        else:
            agricultural_effect = 0.0
        
        # Water bodies create natural edges
        water_effect = max(0, 1 - row['distance_to_water'] / 15.0) * 0.15
        
        return base_edge + urban_effect + agricultural_effect + water_effect
    
    embeddings_df['edge_density'] = embeddings_df.apply(calculate_edge_density, axis=1)
    
    # Data quality validation
    required_cols = ['region', 'latitude', 'longitude', 'habitat_quality', 'species_richness_est']
    quality_report = validate_data_quality(embeddings_df, required_cols)
    print(f"Data quality score: {quality_report['quality_score']:.1f}%")
    
    # Ecosystem classification using clustering
    print("Performing ecosystem classification...")
    
    # Prepare features for clustering using real environmental data
    embedding_cols = [col for col in embeddings_df.columns if col.startswith('embedding_')]
    ecosystem_features = embedding_cols + ['ndvi_calculated', 'habitat_quality', 
                                         'forest_cover_pct', 'elevation']
    
    # Filter only available features
    available_features = [col for col in ecosystem_features if col in embeddings_df.columns]
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(embeddings_df[available_features].fillna(0))
    
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
        
        # Calculate species abundance deterministically based on habitat characteristics
        n_species = int(region_data['species_richness_est'].mean())
        mean_habitat_quality = region_data['habitat_quality'].mean()
        
        # Create realistic species abundance distribution based on habitat quality
        species_abundance = []
        for i in range(n_species):
            # Common species have higher abundance
            if i < n_species * 0.3:  # 30% common species
                abundance = int(mean_habitat_quality * 25 + (n_species - i) * 2)
            elif i < n_species * 0.7:  # 40% moderate species
                abundance = int(mean_habitat_quality * 15 + (n_species - i) * 1.5)
            else:  # 30% rare species
                abundance = int(mean_habitat_quality * 8 + (n_species - i) * 0.8)
            species_abundance.append(max(1, abundance))
        
        species_abundance = np.array(species_abundance)
        
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
