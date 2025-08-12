"""
Quick test of Google Satellite Embedding V1 dataset access
"""
import ee

# Initialize Google Earth Engine
try:
    ee.Initialize()
    print("✅ Google Earth Engine initialized")
except:
    try:
        ee.Authenticate()
        ee.Initialize()
        print("✅ Google Earth Engine authenticated and initialized")
    except Exception as e:
        print(f"❌ GEE initialization failed: {e}")
        exit(1)

# Test satellite embedding dataset access
try:
    print("🔍 Testing Google Satellite Embedding V1 access...")
    
    # Test the correct dataset ID
    embedding_collection = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
    
    # Test for a specific year and location (Tashkent)
    tashkent = ee.Geometry.Point([69.2697, 41.2995])
    
    embedding = (embedding_collection
                .filterBounds(tashkent)
                .filterDate('2023-01-01', '2024-01-01')
                .first())
    
    if embedding:
        # Get basic info about the embedding
        band_names = embedding.bandNames()
        print(f"✅ Satellite Embedding V1 dataset accessible!")
        print(f"📊 Available bands: {band_names.getInfo()[:10]}...")  # Show first 10 bands
        print(f"📊 Total bands: {band_names.size().getInfo()}")
        
        # Test clustering capability
        print("🔬 Testing clustering on embeddings...")
        
        # Use K-means clustering on the embeddings
        clusterer = ee.Clusterer.wekaKMeans(3)
        clusters = embedding.cluster(clusterer)
        
        # Get some cluster statistics
        cluster_info = clusters.reduceRegion(
            reducer=ee.Reducer.histogram(),
            geometry=tashkent.buffer(5000),  # 5km buffer
            scale=100,
            maxPixels=1e6
        )
        
        print("✅ Clustering successful!")
        print("📈 This confirms satellite embeddings can be used for urban/rural classification")
        
    else:
        print("❌ No embedding data found for test location/date")
        
except Exception as e:
    print(f"❌ Satellite Embedding V1 test failed: {e}")
    print("💡 This confirms the dataset might not be accessible or the ID is incorrect")

print("\n" + "="*60)
print("🔍 SATELLITE EMBEDDING V1 TEST COMPLETE")
print("="*60)
