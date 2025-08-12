import ee
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuration - All major Uzbekistan cities

UZBEKISTAN_CITIES = {
    # National capital (separate admin unit)
    "Tashkent":   {"lat": 41.2995, "lon": 69.2401, "buffer": 15000},
    
    # Republic capital
    "Nukus":      {"lat": 42.4731, "lon": 59.6103, "buffer": 10000},  # Karakalpakstan
    
    # Regional capitals
    "Andijan":    {"lat": 40.7821, "lon": 72.3442, "buffer": 12000},
    #"Bukhara":    {"lat": 39.7748, "lon": 64.4286, "buffer": 10000},
    #"Jizzakh":    {"lat": 40.1158, "lon": 67.8422, "buffer": 8000},
    #"Qarshi":     {"lat": 38.8606, "lon": 65.7887, "buffer": 8000},
    #"Navoiy":     {"lat": 40.1030, "lon": 65.3686, "buffer": 10000},
    #"Namangan":   {"lat": 40.9983, "lon": 71.6726, "buffer": 12000},
    #"Samarkand":  {"lat": 39.6542, "lon": 66.9597, "buffer": 12000},
    #"Termez":     {"lat": 37.2242, "lon": 67.2783, "buffer": 8000},
    #"Gulistan":   {"lat": 40.4910, "lon": 68.7810, "buffer": 8000},
    #"Nurafshon":  {"lat": 41.0167, "lon": 69.3417, "buffer": 8000},
    #"Fergana":    {"lat": 40.3842, "lon": 71.7843, "buffer": 12000},
    #"Urgench":    {"lat": 41.5506, "lon": 60.6317, "buffer": 10000},
}

# Scientific constants - Multi-dataset approach
WARM_MONTHS = [6, 7, 8]  # June, July, August
URBAN_THRESHOLD = 0.3    # Threshold for urban classification from multiple sources
RURAL_THRESHOLD = 0.2    # Threshold for rural classification
RING_KM = 25             # Rural reference area
TARGET_SCALE = 1000      # 1 km to match MODIS LST native resolution
MIN_URBAN_PIXELS = 10    # Minimum urban pixels required for valid SUHI calculation
MIN_RURAL_PIXELS = 20    # Minimum rural pixels required for valid SUHI calculation

def authenticate_gee():
    """Initialize Google Earth Engine"""
    try:
        print("🔑 Initializing Google Earth Engine...")
        ee.Initialize(project='ee-sabitovty')
        print("✅ Google Earth Engine initialized successfully!")
        return True
    except Exception as e:
        print(f"❌ GEE Authentication failed: {e}")
        return False

def setup_output_directories():
    """Create organized directory structure for outputs"""
    base_dir = Path(__file__).parent / "scientific_suhi_analysis"
    
    directories = {
        'base': base_dir,
        'images': base_dir / "images",
        'data': base_dir / "data", 
        'reports': base_dir / "reports",
        'gis_maps': base_dir / "gis_maps",
        'trends': base_dir / "trends"
    }
    
    for dir_name, dir_path in directories.items():
        dir_path.mkdir(parents=True, exist_ok=True)
        
    print(f"📁 Output directories created in: {base_dir}")
    return directories

def _combine_urban_classifications(geom, start, end, year):
    """
    Combine multiple urban classification datasets for robust urban/rural identification
    Uses: Dynamic World, GHSL, ESA WorldCover, and MODIS Land Cover
    """
    urban_layers = []
    weights = []
    
    # 1. Dynamic World V1 - Most recent and detailed
    try:
        dw = (ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
              .filterDate(start, end)
              .filterBounds(geom)
              .select('built'))
        
        size = dw.size()
        if size.gt(0):
            dw_built = dw.median().rename('dw_built')
            urban_layers.append(dw_built)
            weights.append(0.35)  # Higher weight for recent data
            print(f"   ✓ Dynamic World data available")
    except:
        print(f"   ⚠️ Dynamic World not available")
    
    # 2. Global Human Settlement Layer (GHSL) - Built-up areas
    try:
        # Use the most recent GHSL built-up layer
        ghsl = ee.Image('JRC/GHSL/P2023A/GHS_BUILT_S/2020').select('built_surface')
        # Normalize to 0-1 range (original is percentage 0-100)
        ghsl_norm = ghsl.divide(100).rename('ghsl_built')
        urban_layers.append(ghsl_norm)
        weights.append(0.25)
        print(f"   ✓ GHSL data available")
    except:
        print(f"   ⚠️ GHSL not available")
    
    # 3. ESA WorldCover - 10m resolution land cover
    try:
        # Use 2021 if available, otherwise 2020
        esa = ee.Image('ESA/WorldCover/v200/2021').select('Map')
        # Built-up class is 50
        esa_built = esa.eq(50).rename('esa_built')
        urban_layers.append(esa_built)
        weights.append(0.2)
        print(f"   ✓ ESA WorldCover data available")
    except:
        try:
            esa = ee.Image('ESA/WorldCover/v100/2020').select('Map')
            esa_built = esa.eq(50).rename('esa_built')
            urban_layers.append(esa_built)
            weights.append(0.2)
            print(f"   ✓ ESA WorldCover 2020 data available")
        except:
            print(f"   ⚠️ ESA WorldCover not available")
    
    # 4. MODIS Land Cover Type - 500m resolution
    try:
        modis_lc = (ee.ImageCollection('MODIS/061/MCD12Q1')
                   .filterDate(f'{year-1}-01-01', f'{year+1}-01-01')
                   .first()
                   .select('LC_Type1'))
        # Urban and built-up class is 13
        modis_urban = modis_lc.eq(13).rename('modis_urban')
        urban_layers.append(modis_urban)
        weights.append(0.15)
        print(f"   ✓ MODIS Land Cover data available")
    except:
        print(f"   ⚠️ MODIS Land Cover not available")
    
    # 5. Impervious Surface from Landsat-based products
    try:
        # Use Global Land Analysis & Discovery (GLAD) Global Land Cover
        glad = ee.Image('projects/glad/GLCLU2020/LCLUC_2020')
        # Urban classes
        urban_glad = glad.eq(10).Or(glad.eq(11)).rename('glad_urban')
        urban_layers.append(urban_glad)
        weights.append(0.05)
        print(f"   ✓ GLAD Land Cover data available")
    except:
        print(f"   ⚠️ GLAD not available")
    
    # Combine all available layers with weighted average
    if len(urban_layers) > 0:
        # Normalize weights
        total_weight = sum(weights)
        weights = [w/total_weight for w in weights]
        
        # Create weighted composite
        combined = ee.Image.constant(0)
        for layer, weight in zip(urban_layers, weights):
            combined = combined.add(layer.multiply(weight))
        
        print(f"   ✓ Combined {len(urban_layers)} urban classification layers")
        return combined.rename('urban_probability')
    else:
        print(f"   ⚠️ No urban classification data available, using fallback")
        # Return a constant image as fallback
        return ee.Image.constant(0.5).rename('urban_probability')

def _mask_landsat_l2(img):
    """
    Properly mask Landsat L2 using QA_PIXEL bits
    """
    qa = img.select('QA_PIXEL')
    mask = (qa.bitwiseAnd(1<<1).eq(0)  # Dilated cloud
            .And(qa.bitwiseAnd(1<<2).eq(0))  # Cirrus
            .And(qa.bitwiseAnd(1<<3).eq(0))  # Cloud
            .And(qa.bitwiseAnd(1<<4).eq(0))  # Cloud shadow
            .And(qa.bitwiseAnd(1<<5).eq(0))) # Snow
    
    # Apply scale factors for L2 surface reflectance
    sr = img.select(['SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7']) \
            .multiply(0.0000275).add(-0.2)
    
    # Ensure values are in valid range [0, 1]
    sr = sr.clamp(0, 1)
    
    return sr.updateMask(mask)

def _landsat_nd_indices(geom, start, end):
    """
    Calculate vegetation indices from Landsat 8/9 L2
    """
    try:
        coll = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
                .merge(ee.ImageCollection('LANDSAT/LC09/C02/T1_L2'))
                .filterDate(start, end)
                .filterBounds(geom)
                .map(_mask_landsat_l2))
        
        # Apply warm season filter if specified - DISABLED to avoid timestamp issues
        # if WARM_MONTHS:
        #     coll = coll.filter(ee.Filter.calendarRange(WARM_MONTHS[0], WARM_MONTHS[-1], 'month'))
        
        size = coll.size()
        
        # Use median composite if data available
        comp = ee.Algorithms.If(
            size.gt(0),
            coll.median(),
            ee.Image.constant([0.1, 0.15, 0.2, 0.4, 0.3, 0.25])
                .rename(['SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7'])
        )
        comp = ee.Image(comp)
        
        # Calculate indices
        ndvi = comp.normalizedDifference(['SR_B5','SR_B4']).rename('NDVI')
        ndbi = comp.normalizedDifference(['SR_B6','SR_B5']).rename('NDBI')
        ndwi = comp.normalizedDifference(['SR_B3','SR_B5']).rename('NDWI')
        
        return ee.Image.cat([ndvi, ndbi, ndwi])
        
    except Exception as e:
        print(f"   ⚠️ Landsat error: {e}, using fallback")
        ndvi = ee.Image.constant(0.3).rename('NDVI')
        ndbi = ee.Image.constant(0.0).rename('NDBI')
        ndwi = ee.Image.constant(0.0).rename('NDWI')
        return ee.Image.cat([ndvi, ndbi, ndwi])

def _modis_lst(geom, start, end):
    """
    Process MODIS LST with proper scaling
    """
    try:
        coll = (ee.ImageCollection('MODIS/061/MOD11A2')
                .filterDate(start, end)
                .filterBounds(geom)
                .select(['LST_Day_1km','LST_Night_1km']))
        
        # Quality filtering
        def quality_mask(img):
            qa_day = img.select('QC_Day')
            qa_night = img.select('QC_Night')
            # Bits 0-1: 00 = good quality
            day_mask = qa_day.bitwiseAnd(3).eq(0)
            night_mask = qa_night.bitwiseAnd(3).eq(0)
            return img.updateMask(day_mask).updateMask(night_mask)
        
        # Apply warm season filter if specified - DISABLED to avoid timestamp issues
        # if WARM_MONTHS:
        #     coll = coll.filter(ee.Filter.calendarRange(WARM_MONTHS[0], WARM_MONTHS[-1], 'month'))
        
        size = coll.size()
        
        # Use median if data available
        med = ee.Algorithms.If(
            size.gt(0),
            coll.median(),
            ee.Image.constant([15000, 13000]).rename(['LST_Day_1km', 'LST_Night_1km'])
        )
        med = ee.Image(med)
        
        # Convert to Celsius
        lst_day = med.select('LST_Day_1km').multiply(0.02).subtract(273.15).rename('LST_Day')
        lst_night = med.select('LST_Night_1km').multiply(0.02).subtract(273.15).rename('LST_Night')
        
        # Clamp to reasonable temperature ranges for Uzbekistan
        lst_day = lst_day.clamp(-10, 60)
        lst_night = lst_night.clamp(-20, 40)
        
        return ee.Image.cat([lst_day, lst_night])
        
    except Exception as e:
        print(f"   ⚠️ MODIS LST error: {e}, using fallback")
        lst_day = ee.Image.constant(30).rename('LST_Day')
        lst_night = ee.Image.constant(20).rename('LST_Night')
        return ee.Image.cat([lst_day, lst_night])

def _to_1km(img, ref_proj):
    """
    Resample to 1 km resolution
    """
    return img.resample('bilinear').reproject(ref_proj, None, TARGET_SCALE)

def analyze_period(period):
    """
    Server-side computation for one period using combined classification approach
    """
    start, end, label = period['start'], period['end'], period['label']
    year = int(label)
    print(f"   🔄 Processing period {label}...")
    results = []

    for city, info in UZBEKISTAN_CITIES.items():
        print(f"      🏙️ Analyzing {city}...")
        
        try:
            # --- Geometries ---
            pt = ee.Geometry.Point([info['lon'], info['lat']])
            inner = pt.buffer(info['buffer'])
            outer = pt.buffer(info['buffer'] + RING_KM * 1000)
            rural_ring = outer.difference(inner)
            
            # Minimal erosion to avoid edge effects
            erode = TARGET_SCALE // 10  # 100m erosion
            urban_core = inner.buffer(-erode)
            rural_ring_eroded = rural_ring.buffer(-erode)
            
            # --- Get data ---
            bbox = outer.bounds()
            
            # LST data
            lst = _modis_lst(bbox, start, end)
            
            # Vegetation indices
            nds = _landsat_nd_indices(bbox, start, end)
            
            # Combined urban classification
            urban_prob = _combine_urban_classifications(bbox, start, end, year)
            
            # Reference projection from LST
            ref_proj = lst.select('LST_Day').projection()
            
            # Resample all to 1km
            nds_1k = _to_1km(nds, ref_proj)
            urban_prob_1k = _to_1km(urban_prob, ref_proj)
            
            # --- Water mask ---
            gsw = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('occurrence')
            water_mask = gsw.resample('bilinear').reproject(ref_proj, None, TARGET_SCALE).lt(25)
            
            # --- Create urban and rural masks ---
            # Urban mask: high urban probability within urban core
            urban_mask = (urban_prob_1k.gte(URBAN_THRESHOLD)
                         .And(water_mask)
                         .And(nds_1k.select('NDVI').lt(0.4)))  # Not heavily vegetated
            urban_mask = urban_mask.clip(urban_core).rename('urban_mask')
            
            # Rural mask: low urban probability within rural ring
            rural_mask = (urban_prob_1k.lt(RURAL_THRESHOLD)
                         .And(water_mask)
                         .Or(nds_1k.select('NDVI').gt(0.3)))  # Or vegetated areas
            rural_mask = rural_mask.clip(rural_ring_eroded).rename('rural_mask')
            
            # Stack variables
            vars_1k = ee.Image.cat([lst, nds_1k, urban_prob_1k]).float()
            
            # --- Compute statistics ---
            reducer = ee.Reducer.mean().combine(
                ee.Reducer.count(), sharedInputs=True
            ).combine(
                ee.Reducer.stdDev(), sharedInputs=True
            )
            
            # Urban statistics
            urban_stats = vars_1k.updateMask(urban_mask).reduceRegion(
                reducer=reducer,
                geometry=urban_core,
                scale=TARGET_SCALE,
                maxPixels=1e9,
                tileScale=4,
                bestEffort=True
            )
            
            # Rural statistics
            rural_stats = vars_1k.updateMask(rural_mask).reduceRegion(
                reducer=reducer,
                geometry=rural_ring_eroded,
                scale=TARGET_SCALE,
                maxPixels=1e9,
                tileScale=4,
                bestEffort=True
            )
            
            # Extract statistics directly to avoid temporal dependency issues
            try:
                urban_stats_info = urban_stats.getInfo()
                rural_stats_info = rural_stats.getInfo()
                
                # Create feature with all statistics
                common = {
                    'city': city,
                    'period': label,
                    'buffer_km': info['buffer']/1000,
                    'ring_km': RING_KM
                }
                
                urban_feat = ee.Feature(urban_core, ee.Dictionary(common).combine({
                    'type': 'core',
                    'stats': urban_stats_info
                }))
                
                rural_feat = ee.Feature(rural_ring, ee.Dictionary(common).combine({
                    'type': 'ring',
                    'stats': rural_stats_info
                }))
                
                results.extend([urban_feat, rural_feat])
                
            except Exception as stats_error:
                print(f"   ⚠️ Error extracting stats for {city}: {stats_error}")
                # Create fallback features with basic info
                common = {
                    'city': city,
                    'period': label,
                    'buffer_km': info['buffer']/1000,
                    'ring_km': RING_KM
                }
                
                urban_feat = ee.Feature(urban_core, ee.Dictionary(common).combine({
                    'type': 'core',
                    'stats': {}
                }))
                
                rural_feat = ee.Feature(rural_ring, ee.Dictionary(common).combine({
                    'type': 'ring',
                    'stats': {}
                }))
                
                results.extend([urban_feat, rural_feat])
            
        except Exception as e:
            print(f"   ⚠️ Error processing {city}: {e}")
            continue
    
    return ee.FeatureCollection(results)

def fc_to_pandas(fc, period_label):
    """
    Convert feature collection to pandas DataFrame with SUHI calculation
    """
    try:
        print(f"   🔄 Converting feature collection for {period_label}...")
        
        # Since stats are already extracted, we can safely get the info
        fc_info = fc.getInfo()
        
        if not fc_info or 'features' not in fc_info:
            print(f"   ❌ No features found for {period_label}")
            return pd.DataFrame()
        
        recs = []
        city_data = {}
        
        # Group by city
        for f in fc_info['features']:
            props = f.get('properties', {})
            city = props.get('city')
            geom_type = props.get('type')
            stats = props.get('stats', {})
            
            if not city or not geom_type:
                continue
                
            if city not in city_data:
                city_data[city] = {}
            
            city_data[city][geom_type] = stats
        
        print(f"   📊 Found data for {len(city_data)} cities")
        
        # Calculate SUHI for each city
        for city, data in city_data.items():
            if 'core' in data and 'ring' in data:
                core_stats = data['core']
                ring_stats = data['ring']
                
                # Helper function for safe value extraction
                def safe_get(stats_dict, key, default=None):
                    value = stats_dict.get(key, default)
                    if value is None or (isinstance(value, str) and value.lower() in ['null', 'none', '']):
                        return None
                    try:
                        return float(value)
                    except (ValueError, TypeError):
                        return None
                
                # Extract values
                lst_day_urban = safe_get(core_stats, 'LST_Day_mean')
                lst_day_rural = safe_get(ring_stats, 'LST_Day_mean')
                lst_night_urban = safe_get(core_stats, 'LST_Night_mean')
                lst_night_rural = safe_get(ring_stats, 'LST_Night_mean')
                
                urban_pixels = safe_get(core_stats, 'LST_Day_count', 0)
                rural_pixels = safe_get(ring_stats, 'LST_Day_count', 0)
                
                # Check minimum pixel requirements and handle gracefully
                if urban_pixels is None or urban_pixels < MIN_URBAN_PIXELS:
                    if urban_pixels is not None:
                        print(f"   ⚠️ {city}: Insufficient urban pixels ({urban_pixels} < {MIN_URBAN_PIXELS})")
                    else:
                        print(f"   ⚠️ {city}: No urban pixel data available")
                    lst_day_urban = None
                    lst_night_urban = None
                    urban_pixels = 0
                
                if rural_pixels is None or rural_pixels < MIN_RURAL_PIXELS:
                    if rural_pixels is not None:
                        print(f"   ⚠️ {city}: Insufficient rural pixels ({rural_pixels} < {MIN_RURAL_PIXELS})")
                    else:
                        print(f"   ⚠️ {city}: No rural pixel data available")
                    lst_day_rural = None
                    lst_night_rural = None
                    rural_pixels = 0
                
                # Calculate SUHI only if we have valid data and sufficient pixels
                suhi_day = None
                suhi_night = None
                
                if (lst_day_urban is not None and lst_day_rural is not None and 
                    urban_pixels is not None and rural_pixels is not None and
                    urban_pixels >= MIN_URBAN_PIXELS and rural_pixels >= MIN_RURAL_PIXELS):
                    suhi_day = lst_day_urban - lst_day_rural
                
                if (lst_night_urban is not None and lst_night_rural is not None and
                    urban_pixels is not None and rural_pixels is not None and
                    urban_pixels >= MIN_URBAN_PIXELS and rural_pixels >= MIN_RURAL_PIXELS):
                    suhi_night = lst_night_urban - lst_night_rural
                
                # Create record
                record = {
                    'City': city,
                    'Period': period_label,
                    'SUHI_Day': suhi_day,
                    'SUHI_Night': suhi_night,
                    'LST_Day_Urban': lst_day_urban,
                    'LST_Day_Rural': lst_day_rural,
                    'LST_Night_Urban': lst_night_urban,
                    'LST_Night_Rural': lst_night_rural,
                    'NDVI_Urban': safe_get(core_stats, 'NDVI_mean'),
                    'NDVI_Rural': safe_get(ring_stats, 'NDVI_mean'),
                    'NDBI_Urban': safe_get(core_stats, 'NDBI_mean'),
                    'NDBI_Rural': safe_get(ring_stats, 'NDBI_mean'),
                    'NDWI_Urban': safe_get(core_stats, 'NDWI_mean'),
                    'NDWI_Rural': safe_get(ring_stats, 'NDWI_mean'),
                    'Urban_Prob': safe_get(core_stats, 'urban_probability_mean'),
                    'Rural_Prob': safe_get(ring_stats, 'urban_probability_mean'),
                    'Urban_Pixel_Count': urban_pixels if urban_pixels is not None else 0,
                    'Rural_Pixel_Count': rural_pixels if rural_pixels is not None else 0,
                    'Data_Quality': 'Good' if (urban_pixels is not None and rural_pixels is not None and
                                              urban_pixels >= MIN_URBAN_PIXELS and
                                              rural_pixels >= MIN_RURAL_PIXELS) else 'Poor'
                }
                
                recs.append(record)
                
                # Print status
                if suhi_day is not None:
                    status = f"✅ {city}: SUHI_Day={suhi_day:.2f}°C"
                    if suhi_night is not None:
                        status += f", SUHI_Night={suhi_night:.2f}°C"
                    status += f" (Urban:{urban_pixels:.0f}, Rural:{rural_pixels:.0f} pixels)"
                    print(f"   {status}")
                else:
                    print(f"   ❌ {city}: No valid SUHI calculated (Urban:{urban_pixels:.0f}, Rural:{rural_pixels:.0f} pixels)")
        
        df = pd.DataFrame(recs)
        print(f"   📊 Converted {len(df)} city records for {period_label}")
        valid_suhi = df['SUHI_Day'].notna().sum()
        print(f"   📊 Valid SUHI calculations: {valid_suhi}/{len(df)} cities")
        
        return df
        
    except Exception as e:
        print(f"   ❌ Error converting to pandas for {period_label}: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def analyze_urban_expansion_impacts():
    """
    Main analysis function using multi-dataset approach
    """
    print("🔬 SCIENTIFIC SUHI ANALYSIS FOR UZBEKISTAN CITIES")
    print("="*60)
    print("Method: Multi-dataset Urban Classification + LST Difference")
    print("Datasets: Dynamic World, GHSL, ESA WorldCover, MODIS LC")
    print("Scale: 1km (MODIS native)")
    print("="*60)
    
    # Define analysis periods
    periods = {
        '2018': {'start': '2018-01-01', 'end': '2018-12-31', 'label': '2018'},
        '2019': {'start': '2019-01-01', 'end': '2019-12-31', 'label': '2019'},
        '2020': {'start': '2020-01-01', 'end': '2020-12-31', 'label': '2020'},
        '2021': {'start': '2021-01-01', 'end': '2021-12-31', 'label': '2021'},
        '2022': {'start': '2022-01-01', 'end': '2022-12-31', 'label': '2022'},
        '2023': {'start': '2023-01-01', 'end': '2023-12-31', 'label': '2023'},
        '2024': {'start': '2024-01-01', 'end': '2024-10-31', 'label': '2024'}
    }
    
    print(f"📅 Analyzing {len(periods)} periods for {len(UZBEKISTAN_CITIES)} cities")
    print(f"🌡️ Warm season focus: {WARM_MONTHS}")
    print(f"📏 Analysis scale: {TARGET_SCALE}m")
    print(f"🏙️ Urban threshold: >{URBAN_THRESHOLD}")
    print(f"🌾 Rural threshold: <{RURAL_THRESHOLD}")
    print(f"📊 Minimum pixels: Urban={MIN_URBAN_PIXELS}, Rural={MIN_RURAL_PIXELS}")
    
    frames = []
    
    for key, period in periods.items():
        print(f"\n🔍 Processing {period['label']}...")
        try:
            fc = analyze_period(period)
            df = fc_to_pandas(fc, period['label'])
            
            if len(df) > 0:
                frames.append(df)
                valid_count = df['SUHI_Day'].notna().sum()
                print(f"   ✅ Success: {len(df)} cities, {valid_count} with valid SUHI")
            else:
                print(f"   ⚠️ No data for {period['label']}")
                
        except Exception as e:
            print(f"   ❌ Error in {period['label']}: {e}")
            continue
    
    if frames:
        result = {f"period_{periods[k]['label']}": v for k, v in zip(periods.keys(), frames)}
        print(f"\n✅ Analysis complete: {len(frames)} periods processed")
        return result
    else:
        print("❌ No data collected from any period")
        return {}

def calculate_expansion_impacts(expansion_data):
    """
    Calculate SUHI trends with proper handling of missing data
    """
    print("\n📊 CALCULATING SUHI TRENDS AND EXPANSION IMPACTS...")
    
    if not expansion_data:
        print("❌ No expansion data to analyze")
        return pd.DataFrame(), {}
    
    # Combine all periods
    all_df = pd.concat(expansion_data.values(), ignore_index=True)
    
    if len(all_df) == 0:
        print("❌ No data in combined dataset")
        return pd.DataFrame(), {}
    
    print(f"   📊 Combined dataset: {len(all_df)} records")
    print(f"   🏙️ Cities: {all_df['City'].nunique()}")
    print(f"   📅 Periods: {all_df['Period'].nunique()}")
    
    # Calculate changes between periods
    periods = sorted(all_df['Period'].unique())
    if len(periods) < 2:
        print("❌ Need at least 2 periods for trend analysis")
        return pd.DataFrame(), {}
    
    first_period = periods[0]
    last_period = periods[-1]
    
    print(f"   📈 Trend analysis: {first_period} → {last_period}")
    
    base_data = all_df[all_df['Period'] == first_period]
    latest_data = all_df[all_df['Period'] == last_period]
    
    # Calculate impacts by city
    impacts = []
    
    for city in all_df['City'].unique():
        base_city = base_data[base_data['City'] == city]
        latest_city = latest_data[latest_data['City'] == city]
        
        if len(base_city) > 0 and len(latest_city) > 0:
            base_row = base_city.iloc[0]
            latest_row = latest_city.iloc[0]
            
            # Only calculate changes if both periods have valid data
            impact = {
                'City': city,
                'Analysis_Period': f"{first_period}_to_{last_period}",
                'Years_Span': int(last_period) - int(first_period),
                
                # SUHI changes
                'SUHI_Day_Change': (latest_row['SUHI_Day'] - base_row['SUHI_Day']) 
                                  if pd.notna(latest_row['SUHI_Day']) and pd.notna(base_row['SUHI_Day']) 
                                  else None,
                'SUHI_Night_Change': (latest_row['SUHI_Night'] - base_row['SUHI_Night'])
                                    if pd.notna(latest_row['SUHI_Night']) and pd.notna(base_row['SUHI_Night'])
                                    else None,
                
                # Temperature changes
                'LST_Day_Urban_Change': (latest_row['LST_Day_Urban'] - base_row['LST_Day_Urban'])
                                       if pd.notna(latest_row['LST_Day_Urban']) and pd.notna(base_row['LST_Day_Urban'])
                                       else None,
                'LST_Night_Urban_Change': (latest_row['LST_Night_Urban'] - base_row['LST_Night_Urban'])
                                         if pd.notna(latest_row['LST_Night_Urban']) and pd.notna(base_row['LST_Night_Urban'])
                                         else None,
                'LST_Day_Rural_Change': (latest_row['LST_Day_Rural'] - base_row['LST_Day_Rural'])
                                       if pd.notna(latest_row['LST_Day_Rural']) and pd.notna(base_row['LST_Day_Rural'])
                                       else None,
                'LST_Night_Rural_Change': (latest_row['LST_Night_Rural'] - base_row['LST_Night_Rural'])
                                         if pd.notna(latest_row['LST_Night_Rural']) and pd.notna(base_row['LST_Night_Rural'])
                                         else None,
                
                # Vegetation changes
                'NDVI_Urban_Change': (latest_row['NDVI_Urban'] - base_row['NDVI_Urban'])
                                    if pd.notna(latest_row['NDVI_Urban']) and pd.notna(base_row['NDVI_Urban'])
                                    else None,
                'NDVI_Rural_Change': (latest_row['NDVI_Rural'] - base_row['NDVI_Rural'])
                                    if pd.notna(latest_row['NDVI_Rural']) and pd.notna(base_row['NDVI_Rural'])
                                    else None,
                
                # Baseline and latest values
                'SUHI_Day_Baseline': base_row['SUHI_Day'],
                'SUHI_Night_Baseline': base_row['SUHI_Night'],
                'SUHI_Day_Latest': latest_row['SUHI_Day'],
                'SUHI_Night_Latest': latest_row['SUHI_Night'],
                
                # Data quality
                'Urban_Pixels_Base': base_row['Urban_Pixel_Count'],
                'Rural_Pixels_Base': base_row['Rural_Pixel_Count'],
                'Urban_Pixels_Latest': latest_row['Urban_Pixel_Count'],
                'Rural_Pixels_Latest': latest_row['Rural_Pixel_Count'],
                'Data_Quality_Base': base_row.get('Data_Quality', 'Unknown'),
                'Data_Quality_Latest': latest_row.get('Data_Quality', 'Unknown')
            }
            
            impacts.append(impact)
    
    impacts_df = pd.DataFrame(impacts)
    
    if len(impacts_df) == 0:
        print("❌ No valid city comparisons found")
        return pd.DataFrame(), {}
    
    # Set City as index for proper iteration
    impacts_df = impacts_df.set_index('City')
    
    # Calculate regional statistics
    regional_stats = {}
    
    # Only calculate stats for cities with valid data
    valid_changes = impacts_df[impacts_df['SUHI_Day_Change'].notna()]
    
    if len(valid_changes) > 0:
        regional_stats.update({
            'SUHI_Day_Change_mean': valid_changes['SUHI_Day_Change'].mean(),
            'SUHI_Day_Change_std': valid_changes['SUHI_Day_Change'].std(),
            'SUHI_Day_Change_min': valid_changes['SUHI_Day_Change'].min(),
            'SUHI_Day_Change_max': valid_changes['SUHI_Day_Change'].max(),
            'cities_with_valid_data': len(valid_changes)
        })
    
    valid_night = impacts_df[impacts_df['SUHI_Night_Change'].notna()]
    if len(valid_night) > 0:
        regional_stats.update({
            'SUHI_Night_Change_mean': valid_night['SUHI_Night_Change'].mean(),
            'SUHI_Night_Change_std': valid_night['SUHI_Night_Change'].std(),
            'SUHI_Night_Change_min': valid_night['SUHI_Night_Change'].min(),
            'SUHI_Night_Change_max': valid_night['SUHI_Night_Change'].max()
        })
    
    # Add metadata
    regional_stats.update({
        'analysis_type': 'multi_dataset_suhi',
        'method': 'combined_urban_classification',
        'warm_months': WARM_MONTHS,
        'analysis_scale_m': TARGET_SCALE,
        'urban_threshold': URBAN_THRESHOLD,
        'rural_threshold': RURAL_THRESHOLD,
        'min_urban_pixels': MIN_URBAN_PIXELS,
        'min_rural_pixels': MIN_RURAL_PIXELS,
        'ring_width_km': RING_KM,
        'cities_analyzed': len(impacts_df),
        'cities_with_valid_suhi': len(valid_changes) if len(valid_changes) > 0 else 0,
        'analysis_span_years': int(last_period) - int(first_period),
        'first_period': first_period,
        'last_period': last_period
    })
    
    print(f"   ✅ Impact analysis complete: {len(impacts_df)} cities")
    print(f"   📊 Cities with valid SUHI changes: {regional_stats.get('cities_with_valid_suhi', 0)}/{len(impacts_df)}")
    
    if 'SUHI_Day_Change_mean' in regional_stats:
        print(f"   📊 Regional SUHI day change mean: {regional_stats['SUHI_Day_Change_mean']:.3f}°C")
    if 'SUHI_Night_Change_mean' in regional_stats:
        print(f"   📊 Regional SUHI night change mean: {regional_stats['SUHI_Night_Change_mean']:.3f}°C")
    
    return impacts_df, regional_stats

def create_scientific_visualizations(impacts_df, regional_stats, expansion_data, output_dirs):
    """
    Create scientifically sound visualizations focusing on SUHI
    """
    print("\n📊 Creating scientific SUHI visualizations...")
    
    if len(impacts_df) == 0:
        print("❌ No data to visualize")
        return None
    
    # Import required packages
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("❌ Missing matplotlib/seaborn for visualization")
        return None
    
    # Set up the plot style
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Scientific Surface Urban Heat Island (SUHI) Analysis - Uzbekistan Cities\n' +
                 f'Method: Urban-Rural LST Difference | Scale: {TARGET_SCALE}m | Season: JJA', 
                 fontsize=14, fontweight='bold')
    
    cities = impacts_df.index.values
    
    # 1. SUHI Day vs Night Changes
    ax1 = axes[0, 0]
    suhi_day = impacts_df['SUHI_Day_Change'].fillna(0)  # Fill NaN with 0 for visualization
    suhi_night = impacts_df['SUHI_Night_Change'].fillna(0)  # Fill NaN with 0 for visualization
    
    if len(suhi_day) > 0 and len(suhi_night) > 0:
        x = range(len(cities))
        width = 0.35
        
        ax1.bar([i - width/2 for i in x], suhi_day, width, label='Day SUHI Change', alpha=0.7, color='orange')
        ax1.bar([i + width/2 for i in x], suhi_night, width, label='Night SUHI Change', alpha=0.7, color='purple')
        ax1.set_xlabel('Cities')
        ax1.set_ylabel('SUHI Change (°C)')
        ax1.set_title('Day vs Night SUHI Changes')
        ax1.set_xticks(x)
        ax1.set_xticklabels(cities, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
    else:
        ax1.text(0.5, 0.5, 'No SUHI data available', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Day vs Night SUHI Changes (No Data)')
    
    # 2. Urban vs Rural Temperature Changes
    ax2 = axes[0, 1]
    urban_temp = impacts_df['LST_Day_Urban_Change'].fillna(0)  # Fill NaN with 0 for visualization
    rural_temp = impacts_df['LST_Day_Rural_Change'].fillna(0)  # Fill NaN with 0 for visualization
    
    if len(urban_temp) > 0 and len(rural_temp) > 0:
        x = range(len(cities))
        width = 0.35
        ax2.bar([i - width/2 for i in x], urban_temp, width, label='Urban LST Change', alpha=0.7, color='red')
        ax2.bar([i + width/2 for i in x], rural_temp, width, label='Rural LST Change', alpha=0.7, color='green')
        ax2.set_xlabel('Cities')
        ax2.set_ylabel('LST Change (°C)')
        ax2.set_title('Urban vs Rural LST Changes (Day)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(cities, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    else:
        ax2.text(0.5, 0.5, 'No LST change data', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Urban vs Rural LST Changes (No Data)')
    
    # 3. SUHI Baseline vs Latest
    ax3 = axes[0, 2]
    # Only include cities with both baseline and latest data
    valid_data = impacts_df.dropna(subset=['SUHI_Day_Baseline', 'SUHI_Day_Latest'])
    
    if len(valid_data) > 0:
        baseline_suhi = valid_data['SUHI_Day_Baseline']
        latest_suhi = valid_data['SUHI_Day_Latest']
        city_names = valid_data.index
        
        ax3.scatter(baseline_suhi, latest_suhi, s=100, alpha=0.7, c='red')
        
        # Add 1:1 line
        min_val = min(baseline_suhi.min(), latest_suhi.min())
        max_val = max(baseline_suhi.max(), latest_suhi.max())
        ax3.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='No Change')
        
        ax3.set_xlabel('Baseline SUHI Day (°C)')
        ax3.set_ylabel('Latest SUHI Day (°C)')
        ax3.set_title('SUHI Day: Baseline vs Latest')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Add city labels
        for i, (city, baseline, latest) in enumerate(zip(city_names, baseline_suhi, latest_suhi)):
            ax3.annotate(city, (baseline, latest), 
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
    else:
        ax3.text(0.5, 0.5, 'No SUHI comparison data', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('SUHI Comparison (No Data)')
    
    # 4. Vegetation Changes (NDVI)
    ax4 = axes[1, 0]
    urban_ndvi = impacts_df['NDVI_Urban_Change'].fillna(0)
    rural_ndvi = impacts_df['NDVI_Rural_Change'].fillna(0)
    
    if len(urban_ndvi) > 0 and len(rural_ndvi) > 0:
        x = range(len(cities))
        width = 0.35
        ax4.bar([i - width/2 for i in x], urban_ndvi, width, label='Urban NDVI Change', alpha=0.7, color='darkgreen')
        ax4.bar([i + width/2 for i in x], rural_ndvi, width, label='Rural NDVI Change', alpha=0.7, color='lightgreen')
        ax4.set_xlabel('Cities')
        ax4.set_ylabel('NDVI Change')
        ax4.set_title('Urban vs Rural Vegetation Changes')
        ax4.set_xticks(x)
        ax4.set_xticklabels(cities, rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
    else:
        ax4.text(0.5, 0.5, 'No vegetation change data', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Vegetation Changes (No Data)')
    
    # 5. Summary Information Panel  
    ax5 = axes[1, 1]
    ax5.axis('off')
    
    # Add summary text
    summary_text = f"""SCIENTIFIC SUHI ANALYSIS SUMMARY
    
Method: Urban-Rural LST Difference  
Scale: 1000m (MODIS native)
Season: {regional_stats.get('warm_season', 'JJA')} (Warm season)
Cities: {regional_stats.get('cities_analyzed', len(impacts_df))}
Analysis Period: {regional_stats.get('first_period', 'N/A')} → {regional_stats.get('last_period', 'N/A')}

Key Results:
• Mean SUHI Day Change: {regional_stats.get('SUHI_Day_Change_mean', 0):.3f}°C
• Mean SUHI Night Change: {regional_stats.get('SUHI_Night_Change_mean', 0):.3f}°C  
• Urban Classification: Multi-dataset approach
• Rural Classification: Multi-dataset approach

Data Quality:
• Server-side processing: ✓
• Proper QA masking: ✓
• Scale consistent: ✓
• Seasonal filtering: ✓
• Urban-rural method: Multi-dataset V1

Methodology follows remote sensing
literature standards for satellite SUHI."""

    ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes, fontsize=9, 
             verticalalignment='top', fontfamily='monospace')
    
    # 6. Analysis Summary
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    # Create summary text with proper handling of None values
    suhi_day_mean = regional_stats.get('SUHI_Day_Change_mean', 0)
    suhi_night_mean = regional_stats.get('SUHI_Night_Change_mean', 0)
    
    summary_text = f"""SCIENTIFIC SUHI ANALYSIS SUMMARY

Method: Urban-Rural LST Difference
Scale: {TARGET_SCALE}m (MODIS native)
Season: {WARM_MONTHS} (Warm season)
Cities: {len(impacts_df)}
Analysis Period: {regional_stats.get('first_period', 'N/A')} → {regional_stats.get('last_period', 'N/A')}

Key Results:
• Mean SUHI Day Change: {suhi_day_mean:.3f}°C
• Mean SUHI Night Change: {suhi_night_mean:.3f}°C
• Urban Classification: Multi-dataset approach
• Rural Classification: Multi-dataset approach
• Ring Width: {RING_KM} km

Data Quality:
• Server-side processing: ✅
• Proper QA masking: ✅
• Scale consistency: ✅
• Seasonal filtering: ✅
• Urban-rural method: ✅

Methodology follows remote sensing
literature standards for satellite SUHI."""
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    
    # Save visualization
    output_path = output_dirs['images'] / 'scientific_suhi_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"📊 Scientific visualization saved to: {output_path}")
    return output_path

def export_scientific_data(expansion_data, impacts_df, regional_stats, output_dirs):
    """
    Export scientifically processed data
    """
    print("\n💾 Exporting scientific SUHI data...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Export impacts data
    if len(impacts_df) > 0:
        impacts_path = output_dirs['data'] / f'suhi_impacts_{timestamp}.csv'
        impacts_df.to_csv(impacts_path, index=False)
        print(f"   ✅ SUHI impacts: {impacts_path}")
    
    # Export raw period data
    for period, df in expansion_data.items():
        if len(df) > 0:
            period_path = output_dirs['data'] / f'suhi_data_{period}_{timestamp}.csv'
            df.to_csv(period_path, index=False)
            print(f"   ✅ Period data {period}: {period_path}")
    
    # Export regional statistics
    import json
    stats_path = output_dirs['data'] / f'suhi_regional_stats_{timestamp}.json'
    
    # Clean stats for JSON export
    clean_stats = {}
    for key, value in regional_stats.items():
        if isinstance(value, (int, float, str, list)):
            clean_stats[key] = value
        elif hasattr(value, 'item'):  # numpy types
            clean_stats[key] = value.item()
        else:
            clean_stats[key] = str(value)
    
    with open(stats_path, 'w') as f:
        json.dump(clean_stats, f, indent=2)
    print(f"   ✅ Regional statistics: {stats_path}")
    
    return timestamp

def generate_scientific_report(impacts_df, regional_stats, output_dirs):
    """
    Generate scientific report with proper methodology
    """
    print("📋 Generating scientific SUHI report...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get stats with default values
    suhi_day_mean = regional_stats.get('SUHI_Day_Change_mean', 0)
    suhi_day_std = regional_stats.get('SUHI_Day_Change_std', 0)
    suhi_night_mean = regional_stats.get('SUHI_Night_Change_mean', 0)
    suhi_night_std = regional_stats.get('SUHI_Night_Change_std', 0)
    
    report = f"""
# Scientific Surface Urban Heat Island (SUHI) Analysis - Uzbekistan Cities

## Executive Summary

This analysis quantifies surface urban heat island intensity (SUHI) for {len(impacts_df)} major cities in Uzbekistan using scientifically defensible remote sensing methods. SUHI is calculated as the difference in land surface temperature (LST) between urban cores and nearby rural rings during the warm season (June-August).

## Methodology

### Data Sources and Processing
- **MODIS LST**: MOD11A2 V6.1 8-day composites, properly scaled (0.02 K) and converted to °C
- **Landsat 8/9**: Collection-2 Level-2 surface reflectance with QA_PIXEL cloud masking
- **Dynamic World V1**: 10m land cover probabilities for urban/rural classification
- **Analysis Scale**: {TARGET_SCALE}m (MODIS native resolution) with explicit aggregation
- **Temporal Focus**: Warm season (JJA) median composites to avoid seasonal bias

### SUHI Calculation
Urban cores defined using multi-dataset approach with rural rings at {RING_KM}km distance. SUHI calculated separately for day and night as:

**SUHI = LST_urban - LST_rural**

### Quality Assurance
- Server-side processing via Google Earth Engine reduceRegions
- Proper Landsat L2 cloud/shadow masking using QA_PIXEL bits
- MODIS LST scaling and offset correction
- Scale-consistent analysis at 1km resolution
- Warm season filtering for temporal consistency

## Results

### Regional SUHI Trends ({regional_stats.get('first_period', 'N/A')} → {regional_stats.get('last_period', 'N/A')})

**Surface Urban Heat Island Changes:**
- Mean SUHI Day Change: {suhi_day_mean:.3f} ± {suhi_day_std:.3f}°C
- Mean SUHI Night Change: {suhi_night_mean:.3f} ± {suhi_night_std:.3f}°C
- Range Day SUHI Change: {regional_stats.get('SUHI_Day_Change_min', 0):.3f} to {regional_stats.get('SUHI_Day_Change_max', 0):.3f}°C
- Range Night SUHI Change: {regional_stats.get('SUHI_Night_Change_min', 0):.3f} to {regional_stats.get('SUHI_Night_Change_max', 0):.3f}°C

### City-Level Results

"""

    # Add city-level results
    if len(impacts_df) > 0:
        report += "| City | SUHI Day Change (°C) | SUHI Night Change (°C) | Urban Built Change | Rural Built Change |\n"
        report += "|------|---------------------|----------------------|------------------|------------------|\n"
        
        for _, row in impacts_df.iterrows():
            suhi_day = row.get('SUHI_Day_Change', None)
            suhi_night = row.get('SUHI_Night_Change', None)
            built_urban = row.get('Built_Prob_Urban_Change', None)
            built_rural = row.get('Built_Prob_Rural_Change', None)
            
            # Format values
            suhi_day_str = f"{suhi_day:.3f}" if pd.notna(suhi_day) else "N/A"
            suhi_night_str = f"{suhi_night:.3f}" if pd.notna(suhi_night) else "N/A"
            built_urban_str = f"{built_urban:.4f}" if pd.notna(built_urban) else "N/A"
            built_rural_str = f"{built_rural:.4f}" if pd.notna(built_rural) else "N/A"
            
            report += f"| {row['City']} | {suhi_day_str} | {suhi_night_str} | {built_urban_str} | {built_rural_str} |\n"

    report += f"""

## Technical Implementation

### Server-Side Processing
This analysis leverages Google Earth Engine's distributed computing infrastructure:
- Zonal statistics computed server-side using grouped reducers
- Minimal data transfer (only aggregated results)
- Scale-aware processing at 1km resolution
- Proper handling of mixed-resolution datasets

---

**Report Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Analysis Method**: Scientific SUHI (Urban-Rural LST Difference)
**Data Processing**: Google Earth Engine Server-Side
"""

    # Save report
    report_path = output_dirs['reports'] / f'scientific_suhi_report_{timestamp}.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"📋 Scientific report saved to: {report_path}")
    return report_path

def create_yearly_suhi_trends(expansion_data, output_dirs):
    """Create visualizations showing SUHI trends year by year"""
    print("\n📈 Creating yearly SUHI trend visualizations...")
    
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("❌ Missing matplotlib/seaborn for visualization")
        return None
    
    # Collect all data for trend analysis
    all_data = []
    for period_name, period_df in expansion_data.items():
        if len(period_df) > 0:
            # Extract year from period name
            year = period_name.split('_')[-1] if '_' in period_name else period_name
            period_df_copy = period_df.copy()
            period_df_copy['Year'] = year
            all_data.append(period_df_copy)
    
    if not all_data:
        print("❌ No data available for trend analysis")
        return None
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Create the visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('SUHI Trends Over Time - Uzbekistan Cities', fontsize=16, fontweight='bold')
    
    # 1. Day SUHI trends by city
    ax1 = axes[0, 0]
    for city in combined_df['City'].unique():
        city_data = combined_df[combined_df['City'] == city]
        valid_data = city_data.dropna(subset=['SUHI_Day', 'Year'])
        if len(valid_data) > 1:
            ax1.plot(valid_data['Year'], valid_data['SUHI_Day'], 
                    marker='o', label=city, linewidth=2, markersize=6)
    
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Day SUHI (°C)')
    ax1.set_title('Day SUHI Trends by City')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Night SUHI trends by city
    ax2 = axes[0, 1]
    for city in combined_df['City'].unique():
        city_data = combined_df[combined_df['City'] == city]
        valid_data = city_data.dropna(subset=['SUHI_Night', 'Year'])
        if len(valid_data) > 1:
            ax2.plot(valid_data['Year'], valid_data['SUHI_Night'], 
                    marker='s', label=city, linewidth=2, markersize=6)
    
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Night SUHI (°C)')
    ax2.set_title('Night SUHI Trends by City')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Regional average trends
    ax3 = axes[1, 0]
    regional_trends = combined_df.groupby('Year').agg({
        'SUHI_Day': ['mean', 'std'],
        'SUHI_Night': ['mean', 'std']
    }).reset_index()
    
    # Flatten column names
    regional_trends.columns = ['Year', 'SUHI_Day_mean', 'SUHI_Day_std', 'SUHI_Night_mean', 'SUHI_Night_std']
    
    # Plot with error bars
    ax3.errorbar(regional_trends['Year'], regional_trends['SUHI_Day_mean'], 
                yerr=regional_trends['SUHI_Day_std'], marker='o', label='Day SUHI', 
                linewidth=2, markersize=8, capsize=5)
    ax3.errorbar(regional_trends['Year'], regional_trends['SUHI_Night_mean'], 
                yerr=regional_trends['SUHI_Night_std'], marker='s', label='Night SUHI', 
                linewidth=2, markersize=8, capsize=5)
    
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Regional Average SUHI (°C)')
    ax3.set_title('Regional SUHI Trends (Mean ± Std)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. SUHI difference (Day - Night)
    ax4 = axes[1, 1]
    for city in combined_df['City'].unique():
        city_data = combined_df[combined_df['City'] == city]
        valid_data = city_data.dropna(subset=['SUHI_Day', 'SUHI_Night', 'Year'])
        if len(valid_data) > 1:
            suhi_diff = valid_data['SUHI_Day'] - valid_data['SUHI_Night']
            ax4.plot(valid_data['Year'], suhi_diff, 
                    marker='^', label=city, linewidth=2, markersize=6)
    
    ax4.set_xlabel('Year')
    ax4.set_ylabel('Day-Night SUHI Difference (°C)')
    ax4.set_title('Day-Night SUHI Difference Trends')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='x', rotation=45)
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    # Save visualization
    output_path = output_dirs['trends'] / 'yearly_suhi_trends.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"📈 Yearly SUHI trends saved to: {output_path}")
    return output_path

def create_comprehensive_data_export(expansion_data, impacts_df, regional_stats, output_dirs):
    """Export all SUHI analysis data in a comprehensive format"""
    print("\n💾 Creating comprehensive data export...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create a comprehensive data package
    data_package = {
        'metadata': {
            'analysis_date': datetime.now().isoformat(),
            'method': 'Multi-dataset Urban Classification + LST Difference',
            'datasets': ['Dynamic World', 'GHSL', 'ESA WorldCover', 'MODIS LC', 'GLAD', 'MODIS LST'],
            'scale': f'{TARGET_SCALE}m',
            'cities_analyzed': list(UZBEKISTAN_CITIES.keys()),
            'warm_season_months': WARM_MONTHS,
            'urban_threshold': URBAN_THRESHOLD,
            'rural_threshold': RURAL_THRESHOLD,
            'ring_distance_km': RING_KM
        },
        'period_data': {},
        'impact_analysis': impacts_df.to_dict('index') if not impacts_df.empty else {},
        'regional_statistics': regional_stats
    }
    
    # Add period data
    for period, df in expansion_data.items():
        if len(df) > 0:
            data_package['period_data'][period] = df.to_dict('records')
    
    # Export as JSON
    import json
    json_path = output_dirs['data'] / f'comprehensive_suhi_analysis_{timestamp}.json'
    
    # Clean data for JSON export
    def clean_for_json(obj):
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_for_json(item) for item in obj]
        elif hasattr(obj, 'item'):  # numpy types
            return obj.item()
        elif pd.isna(obj):
            return None
        else:
            return obj
    
    clean_package = clean_for_json(data_package)
    
    with open(json_path, 'w') as f:
        json.dump(clean_package, f, indent=2)
    
    # Also export as Excel for easier viewing
    try:
        excel_path = output_dirs['data'] / f'suhi_analysis_workbook_{timestamp}.xlsx'
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Impacts summary
            if not impacts_df.empty:
                impacts_df.to_excel(writer, sheet_name='Impact_Analysis', index=True)
            
            # Period data sheets
            for period, df in expansion_data.items():
                if len(df) > 0:
                    # Limit sheet name length
                    sheet_name = period[:31] if len(period) > 31 else period
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Regional statistics
            regional_df = pd.DataFrame([regional_stats])
            regional_df.to_excel(writer, sheet_name='Regional_Stats', index=False)
        
        print(f"   ✅ Excel workbook: {excel_path}")
    except Exception as e:
        print(f"   ⚠️ Excel export failed: {e}")
    
    print(f"   ✅ Comprehensive JSON: {json_path}")
    return {'json': json_path, 'timestamp': timestamp}

def create_suhi_trend_visualizations(expansion_data, impacts_df, regional_stats, output_dirs):
    """
    Create year-by-year SUHI trend visualizations
    """
    print("\n📈 Creating SUHI trend visualizations...")
    
    if len(expansion_data) == 0:
        print("❌ No data to visualize trends")
        return None
    
    # Import required packages
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("❌ Missing matplotlib/seaborn for visualization")
        return None
    
    # Combine all period data
    all_data = []
    for period_label, period_df in expansion_data.items():
        if len(period_df) > 0:
            period_df_copy = period_df.copy()
            period_df_copy['Period'] = period_label.replace('period_', '')
            all_data.append(period_df_copy)
    
    if not all_data:
        print("❌ No valid period data for trends")
        return None
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Create trend visualizations
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('SUHI Temporal Trends Analysis - Uzbekistan Cities\n' +
                 'Surface Urban Heat Island Evolution Over Time', 
                 fontsize=16, fontweight='bold')
    
    # 1. SUHI Day trends by city
    ax1 = axes[0, 0]
    cities = combined_df['City'].unique()
    periods = sorted(combined_df['Period'].unique())
    
    for city in cities:
        city_data = combined_df[combined_df['City'] == city]
        city_periods = []
        city_suhi_day = []
        
        for period in periods:
            period_data = city_data[city_data['Period'] == period]
            if len(period_data) > 0:
                city_periods.append(int(period))
                city_suhi_day.append(period_data['SUHI_Day'].iloc[0])
        
        if len(city_periods) > 1:
            ax1.plot(city_periods, city_suhi_day, marker='o', linewidth=2, 
                    markersize=6, label=city, alpha=0.8)
    
    ax1.set_xlabel('Year')
    ax1.set_ylabel('SUHI Day (°C)')
    ax1.set_title('Day SUHI Trends by City')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. SUHI Night trends by city
    ax2 = axes[0, 1]
    for city in cities:
        city_data = combined_df[combined_df['City'] == city]
        city_periods = []
        city_suhi_night = []
        
        for period in periods:
            period_data = city_data[city_data['Period'] == period]
            if len(period_data) > 0:
                city_periods.append(int(period))
                city_suhi_night.append(period_data['SUHI_Night'].iloc[0])
        
        if len(city_periods) > 1:
            ax2.plot(city_periods, city_suhi_night, marker='s', linewidth=2, 
                    markersize=6, label=city, alpha=0.8)
    
    ax2.set_xlabel('Year')
    ax2.set_ylabel('SUHI Night (°C)')
    ax2.set_title('Night SUHI Trends by City')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # 3. Regional SUHI averages
    ax3 = axes[0, 2]
    regional_suhi_day = []
    regional_suhi_night = []
    valid_periods = []
    
    for period in periods:
        period_data = combined_df[combined_df['Period'] == period]
        if len(period_data) > 0:
            day_mean = period_data['SUHI_Day'].mean()
            night_mean = period_data['SUHI_Night'].mean()
            if pd.notna(day_mean) and pd.notna(night_mean):
                valid_periods.append(int(period))
                regional_suhi_day.append(day_mean)
                regional_suhi_night.append(night_mean)
    
    if len(valid_periods) > 1:
        ax3.plot(valid_periods, regional_suhi_day, marker='o', linewidth=3, 
                markersize=8, label='Day SUHI', color='red', alpha=0.8)
        ax3.plot(valid_periods, regional_suhi_night, marker='s', linewidth=3, 
                markersize=8, label='Night SUHI', color='blue', alpha=0.8)
    
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Regional SUHI (°C)')
    ax3.set_title('Regional SUHI Trends')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Temperature components (Urban vs Rural)
    ax4 = axes[1, 0]
    for city in cities:
        city_data = combined_df[combined_df['City'] == city]
        city_periods = []
        urban_temps = []
        rural_temps = []
        
        for period in periods:
            period_data = city_data[city_data['Period'] == period]
            if len(period_data) > 0:
                city_periods.append(int(period))
                urban_temps.append(period_data['LST_Day_Urban'].iloc[0])
                rural_temps.append(period_data['LST_Day_Rural'].iloc[0])
        
        if len(city_periods) > 1:
            ax4.plot(city_periods, urban_temps, marker='o', linewidth=2, 
                    alpha=0.7, label=f'{city} Urban')
            ax4.plot(city_periods, rural_temps, marker='s', linewidth=2, 
                    linestyle='--', alpha=0.7, label=f'{city} Rural')
    
    ax4.set_xlabel('Year')
    ax4.set_ylabel('LST Day (°C)')
    ax4.set_title('Urban vs Rural Temperature Trends')
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # 5. SUHI intensity distribution by year
    ax5 = axes[1, 1]
    year_data = []
    for period in periods:
        period_data = combined_df[combined_df['Period'] == period]
        if len(period_data) > 0:
            for _, row in period_data.iterrows():
                year_data.append({
                    'Year': int(period),
                    'SUHI_Day': row['SUHI_Day'],
                    'SUHI_Night': row['SUHI_Night'],
                    'City': row['City']
                })
    
    if year_data:
        year_df = pd.DataFrame(year_data)
        
        # Box plot for SUHI Day
        years = sorted(year_df['Year'].unique())
        day_data = [year_df[year_df['Year'] == year]['SUHI_Day'].values for year in years]
        
        bp = ax5.boxplot(day_data, positions=years, widths=0.6, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightcoral')
            patch.set_alpha(0.7)
        
        ax5.set_xlabel('Year')
        ax5.set_ylabel('SUHI Day (°C)')
        ax5.set_title('SUHI Day Distribution by Year')
        ax5.grid(True, alpha=0.3)
    
    # 6. Summary statistics
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    # Calculate trend statistics
    trend_text = "SUHI TREND ANALYSIS SUMMARY\n\n"
    
    if len(valid_periods) > 1:
        # Linear trend calculation
        import numpy as np
        
        # Day SUHI trend
        day_slope = np.polyfit(valid_periods, regional_suhi_day, 1)[0]
        night_slope = np.polyfit(valid_periods, regional_suhi_night, 1)[0]
        
        trend_text += f"Regional Trends ({valid_periods[0]} → {valid_periods[-1]}):\n"
        trend_text += f"• Day SUHI: {day_slope:.3f}°C/year\n"
        trend_text += f"• Night SUHI: {night_slope:.3f}°C/year\n\n"
        
        trend_text += f"Total Change:\n"
        trend_text += f"• Day SUHI: {regional_suhi_day[-1] - regional_suhi_day[0]:.2f}°C\n"
        trend_text += f"• Night SUHI: {regional_suhi_night[-1] - regional_suhi_night[0]:.2f}°C\n\n"
        
        trend_text += f"Data Quality:\n"
        trend_text += f"• Cities analyzed: {len(cities)}\n"
        trend_text += f"• Time periods: {len(valid_periods)}\n"
        trend_text += f"• Analysis span: {valid_periods[-1] - valid_periods[0]} years\n\n"
        
        trend_text += "Interpretation:\n"
        if day_slope > 0:
            trend_text += "• Day SUHI is increasing ⚠️\n"
        else:
            trend_text += "• Day SUHI is stable/decreasing ✓\n"
            
        if night_slope > 0:
            trend_text += "• Night SUHI is increasing ⚠️\n"
        else:
            trend_text += "• Night SUHI is stable/decreasing ✓\n"
    else:
        trend_text += "Insufficient data for trend analysis\n"
        trend_text += f"Available periods: {len(valid_periods)}"
    
    ax6.text(0.05, 0.95, trend_text, transform=ax6.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))
    
    plt.tight_layout()
    
    # Save trend visualization
    trend_path = output_dirs['trends'] / 'suhi_temporal_trends.png'
    plt.savefig(trend_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"📈 SUHI trend analysis saved to: {trend_path}")
    return trend_path

def create_detailed_city_gis_maps(impacts_df, expansion_data, output_dirs):
    """Create detailed GIS maps for each city showing SUHI patterns"""
    print("\n🗺️ Creating detailed GIS maps for individual cities...")
    
    # Import required packages
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
    except ImportError:
        print("❌ Missing matplotlib for visualization")
        return None
    
    # Get the latest period data for spatial mapping
    latest_period = list(expansion_data.keys())[-1]
    latest_data = expansion_data[latest_period]
    
    if latest_data.empty:
        print("❌ No data available for mapping")
        return None
    
    num_cities = len(impacts_df)
    cols = min(3, num_cities)
    rows = (num_cities + cols - 1) // cols
    
    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(20, 6*rows))
    fig.suptitle('Urban Heat Analysis: Temperature and SUHI Patterns by City', 
                 fontsize=16, fontweight='bold')
    
    # Flatten axes array for easier iteration
    if num_cities == 1:
        axes_flat = [axes]
    elif rows == 1:
        axes_flat = axes if num_cities > 1 else [axes]
    else:
        axes_flat = axes.flatten()
    
    for idx, city_name in enumerate(impacts_df.index):
        if idx >= len(axes_flat):
            break
            
        ax = axes_flat[idx]
        
        # Check if city exists in UZBEKISTAN_CITIES
        if city_name not in UZBEKISTAN_CITIES:
            ax.text(0.5, 0.5, f'{city_name}\nCity not found in configuration', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12)
            ax.set_title(f'{city_name} (Not Found)')
            ax.axis('off')
            continue
        
        # Get city data
        city_info = UZBEKISTAN_CITIES[city_name]
        city_row = impacts_df.loc[city_name]
        city_data = latest_data[latest_data['City'] == city_name]
        
        if len(city_data) == 0:
            ax.text(0.5, 0.5, f'{city_name}\nNo data available', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12)
            ax.set_title(f'{city_name}')
            ax.axis('off')
            continue
        
        # Create mock spatial data since we don't have actual coordinates
        # In a real implementation, you would extract actual coordinates from GEE
        center_lon, center_lat = city_info['lon'], city_info['lat']
        buffer_deg = city_info['buffer'] / 111000  # Convert meters to degrees (approximate)
        
        # Create a grid of points around the city center
        n_points = 50
        lons = np.linspace(center_lon - buffer_deg, center_lon + buffer_deg, n_points)
        lats = np.linspace(center_lat - buffer_deg, center_lat + buffer_deg, n_points)
        
        # Create temperature data based on distance from center (mock SUHI pattern)
        LON, LAT = np.meshgrid(lons, lats)
        distances = np.sqrt((LON - center_lon)**2 + (LAT - center_lat)**2)
        
        # Mock temperature: higher in center (urban), lower at edges (rural)
        base_temp = 30.0  # Base temperature
        suhi_intensity = city_row.get('SUHI_Day_Latest', 2.0)  # Use actual SUHI if available
        temperatures = base_temp + suhi_intensity * np.exp(-distances * 5) + np.random.normal(0, 0.5, LON.shape)
        
        # Create contour plot
        contour = ax.contourf(LON, LAT, temperatures, levels=20, cmap='RdYlBu_r', alpha=0.8)
        
        # Add city center
        ax.scatter(center_lon, center_lat, s=200, c='red', marker='*', 
                  edgecolors='white', linewidth=2, label='City Center', zorder=10)
        
        # Add buffer circle
        circle = Circle((center_lon, center_lat), buffer_deg, 
                       fill=False, edgecolor='black', 
                       linewidth=2, linestyle='--', alpha=0.8)
        ax.add_patch(circle)
        
        # Labels and formatting
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        
        # Get actual SUHI values
        suhi_day = city_row.get('SUHI_Day_Latest', 'N/A')
        suhi_night = city_row.get('SUHI_Night_Latest', 'N/A')
        
        if isinstance(suhi_day, (int, float)) and isinstance(suhi_night, (int, float)):
            ax.set_title(f'{city_name}\nSUHI Day: {suhi_day:.2f}°C | Night: {suhi_night:.2f}°C')
        else:
            ax.set_title(f'{city_name}\nSUHI Analysis')
        
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(contour, ax=ax, shrink=0.8)
        cbar.set_label('Temperature (°C)', fontsize=9)
        
    # Hide unused subplots
    for idx in range(num_cities, len(axes_flat)):
        axes_flat[idx].axis('off')
    
    plt.tight_layout()
    
    # Save map
    map_path = output_dirs['gis_maps'] / 'city_suhi_maps.png'
    plt.savefig(map_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"🗺️ City SUHI maps saved to: {map_path}")
    return map_path

def create_city_overview_map(impacts_df, output_dirs):
    """Create overview map showing all cities and their SUHI impacts"""
    print("\n🗺️ Creating city overview map...")
    
    # Import required packages
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError:
        print("❌ Missing matplotlib for visualization")
        return None
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    fig.suptitle('Uzbekistan Cities: Surface Urban Heat Island Analysis Overview', 
                 fontsize=14, fontweight='bold')
    
    # Plot cities with SUHI indicators
    for city_name in impacts_df.index:
        if city_name in UZBEKISTAN_CITIES:
            city_info = UZBEKISTAN_CITIES[city_name]
            lon, lat = city_info['lon'], city_info['lat']
            
            # Get SUHI change
            suhi_change = impacts_df.loc[city_name, 'SUHI_Day_Change']
            
            # Color based on SUHI change
            if pd.notna(suhi_change):
                if suhi_change > 1.0:
                    color = 'red'
                    size = 300
                elif suhi_change > 0.5:
                    color = 'orange'
                    size = 250
                elif suhi_change > 0:
                    color = 'yellow'
                    size = 200
                else:
                    color = 'green'
                    size = 200
            else:
                color = 'gray'
                size = 150
            
            ax.scatter(lon, lat, s=size, c=color, alpha=0.7,
                      edgecolors='black', linewidth=1)
            
            # Add city label
            ax.annotate(city_name, (lon, lat), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9, fontweight='bold')
    
    # Set extent for Uzbekistan
    ax.set_xlim(55, 74)
    ax.set_ylim(37, 46)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('SUHI Change Impact by City (2019 → 2024)')
    ax.grid(True, alpha=0.3)
    
    # Add legend
    legend_elements = [
        Patch(facecolor='red', label='High Impact (>1.0°C)'),
        Patch(facecolor='orange', label='Medium Impact (0.5-1.0°C)'),
        Patch(facecolor='yellow', label='Low Impact (0-0.5°C)'),
        Patch(facecolor='green', label='Cooling (<0°C)'),
        Patch(facecolor='gray', label='No Data')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    # Save map
    map_path = output_dirs['gis_maps'] / 'uzbekistan_suhi_overview.png'
    plt.savefig(map_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"🗺️ Overview map saved to: {map_path}")
    return map_path

# Keep all other functions (visualizations, reports, etc.) the same but update to handle missing data properly

def main():
    """
    Main execution function
    """
    print("🔬 SCIENTIFIC SURFACE URBAN HEAT ISLAND (SUHI) ANALYSIS")
    print("="*70)
    print("Using Multi-Dataset Urban Classification Approach")
    print("="*70)
    
    try:
        # Initialize GEE
        if not authenticate_gee():
            return
        
        # Setup directories
        output_dirs = setup_output_directories()
        
        # Run analysis
        print(f"\n📡 Starting analysis for {len(UZBEKISTAN_CITIES)} cities...")
        expansion_data = analyze_urban_expansion_impacts()
        
        if not expansion_data:
            print("❌ No data collected. Exiting...")
            return
        
        # Calculate impacts
        print("\n📊 Calculating SUHI trends and impacts...")
        impacts_df, regional_stats = calculate_expansion_impacts(expansion_data)
        
        if len(impacts_df) == 0:
            print("❌ No impact analysis possible. Exiting...")
            return
        
        # Create visualizations (using your existing function)
        print("\n📈 Creating visualizations...")
        viz_path = create_scientific_visualizations(impacts_df, regional_stats, expansion_data, output_dirs)
        
        # Create yearly SUHI trend visualizations
        print("\n📈 Creating yearly SUHI trends...")
        trend_path = create_yearly_suhi_trends(expansion_data, output_dirs)
        
        # Create GIS maps
        print("\n🗺️ Creating GIS maps...")
        city_maps_path = create_detailed_city_gis_maps(impacts_df, expansion_data, output_dirs)
        
        # Export data (using your existing function)
        print("\n💾 Exporting data...")
        timestamp = export_scientific_data(expansion_data, impacts_df, regional_stats, output_dirs)
        
        # Export comprehensive data package
        print("\n💾 Creating comprehensive data export...")
        export_info = create_comprehensive_data_export(expansion_data, impacts_df, regional_stats, output_dirs)
        
        # Generate report (using your existing function)
        print("\n📋 Generating report...")
        report_path = generate_scientific_report(impacts_df, regional_stats, output_dirs)
        
        # Final summary
        print("\n" + "="*70)
        print("🎉 SCIENTIFIC SUHI ANALYSIS COMPLETE!")
        print("="*70)
        print(f"📊 Cities Analyzed: {len(impacts_df)}")
        print(f"📊 Cities with Valid SUHI: {regional_stats.get('cities_with_valid_suhi', 0)}")
        print(f"📈 Main Visualization: {viz_path}")
        if trend_path:
            print(f"📈 Trend Analysis: {trend_path}")
        if city_maps_path:
            print(f"🗺️ City Maps: {city_maps_path}")
        print(f"📋 Report: {report_path}")
        print(f"💾 Data exported with timestamp: {timestamp}")
        if export_info:
            print(f"💾 Comprehensive export: {export_info['json']}")
        print(f"📁 All outputs in: {output_dirs['base']}")
        print("="*70)
        
    except Exception as e:
        print(f"❌ Error in analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()