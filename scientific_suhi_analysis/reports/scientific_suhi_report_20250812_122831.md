
# Scientific Surface Urban Heat Island (SUHI) Analysis - Uzbekistan Cities

## Executive Summary

This analysis quantifies surface urban heat island intensity (SUHI) for 14 major cities in Uzbekistan using scientifically defensible remote sensing methods. SUHI is calculated as the difference in land surface temperature (LST) between urban cores and nearby rural rings during the warm season (June-August).

## Methodology

### Data Sources and Processing
- **MODIS LST**: MOD11A2 V6.1 8-day composites, properly scaled (0.02 K) and converted to °C
- **Landsat 8/9**: Collection-2 Level-2 surface reflectance with QA_PIXEL cloud masking
- **Dynamic World V1**: 10m land cover probabilities for urban/rural classification
- **Analysis Scale**: 1000m (MODIS native resolution) with explicit aggregation
- **Temporal Focus**: Warm season (JJA) median composites to avoid seasonal bias

### SUHI Calculation
Urban cores defined using Dynamic World built-up probabilities (threshold > 0.05), with rural rings at 25km distance (built-up probability < 0.2, water < 0.1). SUHI calculated separately for day and night as:

**SUHI = LST_urban - LST_rural**

### Quality Assurance
- Server-side processing via Google Earth Engine reduceRegions
- Proper Landsat L2 cloud/shadow masking using QA_PIXEL bits
- MODIS LST scaling and offset correction
- Scale-consistent analysis at 1km resolution
- Warm season filtering for temporal consistency

## Results

### Regional SUHI Trends (2019 → 2025)

**Surface Urban Heat Island Changes:**
- Mean SUHI Day Change: -0.227 ± 0.631°C
- Mean SUHI Night Change: 0.349 ± 0.544°C
- Range Day SUHI Change: -1.031 to 1.080°C
- Range Night SUHI Change: -0.645 to 1.175°C

### City-Level Results

| City | SUHI Day Change (°C) | SUHI Night Change (°C) | Urban Built Change | Rural Built Change |
|------|---------------------|----------------------|------------------|------------------|
| Tashkent | N/A | N/A | N/A | 0.0075 |
| Nukus | -0.230 | 0.775 | 0.0016 | 0.0014 |
| Andijan | -0.223 | 0.631 | 0.0043 | 0.0013 |
| Bukhara | -0.330 | 0.613 | -0.0006 | 0.0001 |
| Jizzakh | 1.080 | 0.424 | 0.0088 | 0.0089 |
| Qarshi | -0.381 | 0.131 | 0.0120 | 0.0119 |
| Navoiy | N/A | N/A | N/A | 0.0221 |
| Namangan | N/A | N/A | N/A | 0.0011 |
| Samarkand | 0.198 | 0.241 | 0.0314 | 0.0096 |
| Termez | N/A | N/A | N/A | 0.0024 |
| Gulistan | -1.004 | -0.200 | 0.0225 | 0.0221 |
| Nurafshon | N/A | N/A | N/A | 0.0068 |
| Fergana | -1.031 | 1.175 | 0.0013 | 0.0009 |
| Urgench | -0.120 | -0.645 | -0.0054 | -0.0010 |


## Technical Implementation

### Server-Side Processing
This analysis leverages Google Earth Engine's distributed computing infrastructure:
- Zonal statistics computed server-side using grouped reducers
- Minimal data transfer (only aggregated results)
- Scale-aware processing at 1km resolution
- Proper handling of mixed-resolution datasets

---

**Report Generated**: 2025-08-12 12:28:31
**Analysis Method**: Scientific SUHI (Urban-Rural LST Difference)
**Data Processing**: Google Earth Engine Server-Side
