
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
Urban cores defined using Google Satellite Embedding V1 similarity analysis (threshold > 0.85), with rural rings at 25km distance (similarity threshold > 0.75). SUHI calculated separately for day and night as:

**SUHI = LST_urban - LST_rural**

### Quality Assurance
- Server-side processing via Google Earth Engine reduceRegions
- Proper Landsat L2 cloud/shadow masking using QA_PIXEL bits
- MODIS LST scaling and offset correction
- Scale-consistent analysis at 1km resolution
- Warm season filtering for temporal consistency

## Results

### Regional SUHI Trends (2019 → 2024)

**Surface Urban Heat Island Changes:**
- Mean SUHI Day Change: 0.086 ± 0.555°C
- Mean SUHI Night Change: 0.219 ± 0.895°C
- Range Day SUHI Change: -0.438 to 1.081°C
- Range Night SUHI Change: -1.082 to 1.450°C

### City-Level Results

| City | SUHI Day Change (°C) | SUHI Night Change (°C) | Urban Built Change | Rural Built Change |
|------|---------------------|----------------------|------------------|------------------|
| Tashkent | N/A | N/A | N/A | N/A |
| Nukus | N/A | N/A | N/A | N/A |
| Andijan | N/A | N/A | N/A | N/A |
| Bukhara | N/A | N/A | N/A | N/A |
| Jizzakh | -0.206 | 0.326 | N/A | N/A |
| Qarshi | N/A | N/A | N/A | N/A |
| Navoiy | -0.438 | -0.041 | N/A | N/A |
| Namangan | 0.036 | 1.450 | N/A | N/A |
| Samarkand | N/A | N/A | N/A | N/A |
| Termez | 1.081 | -1.082 | N/A | N/A |
| Gulistan | 0.322 | -0.257 | N/A | N/A |
| Nurafshon | N/A | N/A | N/A | N/A |
| Fergana | -0.278 | 0.919 | N/A | N/A |
| Urgench | N/A | N/A | N/A | N/A |


## Technical Implementation

### Server-Side Processing
This analysis leverages Google Earth Engine's distributed computing infrastructure:
- Zonal statistics computed server-side using grouped reducers
- Minimal data transfer (only aggregated results)
- Scale-aware processing at 1km resolution
- Proper handling of mixed-resolution datasets

---

**Report Generated**: 2025-08-12 12:54:12
**Analysis Method**: Scientific SUHI (Urban-Rural LST Difference)
**Data Processing**: Google Earth Engine Server-Side
