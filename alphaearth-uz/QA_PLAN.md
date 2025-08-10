# QA PLAN

## Data QA
- Completeness, timeliness, CRS consistency
- Metadata present (source, date, license)
- Spot-check georegistration

## Model QA
- Train/val/test splits pinned
- Metrics reported (R², RMSE for regression; F1/IoU for classification)
- Feature importance + permutation checks

## Sensitivity & Uncertainty
- ±10–20% perturbations on top predictors
- Confidence/uncertainty maps exported

## Reproducibility
- Seeded runs
- 10% rerun consistency check
