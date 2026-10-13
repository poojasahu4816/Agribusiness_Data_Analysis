# Agribusiness_Data_Analysis

A data science internship project that analyzes publicly available agribusiness data to understand crop yield and agricultural market trends in India, using R.

## Objective

- Identify reliable public data sources for crop production, prices, weather and soil.
- Analyze the key factors that affect crop yield (rainfall, fertilizer, pesticide, area, season, state).
- Build and compare models to predict crop yield.
- Present the results in simple, business-friendly language.

## Dataset

**India Crop Yield dataset** (Hugging Face), with columns: Year, State, Crop, Season, Area, Production, Annual Rainfall, Fertilizer, Pesticide and Yield.

Other planned sources: Agmarknet (mandi prices), data.gov.in, India Meteorological Department (weather), CACP (MSP) and FAOSTAT.

## Approach

1. **Data collection:** download datasets and keep an untouched raw copy.
2. **Data cleaning:** handle missing values, duplicates, outliers and inconsistent names.
3. **Exploratory data analysis:** trends, distributions and correlations.
4. **Modelling:** linear regression as a baseline, then Random Forest / Gradient Boosting.
5. **Evaluation:** RMSE, MAE and R-squared on a held-out test set.
6. **Reporting:** key findings, limitations and recommendations.

## Tools

- R, RStudio
- tidyverse (dplyr, tidyr, readr), ggplot2, lubridate
- forecast, caret / tidymodels, randomForest

## Repository Contents

| File | Description |
|------|-------------|
| `Agribusiness_Data_Analysis_Plan_Week1.docx` | Week 1 report: data sources, indicators, methods and timeline |

More files (data, R scripts and later reports) will be added as the project progresses.

## Timeline

| Week | Focus | Status |
|------|-------|--------|
| 1 | Data analysis planning | Done |
| 2 | Model design and algorithm selection | Planned |
| 3 | Data cleaning and exploratory analysis | Planned |
| 4 | Model building and comparison | Planned |
| 5 | Final evaluation and report | Planned |

## Author

**Pooja Sahu**
Junior Data Scientist Intern (Agribusiness)
