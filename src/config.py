"""Central configuration for the data cleaning pipeline.

All thresholds, mappings and paths live here so that the whole process can be
reviewed (and changed) in one place instead of being hidden inside the code.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "farm_data_raw.csv"
TRUTH_PATH = ROOT / "data" / "reference" / "ground_truth.csv"
CLEAN_PATH = ROOT / "data" / "processed" / "farm_data_clean.csv"
TRAIN_PATH = ROOT / "data" / "processed" / "train_scaled.csv"
TEST_PATH = ROOT / "data" / "processed" / "test_scaled.csv"
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"

# Reproducibility
SEED = 42
N_FARMS = 1200

# Values that really mean "missing"
PLACEHOLDERS = {"", "n/a", "na", "-", "null", "none", "nan"}

# Spelling variants -> one standard crop name (keys are lower case)
CROP_MAP = {
    "wheat": "Wheat", "rice": "Rice", "paddy": "Rice",
    "soybean": "Soybean", "soyabean": "Soybean",
    "maize": "Maize", "corn": "Maize",
    "gram": "Gram", "chana": "Gram",
    "cotton": "Cotton",
}

# Area units -> hectares
AREA_TO_HA = {"ha": 1.0, "acre": 0.404686, "sq m": 0.0001}

# Date formats seen in the raw file (tried in this order)
DATE_FORMATS = ["%Y-%m-%d", "%d/%m/%Y", "%d %b %y"]

# Validity rules (agronomic common sense)
PH_RANGE = (0.0, 14.0)
RAINFALL_MIN = 0.0

# Missing value policy
DROP_COLUMN_IF_MISSING_ABOVE = 0.40   # 40 percent

# Outlier policy
IQR_K = 1.5          # fence = Q1 - k*IQR, Q3 + k*IQR
ERROR_FACTOR = 3.0   # beyond fence AND > 3x (or < 1/3 of) crop median -> entry error

# Modelling prep
TEST_SIZE = 0.20
SCALE_COLUMNS = ["Area_ha", "Rainfall_mm", "Soil_pH", "Fertilizer_per_ha",
                 "Crop_Duration_days", "Market_Price_log"]
CATEGORICAL_COLUMNS = ["Crop", "Season", "District"]
TARGET = "Yield_q_ha"
