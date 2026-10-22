"""Data cleaning functions.

Every function takes a DataFrame, returns a DataFrame and writes what it did
into a CleaningLog, so the whole process can be audited afterwards.
The order of the steps matters and follows the workflow in the strategy document.
"""
import re

import numpy as np
import pandas as pd

import config


class CleaningLog:
    """Collects one line per cleaning action (step, issue, action, rows affected)."""

    def __init__(self):
        self.rows = []

    def add(self, step, issue, action, n):
        self.rows.append({"step": step, "issue": issue, "action": action, "records_affected": int(n)})

    def to_frame(self):
        return pd.DataFrame(self.rows)


# ---------------------------------------------------------------- Step 3
def standardize_placeholders(df, log):
    df = df.copy()
    total = 0
    for col in df.columns:
        s = df[col].astype("string").str.strip()
        mask = s.str.lower().isin(config.PLACEHOLDERS) | s.isna()
        newly = int(mask.sum()) - int(df[col].isna().sum())
        total += max(newly, 0)
        df[col] = s.mask(mask, pd.NA).astype(object)
    log.add("3 Standardize", "Placeholders (blank, N/A, -, null)", "Converted to real missing values", total)
    return df


def standardize_categories(df, log):
    df = df.copy()
    before = df["Crop"].nunique()
    key = df["Crop"].str.strip().str.lower()
    changed = int((df["Crop"] != key.map(config.CROP_MAP)).sum())
    df["Crop"] = key.map(config.CROP_MAP)
    unmapped = int(df["Crop"].isna().sum())
    log.add("3 Standardize", f"Crop spelling variants ({before} labels)",
            f"Mapped to {df['Crop'].nunique()} standard crop names", changed)
    if unmapped:
        log.add("3 Standardize", "Unknown crop names", "Set to missing for review", unmapped)
    d_before = int((df["District"].dropna() != df["District"].dropna().str.strip().str.title()).sum())
    df["District"] = df["District"].str.strip().str.title()
    log.add("3 Standardize", "District case / spaces", "Trimmed and title-cased", d_before)
    return df


def standardize_dates(df, log):
    df = df.copy()
    for col in ["Sowing_Date", "Harvest_Date"]:
        parsed = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
        for fmt in config.DATE_FORMATS:
            todo = parsed.isna() & df[col].notna()
            if not todo.any():
                break
            parsed[todo] = pd.to_datetime(df.loc[todo, col], format=fmt, errors="coerce")
        failed = int((parsed.isna() & df[col].notna()).sum())
        n_fmt = int(df[col].notna().sum())
        df[col] = parsed
        log.add("3 Standardize", f"{col}: {len(config.DATE_FORMATS)} text formats",
                "Parsed into one date type (YYYY-MM-DD)", n_fmt - failed)
        if failed:
            log.add("3 Standardize", f"{col}: unreadable dates", "Set to missing", failed)
    return df


def standardize_area(df, log):
    df = df.copy()
    pat = re.compile(r"^\s*([\d.,]+)\s*(ha|acre|sq m)\s*$", re.IGNORECASE)
    values, converted = [], 0
    for text in df["Area"]:
        m = pat.match(text) if isinstance(text, str) else None
        if not m:
            values.append(np.nan)
            continue
        number = float(m.group(1).replace(",", ""))
        unit = m.group(2).lower()
        if unit != "ha":
            converted += 1
        values.append(number * config.AREA_TO_HA[unit])
    df["Area_ha"] = np.round(values, 2)
    df = df.drop(columns="Area")
    log.add("3 Standardize", "Area in ha / acre / sq m", "Converted all to hectares (Area_ha)", converted)
    return df


def standardize_numbers(df, log):
    df = df.copy()
    cols = ["Yield_q_ha", "Rainfall_mm", "Soil_pH", "Fertilizer_kg", "Market_Price"]
    commas = int(df["Market_Price"].astype(str).str.contains(",", regex=False).sum())
    for col in cols:
        s = df[col].astype("string").str.replace(",", "", regex=False)
        df[col] = pd.to_numeric(s, errors="coerce").astype("float64")
    log.add("3 Standardize", "Numbers stored as text (e.g. '2,275')", "Commas removed, converted to numeric", commas)
    return df


# ---------------------------------------------------------------- Step 4
def remove_duplicates(df, log):
    n0 = len(df)
    exact = int(df.duplicated().sum())
    df = df.drop_duplicates()
    log.add("4 Duplicates", "Exact duplicate rows", "Removed", exact)
    # same Farm_ID but not identical (e.g. one copy has more missing values)
    df = df.assign(_filled=df.notna().sum(axis=1)).sort_values("_filled", ascending=False, kind="stable")
    key_dups = int(df.duplicated(subset="Farm_ID").sum())
    df = df.drop_duplicates(subset="Farm_ID", keep="first").drop(columns="_filled")
    df = df.sort_values("Farm_ID").reset_index(drop=True)
    log.add("4 Duplicates", "Same Farm_ID, different content", "Kept the most complete record", key_dups)
    return df


# ---------------------------------------------------------------- Step 5
def apply_validity_rules(df, log):
    df = df.copy()
    lo, hi = config.PH_RANGE
    bad = df["Soil_pH"].notna() & ~df["Soil_pH"].between(lo, hi)
    df.loc[bad, "Soil_pH"] = np.nan
    log.add("5 Validity", f"Soil pH outside {lo:g} to {hi:g}", "Set to missing", bad.sum())
    bad = df["Rainfall_mm"].notna() & (df["Rainfall_mm"] < config.RAINFALL_MIN)
    df.loc[bad, "Rainfall_mm"] = np.nan
    log.add("5 Validity", "Negative rainfall (e.g. -999 code)", "Set to missing", bad.sum())
    bad = df["Area_ha"].notna() & (df["Area_ha"] <= 0)
    df.loc[bad, "Area_ha"] = np.nan
    log.add("5 Validity", "Area equal to zero", "Set to missing", bad.sum())
    bad = df["Harvest_Date"].notna() & df["Sowing_Date"].notna() & (df["Harvest_Date"] <= df["Sowing_Date"])
    # a swapped pair: harvest is earlier than sowing; the pair can be repaired by swapping back
    swapped = bad & ((df["Sowing_Date"] - df["Harvest_Date"]).dt.days.between(60, 300))
    s, h = df.loc[swapped, "Sowing_Date"].copy(), df.loc[swapped, "Harvest_Date"].copy()
    df.loc[swapped, "Sowing_Date"], df.loc[swapped, "Harvest_Date"] = h, s
    log.add("5 Validity", "Harvest date before sowing date", "Swapped back (crop cycle 60-300 days)", swapped.sum())
    still_bad = bad & ~swapped
    df.loc[still_bad, "Harvest_Date"] = pd.NaT
    log.add("5 Validity", "Harvest date still invalid", "Set to missing", still_bad.sum())
    return df


# ---------------------------------------------------------------- Step 6
def _fill_by_group(df, col, group_sets):
    """Fill missing values with the group median, trying finer groups first."""
    for groups in group_sets:
        med = df.groupby(groups)[col].transform("median")
        df[col] = df[col].fillna(med)
    return df


def handle_missing(df, log):
    df = df.copy()
    # 6a. rows we cannot rescue
    n = int(df["Yield_q_ha"].isna().sum())
    df = df[df["Yield_q_ha"].notna()]
    log.add("6 Missing", "Target (Yield) missing", "Row dropped (cannot invent the target)", n)
    n = int(df["Area_ha"].isna().sum())
    df = df[df["Area_ha"].notna()].copy()
    log.add("6 Missing", "Area missing or invalid", "Row dropped (needed for per-hectare features)", n)
    # 6b. columns with too many gaps
    share = df.isna().mean()
    too_many = share[share > config.DROP_COLUMN_IF_MISSING_ABOVE].index.tolist()
    if too_many:
        df = df.drop(columns=too_many)
    log.add("6 Missing", f"Columns above {config.DROP_COLUMN_IF_MISSING_ABOVE:.0%} missing", f"Dropped: {too_many or 'none'}", len(too_many))
    # 6c. flags first, then fill
    for col in ["Rainfall_mm", "Soil_pH", "Fertilizer_kg", "Market_Price", "Harvest_Date"]:
        df[col + "_was_missing"] = df[col].isna().astype(int)
    n = int(df["District"].isna().sum())
    df["District"] = df["District"].fillna("Unknown")
    log.add("6 Missing", "District missing", "Filled with 'Unknown'", n)
    df["Year"] = df["Sowing_Date"].dt.year
    n = int(df["Rainfall_mm"].isna().sum())
    df = _fill_by_group(df, "Rainfall_mm", [["District", "Season"], ["Season"]])
    log.add("6 Missing", "Rainfall missing", "Median of same District + Season", n)
    n = int(df["Soil_pH"].isna().sum())
    df = _fill_by_group(df, "Soil_pH", [["District"], ["Season"]])
    log.add("6 Missing", "Soil pH missing", "Median of same District", n)
    n = int(df["Market_Price"].isna().sum())
    df = _fill_by_group(df, "Market_Price", [["Crop", "Year"], ["Crop"]])
    log.add("6 Missing", "Market price missing", "Median of same Crop + Year", n)
    n = int(df["Fertilizer_kg"].isna().sum())
    df["Fertilizer_per_ha"] = df["Fertilizer_kg"] / df["Area_ha"]
    df = _fill_by_group(df, "Fertilizer_per_ha", [["Crop"]])
    df["Fertilizer_kg"] = df["Fertilizer_kg"].fillna((df["Fertilizer_per_ha"] * df["Area_ha"]).round(1))
    log.add("6 Missing", "Fertilizer missing", "Median dose per ha of same Crop x Area", n)
    n = int(df["Harvest_Date"].isna().sum())
    dur = (df["Harvest_Date"] - df["Sowing_Date"]).dt.days
    med = dur.groupby(df["Crop"]).transform("median")
    df["Harvest_Date"] = df["Harvest_Date"].fillna(df["Sowing_Date"] + pd.to_timedelta(med.round(), unit="D"))
    log.add("6 Missing", "Harvest date missing / invalid", "Sowing date + median duration of the crop", n)
    return df


# ---------------------------------------------------------------- Step 7
def iqr_fences(series, k=config.IQR_K):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def flag_outliers_iqr(df, col, by="Crop", k=config.IQR_K):
    """Return a boolean Series: True where the value is outside the crop-wise IQR fences."""
    flag = pd.Series(False, index=df.index)
    for _, g in df.groupby(by):
        lo, hi = iqr_fences(g[col], k)
        flag.loc[g.index] = (g[col] < lo) | (g[col] > hi)
    return flag


def treat_outliers(df, log):
    """Two-tier rule.
    Tier 1: beyond the IQR fence AND more than 3x (or less than 1/3 of) the crop median
            -> almost surely a typing error -> replace with the crop median.
    Tier 2: beyond the fence but plausible -> real extreme value -> cap at the fence.
    """
    df = df.copy()
    for col, label in [("Yield_q_ha", "Yield"), ("Market_Price", "Market price")]:
        flag = flag_outliers_iqr(df, col)
        med = df.groupby("Crop")[col].transform("median")
        error = flag & ((df[col] > config.ERROR_FACTOR * med) | (df[col] < med / config.ERROR_FACTOR))
        cap = flag & ~error
        df[col + "_was_outlier"] = flag.astype(int)
        df.loc[error, col] = med[error]
        for crop, g in df.groupby("Crop"):
            lo, hi = iqr_fences(g.loc[~error.loc[g.index], col])
            idx = g.index[cap.loc[g.index]]
            df.loc[idx, col] = df.loc[idx, col].clip(lo, hi)
        log.add("7 Outliers", f"{label}: extreme typing errors", "Replaced with crop median", error.sum())
        log.add("7 Outliers", f"{label}: extreme but plausible values", "Capped at IQR fence", cap.sum())
    return df


# ---------------------------------------------------------------- Step 8
def add_features(df, log):
    df = df.copy()
    df["Fertilizer_per_ha"] = (df["Fertilizer_kg"] / df["Area_ha"]).round(2)
    df["Crop_Duration_days"] = (df["Harvest_Date"] - df["Sowing_Date"]).dt.days
    df["Sowing_Month"] = df["Sowing_Date"].dt.month
    df["Market_Price_log"] = np.log(df["Market_Price"]).round(4)
    log.add("8 Features", "Derived variables", "Fertilizer_per_ha, Crop_Duration_days, Sowing_Month, Market_Price_log", len(df))
    return df


# ---------------------------------------------------------------- pipeline
def clean(raw):
    """Run steps 3-8 in order. Returns (clean_df, log, snapshots)."""
    log = CleaningLog()
    snaps = {}
    df = standardize_placeholders(raw, log)
    df = standardize_categories(df, log)
    df = standardize_dates(df, log)
    df = standardize_area(df, log)
    df = standardize_numbers(df, log)
    snaps["after_standardize"] = df.copy()
    df = remove_duplicates(df, log)
    snaps["after_dedupe"] = df.copy()
    df = apply_validity_rules(df, log)
    snaps["after_validity"] = df.copy()
    df = handle_missing(df, log)
    snaps["before_outliers"] = df.copy()
    df = treat_outliers(df, log)
    df = add_features(df, log)
    return df, log, snaps
