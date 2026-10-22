"""Create a HYPOTHETICAL (synthetic) farm-level agribusiness dataset.

Step 1: build a perfectly clean "ground truth" table.
Step 2: damage it on purpose with realistic problems (mixed formats, missing
        values, invalid values, outliers, duplicates).

Because we know the truth, we can later measure how well the cleaning
pipeline repaired the damage. The data is NOT real; it only imitates the kind
of problems found in real agricultural records.
"""
import numpy as np
import pandas as pd

import config

CROPS = {
    # crop: season, yield mean/sd (quintal per ha), price mean/sd (Rs per quintal),
    #       duration range (days), rainfall mean/sd (mm per season)
    "Wheat":   dict(season="Rabi",   y=(32, 4),  p=(2275, 120),  d=(120, 135), r=(55, 20)),
    "Gram":    dict(season="Rabi",   y=(11, 2),  p=(5440, 250),  d=(100, 115), r=(45, 18)),
    "Rice":    dict(season="Kharif", y=(38, 5),  p=(2183, 110),  d=(115, 140), r=(950, 140)),
    "Soybean": dict(season="Kharif", y=(12, 2),  p=(4600, 220),  d=(90, 110),  r=(880, 130)),
    "Maize":   dict(season="Kharif", y=(30, 4),  p=(2090, 100),  d=(95, 115),  r=(850, 130)),
    "Cotton":  dict(season="Kharif", y=(14, 3),  p=(7020, 300),  d=(150, 180), r=(800, 120)),
}
CROP_PROBS = [0.22, 0.12, 0.20, 0.20, 0.14, 0.12]
DISTRICTS = ["Indore", "Ujjain", "Dewas", "Nagpur", "Wardha", "Amravati"]

CROP_VARIANTS = {
    "Wheat": ["WHEAT", "wheat ", " Wheat"],
    "Gram": ["Chana", "GRAM", "gram"],
    "Rice": ["Paddy", "RICE", "rice"],
    "Soybean": ["Soyabean", "SOYBEAN", "soybean "],
    "Maize": ["Corn", "MAIZE", "maize"],
    "Cotton": ["COTTON", "cotton ", "Cotton "],
}
RAW_COLUMNS = ["Farm_ID", "District", "Crop", "Season", "Sowing_Date", "Harvest_Date",
               "Area", "Yield_q_ha", "Rainfall_mm", "Soil_pH", "Fertilizer_kg", "Market_Price"]
PLACEHOLDER_CHOICES = ["", "N/A", "-", "null"]


def make_truth(rng, n=config.N_FARMS):
    crop_names = list(CROPS)
    crop = rng.choice(crop_names, size=n, p=CROP_PROBS)
    spec = [CROPS[c] for c in crop]
    season = np.array([s["season"] for s in spec])
    year = rng.choice([2022, 2023, 2024], size=n)
    start = np.where(season == "Kharif", rng.integers(0, 35, n), rng.integers(0, 30, n))
    sow = []
    for yr, s, off in zip(year, season, start):
        base = pd.Timestamp(int(yr), 6, 15) if s == "Kharif" else pd.Timestamp(int(yr), 11, 1)
        sow.append(base + pd.Timedelta(days=int(off)))
    dur = np.array([rng.integers(s["d"][0], s["d"][1] + 1) for s in spec])
    area = np.clip(rng.lognormal(mean=np.log(1.8), sigma=0.5, size=n), 0.3, 12).round(2)
    yld = np.array([max(2.0, rng.normal(*s["y"])) for s in spec]).round(1)
    rain = np.array([max(0.0, rng.normal(*s["r"])) for s in spec]).round(0)
    ph = np.clip(rng.normal(7.0, 0.55, n), 5.5, 8.8).round(1)
    fert_ha = np.clip(rng.normal(110, 25, n), 30, None)
    price = np.array([rng.normal(*s["p"]) for s in spec]).round(0)
    sow = pd.to_datetime(sow)
    truth = pd.DataFrame({
        "Farm_ID": [f"F{i:04d}" for i in range(1, n + 1)],
        "District": rng.choice(DISTRICTS, size=n),
        "Crop": crop, "Season": season,
        "Sowing_Date": sow, "Harvest_Date": sow + pd.to_timedelta(dur, unit="D"),
        "Area_ha": area, "Yield_q_ha": yld, "Rainfall_mm": rain, "Soil_pH": ph,
        "Fertilizer_kg": (area * fert_ha).round(1), "Market_Price": price,
    })
    return truth


def render_raw(truth, rng):
    """Turn clean truth into messy text: mixed date formats, units, spellings."""
    n = len(truth)
    raw = pd.DataFrame({"Farm_ID": truth["Farm_ID"].values})
    # district: some upper-case / lower-case / trailing space
    d = truth["District"].astype(str).values.copy()
    for i in np.where(rng.random(n) < 0.10)[0]:
        d[i] = rng.choice([d[i].upper(), d[i].lower(), d[i] + " "])
    raw["District"] = d
    # crop spelling variants
    c = truth["Crop"].astype(str).values.copy()
    for i in np.where(rng.random(n) < 0.35)[0]:
        c[i] = rng.choice(CROP_VARIANTS[c[i]])
    raw["Crop"] = c
    raw["Season"] = truth["Season"].values
    # dates in three formats
    fmts = ["%Y-%m-%d", "%d/%m/%Y", "%d %b %y"]
    for col in ["Sowing_Date", "Harvest_Date"]:
        pick = rng.integers(0, 3, n)
        raw[col] = [truth[col].iloc[i].strftime(fmts[pick[i]]) for i in range(n)]
    # area in three units
    unit = rng.choice(["ha", "acre", "sq m"], size=n, p=[0.6, 0.3, 0.1])
    area = []
    for a, u in zip(truth["Area_ha"], unit):
        if u == "ha":
            area.append(f"{a:.2f} ha")
        elif u == "acre":
            area.append(f"{a / config.AREA_TO_HA['acre']:.2f} acre")
        else:
            area.append(f"{a / config.AREA_TO_HA['sq m']:.0f} sq m")
    raw["Area"] = area
    raw["Yield_q_ha"] = truth["Yield_q_ha"].astype(str).values
    raw["Rainfall_mm"] = truth["Rainfall_mm"].map(lambda v: f"{v:.0f}").values
    raw["Soil_pH"] = truth["Soil_pH"].astype(str).values
    raw["Fertilizer_kg"] = truth["Fertilizer_kg"].astype(str).values
    price = truth["Market_Price"].values
    raw["Market_Price"] = [f"{p:,.0f}" if rng.random() < 0.15 else f"{p:.0f}" for p in price]
    return raw[RAW_COLUMNS]


def make_raw(truth, rng):
    """Create the damaged raw table and a dictionary describing what was injected."""
    n = len(truth)
    raw = render_raw(truth, rng)
    ids = raw["Farm_ID"].values
    meta = {"n_base_rows": n}
    all_idx = np.arange(n)
    used = set()

    def pick(k, pool=None):
        pool = [i for i in (all_idx if pool is None else pool) if i not in used]
        chosen = rng.choice(pool, size=k, replace=False)
        used.update(chosen.tolist())
        return chosen

    # --- outliers (digit slips): yield x10, price x8 ---
    idx = pick(12)
    raw.loc[idx, "Yield_q_ha"] = (truth.loc[idx, "Yield_q_ha"] * 10).round(1).astype(str).values
    meta["yield_slip_ids"] = ids[idx].tolist()
    idx = pick(6)
    raw.loc[idx, "Market_Price"] = (truth.loc[idx, "Market_Price"] * 8).round(0).map(lambda v: f"{v:.0f}").values
    meta["price_spike_ids"] = ids[idx].tolist()

    # --- invalid values ---
    idx = pick(10)
    raw.loc[idx, "Soil_pH"] = rng.uniform(14.5, 18.0, size=10).round(1).astype(str)
    meta["invalid_ph_ids"] = ids[idx].tolist()
    idx = pick(6)
    raw.loc[idx, "Rainfall_mm"] = "-999"
    meta["invalid_rain_ids"] = ids[idx].tolist()
    idx = pick(5)
    raw.loc[idx, "Area"] = "0 ha"
    meta["zero_area_ids"] = ids[idx].tolist()
    idx = pick(10)
    sow, har = raw.loc[idx, "Sowing_Date"].values.copy(), raw.loc[idx, "Harvest_Date"].values.copy()
    raw.loc[idx, "Sowing_Date"], raw.loc[idx, "Harvest_Date"] = har, sow
    meta["swapped_date_ids"] = ids[idx].tolist()

    # --- missing values (true value stays in the truth table) ---
    miss_spec = {"Rainfall_mm": 0.08, "Soil_pH": 0.05, "Fertilizer_kg": 0.06,
                 "Market_Price": 0.07, "Yield_q_ha": 0.03, "District": 0.01}
    meta["missing_ids"] = {}
    for col, frac in miss_spec.items():
        idx = pick(int(round(frac * n)))
        raw.loc[idx, col] = rng.choice(PLACEHOLDER_CHOICES, size=len(idx))
        meta["missing_ids"][col] = ids[idx].tolist()

    # --- duplicates: 30 exact copies + 15 re-typed copies with other formats ---
    exact = raw.iloc[rng.choice(n, size=30, replace=False)]
    retyped_pos = rng.choice(n, size=15, replace=False)
    retyped = render_raw(truth.iloc[retyped_pos], rng)
    meta["n_exact_duplicates"] = len(exact)
    meta["n_retyped_duplicates"] = len(retyped)
    raw = pd.concat([raw, exact, retyped], ignore_index=True)
    raw = raw.iloc[rng.permutation(len(raw))].reset_index(drop=True)
    return raw, meta


def generate(seed=config.SEED):
    rng = np.random.default_rng(seed)
    truth = make_truth(rng)
    raw, meta = make_raw(truth, rng)
    return raw, truth, meta


if __name__ == "__main__":
    raw, truth, meta = generate()
    config.RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.TRUTH_PATH.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(config.RAW_PATH, index=False)
    truth.to_csv(config.TRUTH_PATH, index=False)
    print(f"Raw data   : {len(raw)} rows -> {config.RAW_PATH}")
    print(f"Ground truth: {len(truth)} rows -> {config.TRUTH_PATH}")
