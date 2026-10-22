"""Check the cleaning results against the known ground truth.

This only works because the dataset is synthetic: we know what every value
SHOULD have been, so we can measure how good each cleaning decision was.
"""
import numpy as np
import pandas as pd

import cleaning
import config


def _mae(a, b):
    return float(np.mean(np.abs(np.asarray(a, float) - np.asarray(b, float))))


def _rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


def evaluate(clean_df, snaps, truth, meta, log_df):
    truth = truth.set_index("Farm_ID")
    out = {}
    clean = clean_df.set_index("Farm_ID")
    before = snaps["before_outliers"].set_index("Farm_ID")

    # ---- 1. row accounting
    out["rows"] = {
        "raw_rows": meta["n_base_rows"] + meta["n_exact_duplicates"] + meta["n_retyped_duplicates"],
        "clean_rows": len(clean), "true_farms": len(truth),
        "duplicates_injected": meta["n_exact_duplicates"] + meta["n_retyped_duplicates"],
        "duplicates_removed": int(log_df[log_df.step == "4 Duplicates"].records_affected.sum()),
    }

    # ---- 2. accuracy of area conversion
    out["area_mae_ha"] = _mae(clean["Area_ha"], truth.loc[clean.index, "Area_ha"])

    # ---- 3. imputation accuracy vs simple baselines
    imp = {}
    after_validity = snaps["after_validity"].set_index("Farm_ID")
    for col, flag in [("Rainfall_mm", "Rainfall_mm_was_missing"), ("Soil_pH", "Soil_pH_was_missing"),
                      ("Market_Price", "Market_Price_was_missing"), ("Fertilizer_kg", "Fertilizer_kg_was_missing")]:
        idx = clean.index[clean[flag] == 1]
        observed = after_validity.loc[after_validity[col].notna(), col]
        t = truth.loc[idx, col]
        imp[col] = {
            "n_imputed": int(len(idx)),
            "mae_group_median_(used)": _mae(clean.loc[idx, col], t),
            "mae_global_median": _mae(np.full(len(idx), observed.median()), t),
            "mae_global_mean": _mae(np.full(len(idx), observed.mean()), t),
            "typical_value": float(truth[col].mean()),
        }
    out["imputation"] = imp

    # ---- 4. outlier detection: IQR vs Z-score on the same data
    det = {}
    for col, key in [("Yield_q_ha", "yield_slip_ids"), ("Market_Price", "price_spike_ids")]:
        # errors that actually reached this step (a slip can vanish if its clean duplicate was kept)
        still_wrong = {i for i in set(meta[key]) & set(before.index) if before.loc[i, col] != truth.loc[i, col]}
        injected = still_wrong
        iqr_flag = cleaning.flag_outliers_iqr(before.reset_index(), col).values
        iqr_ids = set(before.index[iqr_flag])
        k3_flag = cleaning.flag_outliers_iqr(before.reset_index(), col, k=3.0).values
        k3_ids = set(before.index[k3_flag])
        z = before.groupby("Crop")[col].transform(lambda s: (s - s.mean()) / s.std())
        z_ids = set(before.index[(z.abs() > 3).values])
        det[col] = {
            "injected_total": len(meta[key]), "injected_errors": len(injected),
            "iqr_found": len(injected & iqr_ids), "iqr_false_alarms": len(iqr_ids - injected),
            "iqr_k3_found": len(injected & k3_ids), "iqr_k3_false_alarms": len(k3_ids - injected),
            "z_found": len(injected & z_ids), "z_false_alarms": len(z_ids - injected),
        }
    out["outlier_detection"] = det

    # ---- 5. outlier treatment strategies (yield)
    col = "Yield_q_ha"
    b = before.copy()
    flag = pd.Series(cleaning.flag_outliers_iqr(b.reset_index(), col).values, index=b.index)
    cap = b[col].copy()
    for _, g in b.groupby("Crop"):
        lo, hi = cleaning.iqr_fences(g[col])
        cap.loc[g.index] = g[col].clip(lo, hi)
    variants = {
        "No treatment": b[col],
        "Remove flagged rows": b.loc[~flag, col],
        "Cap all at IQR fence": cap,
        "Two-tier rule (used)": clean[col],
    }
    genuine = (b[col] == truth.loc[b.index, col])          # value untouched by the injected errors
    affected = {
        "No treatment": 0,
        "Remove flagged rows": int((flag & genuine).sum()),                     # real rows thrown away
        "Cap all at IQR fence": int((genuine & (cap != b[col])).sum()),          # real values changed
        "Two-tier rule (used)": int((genuine & (clean[col].reindex(b.index) != b[col])).sum()),
    }
    strat = {}
    for name, s_ in variants.items():
        s_ = s_[s_.index.isin(truth.index)]
        t = truth.loc[s_.index, col]
        crop = truth.loc[s_.index, "Crop"]
        mean_err = (abs(s_.groupby(crop).mean() - t.groupby(crop).mean()) / t.groupby(crop).mean() * 100).mean()
        std_err = (abs(s_.groupby(crop).std() - t.groupby(crop).std()) / t.groupby(crop).std() * 100).mean()
        strat[name] = {"rows": int(len(s_)), "rmse": _rmse(s_, t), "mean_error_pct": float(mean_err),
                       "std_error_pct": float(std_err), "genuine_values_affected": affected[name]}
    out["outlier_strategies_yield"] = strat

    # ---- 6. mean yield by crop: truth vs naive vs cleaned (same farms)
    ids = clean.index
    rows = []
    for crop in sorted(truth["Crop"].unique()):
        i = [x for x in ids if truth.loc[x, "Crop"] == crop]
        tm = truth.loc[i, col].mean()
        nm = before.loc[i, col].mean()
        cm = clean.loc[i, col].mean()
        rows.append({"crop": crop, "n": len(i), "truth": float(tm), "naive": float(nm), "cleaned": float(cm),
                     "naive_err_pct": float((nm - tm) / tm * 100), "clean_err_pct": float((cm - tm) / tm * 100)})
    out["crop_mean_yield"] = rows

    # ---- 7. completeness before / after
    cols = ["District", "Area_ha", "Yield_q_ha", "Rainfall_mm", "Soil_pH", "Fertilizer_kg", "Market_Price", "Harvest_Date"]
    pre = snaps["after_validity"][cols].isna().mean() * 100
    post = clean_df[cols].isna().mean() * 100
    out["missing_pct"] = {c: {"before": float(pre[c]), "after": float(post[c])} for c in cols}
    return out
