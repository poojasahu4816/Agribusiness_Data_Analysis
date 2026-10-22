"""Small unit tests. Run from the repository root with:  python -m unittest discover tests"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import cleaning  # noqa: E402


class TestCleaning(unittest.TestCase):
    def setUp(self):
        self.log = cleaning.CleaningLog()

    def test_placeholders_become_missing(self):
        df = pd.DataFrame({"Rainfall_mm": ["650", "N/A", "-", "", "null", " 700 "]})
        out = cleaning.standardize_placeholders(df, self.log)
        self.assertEqual(int(out["Rainfall_mm"].isna().sum()), 4)

    def test_crop_variants_are_merged(self):
        df = pd.DataFrame({"Crop": ["WHEAT", " wheat", "Paddy", "RICE", "Corn"], "District": ["a", "b", "c", "d", "e"]})
        out = cleaning.standardize_categories(df, self.log)
        self.assertEqual(sorted(out["Crop"].unique()), ["Maize", "Rice", "Wheat"])

    def test_area_units_converted_to_hectares(self):
        df = pd.DataFrame({"Area": ["2 ha", "1 acre", "10000 sq m", "bad text"]})
        out = cleaning.standardize_area(df, self.log)
        self.assertAlmostEqual(out["Area_ha"].iloc[0], 2.0)
        self.assertAlmostEqual(out["Area_ha"].iloc[1], 0.40, places=2)
        self.assertAlmostEqual(out["Area_ha"].iloc[2], 1.0)
        self.assertTrue(np.isnan(out["Area_ha"].iloc[3]))

    def test_dates_with_three_formats(self):
        df = pd.DataFrame({"Sowing_Date": ["2023-06-20", "20/06/2023", "20 Jun 23"],
                           "Harvest_Date": ["2023-10-01", "01/10/2023", "01 Oct 23"]})
        out = cleaning.standardize_dates(df, self.log)
        self.assertEqual(out["Sowing_Date"].nunique(), 1)
        self.assertEqual(out["Harvest_Date"].nunique(), 1)

    def test_duplicates_removed_keep_most_complete(self):
        df = pd.DataFrame({"Farm_ID": ["F1", "F1", "F2", "F2"], "Rainfall_mm": [np.nan, 600.0, 500.0, 500.0]})
        out = cleaning.remove_duplicates(df, self.log)
        self.assertEqual(len(out), 2)
        self.assertEqual(out.loc[out.Farm_ID == "F1", "Rainfall_mm"].iloc[0], 600.0)

    def test_invalid_ph_set_to_missing(self):
        df = pd.DataFrame({"Soil_pH": [6.5, 15.2, 7.0], "Rainfall_mm": [100.0, 100.0, -999.0], "Area_ha": [1.0, 1.0, 1.0],
                           "Sowing_Date": pd.to_datetime(["2023-06-20"] * 3), "Harvest_Date": pd.to_datetime(["2023-10-20"] * 3)})
        out = cleaning.apply_validity_rules(df, self.log)
        self.assertTrue(np.isnan(out["Soil_pH"].iloc[1]))
        self.assertTrue(np.isnan(out["Rainfall_mm"].iloc[2]))

    def test_iqr_flags_digit_slip(self):
        df = pd.DataFrame({"Crop": ["Wheat"] * 8, "Yield_q_ha": [18, 20, 22, 25, 28, 30, 32, 120]})
        flag = cleaning.flag_outliers_iqr(df, "Yield_q_ha")
        self.assertEqual(flag.tolist(), [False] * 7 + [True])
        lo, hi = cleaning.iqr_fences(df["Yield_q_ha"])
        self.assertAlmostEqual(lo, 8.0)
        self.assertAlmostEqual(hi, 44.0)


if __name__ == "__main__":
    unittest.main()
