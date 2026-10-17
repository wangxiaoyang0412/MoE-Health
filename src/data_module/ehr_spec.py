"""EHR variable specification (MIMIC-IV v1.0 itemids).

Chart, lab and procedure variables follow the HAIM lists (9 chart, 22 unique
lab, 10 procedure). Labs are restricted to blood specimens (matching on the
label alone would also pull urine glucose / hematocrit). Medications are an EHR
input in the paper; they are 12 input-event channels encoded as "active during
the hour".

Vital-sign bounds are plausibility limits, not normal ranges: dropping values
outside *normal* ranges (e.g. SBP > 140) would remove the abnormal values that
carry signal.
"""

from __future__ import annotations

__all__ = [
    "CHART_ITEMS",
    "CONTINUOUS_VARS",
    "EXTRA_CHART_ITEMS",
    "EXTRA_LAB_ITEMS",
    "INTERVAL_VARS",
    "LAB_ITEMS",
    "MED_ITEMS",
    "OUTPUT_ITEMS",
    "PROCEDURE_ITEMS",
    "STATIC_CATEGORICAL",
    "chart_rows",
    "continuous_vars",
    "lab_items",
]

# itemid -> (variable, exclusive lower bound, inclusive upper bound)
CHART_ITEMS: dict[int, tuple[str, float, float]] = {
    220045: ("heart_rate", 0.0, 300.0),
    220179: ("nibp_systolic", 0.0, 300.0),
    220180: ("nibp_diastolic", 0.0, 250.0),
    220181: ("nibp_mean", 0.0, 300.0),
    220210: ("resp_rate", 0.0, 100.0),
    220277: ("spo2", 0.0, 100.0),
    223900: ("gcs_verbal", 0.5, 5.0),
    220739: ("gcs_eye", 0.5, 4.0),
    223901: ("gcs_motor", 0.5, 6.0),
}

# variable -> blood-specimen lab itemids
LAB_ITEMS: dict[str, list[int]] = {
    "glucose": [50809, 50931, 52569],
    "potassium": [50971, 52610],
    "sodium": [50983, 52623],
    "chloride": [50902, 52535],
    "creatinine": [50912, 52546],
    "urea_nitrogen": [51006, 52647],
    "bicarbonate": [50882],
    "anion_gap": [50868, 52500],
    "hemoglobin": [50811, 51222, 51640],
    "hematocrit": [51221, 51638, 51639, 52028],
    "magnesium": [50960],
    "platelet_count": [51265, 51704],
    "phosphate": [50970],
    "wbc": [51301, 51755, 51756],
    "calcium_total": [50893],
    "mch": [51248],
    "rbc": [51279, 51711],
    "mchc": [51249],
    "mcv": [51250, 51691],
    "rdw": [51277],
    "neutrophils": [51256, 51697],
    "vancomycin": [51009],
}

PROCEDURE_ITEMS: dict[str, list[int]] = {
    "foley_catheter": [229351],
    "picc_line": [224264],
    "intubation": [224385],
    "peritoneal_dialysis": [225805],
    "bronchoscopy": [225400],
    "eeg": [221223],
    "dialysis_crrt": [225802],
    "dialysis_catheter": [224270],
    "chest_tube_removed": [227712],
    "hemodialysis": [225441],
}

MED_ITEMS: dict[str, list[int]] = {
    "norepinephrine": [221906],
    "epinephrine": [221289, 229617],
    "vasopressin": [222315],
    "phenylephrine": [221749, 229630, 229631, 229632],
    "dopamine": [221662],
    "dobutamine": [221653],
    "propofol": [222168],
    "midazolam": [221668],
    "fentanyl": [221744, 225942, 225972],
    "dexmedetomidine": [225150, 229420],
    "insulin_regular": [223258],
    "heparin": [225152, 225975, 229597, 230044],
}

# Column order of the continuous value tensor and of the binary interval tensor.
CONTINUOUS_VARS: list[str] = [v for v, _, _ in CHART_ITEMS.values()] + list(LAB_ITEMS)
INTERVAL_VARS: list[str] = list(PROCEDURE_ITEMS) + list(MED_ITEMS)

STATIC_CATEGORICAL: list[str] = ["gender", "ethnicity", "insurance", "marital_status", "language"]

# --- Extended variable set (ehr_version = "v2"): common ICU severity predictors missing from the
# HAIM list. itemid -> (variable, exclusive lower bound, inclusive upper bound, unit transform);
# bounds apply after the transform (f2c: Fahrenheit -> Celsius, frac2pct: FiO2 fraction -> %).
EXTRA_CHART_ITEMS: dict[int, tuple[str, float, float, str]] = {
    223762: ("temperature", 25.0, 45.0, "none"),
    223761: ("temperature", 25.0, 45.0, "f2c"),
    223835: ("fio2", 20.0, 100.0, "frac2pct"),
    220050: ("abp_systolic", 0.0, 300.0, "none"),
    220051: ("abp_diastolic", 0.0, 250.0, "none"),
    220052: ("abp_mean", 0.0, 300.0, "none"),
    220074: ("cvp", -20.0, 60.0, "none"),
    220339: ("peep", -0.1, 40.0, "none"),
    223834: ("o2_flow", -0.1, 70.0, "none"),
    226512: ("weight", 20.0, 300.0, "none"),
    224639: ("weight", 20.0, 300.0, "none"),
}

EXTRA_LAB_ITEMS: dict[str, list[int]] = {
    "lactate": [50813, 52442],
    "ph": [50820],
    "po2": [50821],
    "pco2": [50818],
    "base_excess": [50802],
    "bilirubin_total": [50885],
    "inr": [51237, 51675],
    "pt": [51274, 52921],
    "ptt": [51275, 52923],
    "alt": [50861],
    "ast": [50878],
    "albumin": [50862],
    "alkaline_phosphatase": [50863],
    "troponin_t": [51003],
    "lymphocytes": [51244, 51690],
    "monocytes": [51254],
    "eosinophils": [51200],
    "basophils": [51146],
    "crp": [50889],
    "creatine_kinase": [50910],
    "fibrinogen": [51214],
    "o2_saturation_bg": [50817],
    "ntprobnp": [50963],
    "lipase": [50956],
}

# outputevents summed per hour
OUTPUT_ITEMS: dict[str, list[int]] = {"urine_output": [226559, 226560, 226566, 226627, 226631, 226713, 227489]}


def chart_rows(version: str) -> list[tuple[int, str, float, float, str]]:
    """(itemid, variable, lower, upper, transform) rows of the chart-event map."""
    rows = [(i, v, lo, hi, "none") for i, (v, lo, hi) in CHART_ITEMS.items()]
    if version == "v2":
        rows += [(i, v, lo, hi, t) for i, (v, lo, hi, t) in EXTRA_CHART_ITEMS.items()]
    return rows


def lab_items(version: str) -> dict[str, list[int]]:
    return {**LAB_ITEMS, **EXTRA_LAB_ITEMS} if version == "v2" else dict(LAB_ITEMS)


def continuous_vars(version: str) -> list[str]:
    """Column order of the continuous value tensor for an EHR variable-set version."""
    if version == "v1":
        return list(CONTINUOUS_VARS)
    extra_chart = list(dict.fromkeys(v for v, *_ in EXTRA_CHART_ITEMS.values()))
    return list(CONTINUOUS_VARS) + extra_chart + list(EXTRA_LAB_ITEMS) + list(OUTPUT_ITEMS)
