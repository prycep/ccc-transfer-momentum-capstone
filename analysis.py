#!/usr/bin/env python3
"""Create the modeling panel and compare OLS with KNN on held-out cohorts."""

from __future__ import annotations

import argparse
import csv
import gzip
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np


METRICS = {
    "122": ("cohort_size", "1 year", "VALUE"),
    "408": ("course_success", "1 year", "PERC"),
    "428": ("avg_successful_units", "1 year", "VALUE"),
    "453": ("persistence", "1 year", "PERC"),
    "458": ("completed_30_units", "1 year", "PERC"),
    "501": ("math_english", "1 year", "PERC"),
    "650": ("transfer_4yr", "4 year", "PERC"),
}
FEATURES = [
    "course_success",
    "avg_successful_units",
    "persistence",
    "completed_30_units",
    "math_english",
    "log_cohort_size",
]
NONSTANDARD_LOCALES = {
    "Los Angeles ITV",
    "Marin Continuing",
    "North Orange Continuing Education",
    "Rancho Santiago CED",
    "San Diego College of Continuing Education",
    "San Francisco Centers",
    "Santa Barbara Continuing",
}
MISSING = {"", "\\N", "NA", "N/A", "null", "None"}


def number(value: str | None) -> float:
    if value is None or value.strip() in MISSING:
        return math.nan
    return float(value)


def is_overall(row: dict[str, str]) -> bool:
    return (
        row.get("PROGRAM_TYPE") == "All Programs"
        and row.get("PROGRAM_NAME") == "All Programs"
        and row.get("DISAGG1_LABEL") == "Overall"
        and row.get("SUBGROUP1_LABEL") == "All"
        and row.get("DISAGG2_LABEL") == "Overall"
        and row.get("SUBGROUP2_LABEL") == "All"
    )


def build_panel(input_dir: Path) -> list[dict[str, object]]:
    files = sorted(input_dir.glob("*.csv.gz"))
    if not files:
        raise FileNotFoundError(f"No .csv.gz files found in {input_dir}")
    records: dict[tuple[str, str], dict[str, object]] = {}
    for path in files:
        with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                metric = row.get("METRIC_ID", "")
                if metric not in METRICS or not is_overall(row):
                    continue
                variable, timeframe, source_field = METRICS[metric]
                if row.get("SUBJOURNEY_LABEL") != timeframe:
                    continue
                college = row.get("LOCALE_NAME", "").strip()
                year = row.get("SCHOOL_YEAR", "").strip()
                if not college or not year:
                    continue
                key = (college, year)
                record = records.setdefault(
                    key,
                    {
                        "college": college,
                        "cohort_year": year,
                        "analysis_included": college not in NONSTANDARD_LOCALES,
                    },
                )
                value = number(row.get(source_field))
                old = record.get(variable)
                if old is not None and not (math.isnan(float(old)) and math.isnan(value)):
                    if not math.isclose(float(old), value, rel_tol=0, abs_tol=1e-12):
                        raise ValueError(f"Conflicting duplicate for {college}, {year}, {variable}")
                record[variable] = value

    panel = []
    for record in records.values():
        for variable, _, _ in METRICS.values():
            record.setdefault(variable, math.nan)
        cohort_size = float(record["cohort_size"])
        record["log_cohort_size"] = math.log1p(cohort_size) if math.isfinite(cohort_size) else math.nan
        required = FEATURES + ["transfer_4yr"]
        record["complete_case"] = all(math.isfinite(float(record[x])) for x in required)
        panel.append(record)
    return sorted(panel, key=lambda r: (str(r["cohort_year"]), str(r["college"])))


def write_rows(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def matrices(rows: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([[float(row[f]) for f in FEATURES] for row in rows], dtype=float)
    y = np.asarray([float(row["transfer_4yr"]) for row in rows], dtype=float)
    return x, y


def standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    scale = x.std(axis=0, ddof=0)
    scale[scale == 0] = 1.0
    return mean, scale


def ols_fit(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(x)), x])
    return np.linalg.lstsq(design, y, rcond=None)[0]


def ols_predict(beta: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(x)), x]) @ beta


def knn_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, k: int) -> np.ndarray:
    k = min(k, len(x_train))
    predictions = []
    for row in x_test:
        distances = np.sum((x_train - row) ** 2, axis=1)
        nearest = np.argpartition(distances, k - 1)[:k]
        predictions.append(float(np.mean(y_train[nearest])))
    return np.asarray(predictions)


def scores(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    residual = y - pred
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual**2)))
    denominator = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - float(np.sum(residual**2)) / denominator if denominator > 0 else math.nan
    return {"mae": mae, "rmse": rmse, "r2": r2}


def fit_predict(train: list[dict[str, object]], test: list[dict[str, object]], k: int):
    x_train, y_train = matrices(train)
    x_test, y_test = matrices(test)
    mean, scale = standardize_fit(x_train)
    z_train, z_test = (x_train - mean) / scale, (x_test - mean) / scale
    beta = ols_fit(z_train, y_train)
    return y_test, ols_predict(beta, z_test), knn_predict(z_train, y_train, z_test, k), beta


def bootstrap_mae_difference(y: np.ndarray, linear: np.ndarray, knn: np.ndarray, draws: int = 5000):
    rng = np.random.default_rng(20260731)
    differences = np.empty(draws)
    n = len(y)
    for i in range(draws):
        idx = rng.integers(0, n, size=n)
        differences[i] = np.mean(np.abs(y[idx] - knn[idx])) - np.mean(np.abs(y[idx] - linear[idx]))
    return float(np.quantile(differences, 0.025)), float(np.quantile(differences, 0.975))


def main_figure(path: Path, predictions: list[dict[str, object]], metrics: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1100, 620
    margin, plot = 90, 440
    actual = [float(r["actual"]) for r in predictions]
    values = actual + [float(r["linear_prediction"]) for r in predictions] + [float(r["knn_prediction"]) for r in predictions]
    lo, hi = min(values), max(values)
    pad = max((hi - lo) * 0.08, 0.005)
    lo, hi = lo - pad, hi + pad

    def sx(v: float) -> float:
        return margin + (v - lo) / (hi - lo) * plot

    def sy(v: float) -> float:
        return margin + plot - (v - lo) / (hi - lo) * plot

    test_metrics = {r["model"]: r for r in metrics if r["split"] == "test_19-20"}
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#F8FAFC"/>',
        '<text x="55" y="45" font-family="Arial" font-size="26" font-weight="700" fill="#073B57">Held-out 2019-20 cohort: predicted vs. observed four-year transfer rate</text>',
        f'<rect x="{margin}" y="{margin}" width="{plot}" height="{plot}" fill="white" stroke="#CBD5E1"/>',
        f'<line x1="{sx(lo)}" y1="{sy(lo)}" x2="{sx(hi)}" y2="{sy(hi)}" stroke="#64748B" stroke-dasharray="7 6"/>',
    ]
    for row in predictions:
        pieces.append(f'<circle cx="{sx(float(row["actual"])):.2f}" cy="{sy(float(row["linear_prediction"])):.2f}" r="5" fill="#0EA5A4" fill-opacity="0.7"/>')
        pieces.append(f'<circle cx="{sx(float(row["actual"])):.2f}" cy="{sy(float(row["knn_prediction"])):.2f}" r="4" fill="#F97316" fill-opacity="0.65"/>')
    for i in range(6):
        v = lo + i * (hi - lo) / 5
        pieces.extend([
            f'<text x="{sx(v):.1f}" y="{margin+plot+28}" text-anchor="middle" font-family="Arial" font-size="13" fill="#334155">{v:.0%}</text>',
            f'<text x="{margin-12}" y="{sy(v)+5:.1f}" text-anchor="end" font-family="Arial" font-size="13" fill="#334155">{v:.0%}</text>',
        ])
    pieces.extend([
        f'<text x="{margin+plot/2}" y="{height-28}" text-anchor="middle" font-family="Arial" font-size="16" fill="#0F172A">Observed transfer rate</text>',
        f'<text x="22" y="{margin+plot/2}" transform="rotate(-90 22 {margin+plot/2})" text-anchor="middle" font-family="Arial" font-size="16" fill="#0F172A">Predicted transfer rate</text>',
        '<circle cx="610" cy="130" r="6" fill="#0EA5A4"/><text x="628" y="136" font-family="Arial" font-size="16">Linear regression</text>',
        '<circle cx="610" cy="170" r="6" fill="#F97316"/><text x="628" y="176" font-family="Arial" font-size="16">K-nearest neighbors</text>',
    ])
    y0 = 245
    for model, color in [("linear", "#0EA5A4"), ("knn", "#F97316")]:
        metric = test_metrics.get(model, {})
        pieces.extend([
            f'<rect x="600" y="{y0}" width="425" height="105" rx="12" fill="white" stroke="{color}" stroke-width="2"/>',
            f'<text x="625" y="{y0+32}" font-family="Arial" font-size="19" font-weight="700" fill="{color}">{"Linear regression" if model == "linear" else "K-nearest neighbors"}</text>',
            f'<text x="625" y="{y0+66}" font-family="Arial" font-size="17" fill="#0F172A">MAE: {float(metric.get("mae", math.nan)):.2%}   RMSE: {float(metric.get("rmse", math.nan)):.2%}   R²: {float(metric.get("r2", math.nan)):.2f}</text>',
        ])
        y0 += 130
    pieces.append('</svg>')
    path.write_text("\n".join(pieces), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=root / "data" / "raw_gz")
    parser.add_argument("--output-root", type=Path, default=root)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.output_root.resolve()
    panel = build_panel(args.input_dir.resolve())
    panel_columns = [
        "college", "cohort_year", "analysis_included", "complete_case", "cohort_size",
        "log_cohort_size", "course_success", "avg_successful_units", "persistence",
        "completed_30_units", "math_english", "transfer_4yr",
    ]
    write_rows(root / "data/processed/modeling_panel.csv", panel, panel_columns)
    eligible = [r for r in panel if r["analysis_included"] and r["complete_case"]]
    development = [r for r in eligible if r["cohort_year"] <= "17-18"]
    validation = [r for r in eligible if r["cohort_year"] == "18-19"]
    final_train = [r for r in eligible if r["cohort_year"] <= "18-19"]
    test = [r for r in eligible if r["cohort_year"] == "19-20"]
    if not development or not validation or not test:
        raise RuntimeError("Need complete cases in development, 18-19 validation, and 19-20 test cohorts")

    candidates = [k for k in (3, 5, 7, 10, 15, 20) if k <= len(development)]
    y_val, linear_val, _, _ = fit_predict(development, validation, candidates[0])
    x_dev, y_dev = matrices(development)
    x_val, _ = matrices(validation)
    mean, scale = standardize_fit(x_dev)
    z_dev, z_val = (x_dev - mean) / scale, (x_val - mean) / scale
    knn_validation = [(k, knn_predict(z_dev, y_dev, z_val, k)) for k in candidates]
    selected_k, knn_val = min(knn_validation, key=lambda item: scores(y_val, item[1])["mae"])

    y_test, linear_test, knn_test, beta = fit_predict(final_train, test, selected_k)
    metric_rows = []
    for split, y, pairs in [
        ("validation_18-19", y_val, [("linear", linear_val), ("knn", knn_val)]),
        ("test_19-20", y_test, [("linear", linear_test), ("knn", knn_test)]),
    ]:
        for model, pred in pairs:
            metric_rows.append({"split": split, "model": model, "selected_k": selected_k if model == "knn" else "", **scores(y, pred), "n": len(y)})
    low, high = bootstrap_mae_difference(y_test, linear_test, knn_test)
    for row in metric_rows:
        row["knn_minus_linear_mae_ci_low"] = low if row["split"] == "test_19-20" else ""
        row["knn_minus_linear_mae_ci_high"] = high if row["split"] == "test_19-20" else ""
    write_rows(root / "results/model_metrics.csv", metric_rows, list(metric_rows[0]))

    prediction_rows = []
    sizes = np.asarray([float(r["cohort_size"]) for r in test])
    q25 = float(np.quantile(sizes, 0.25))
    for row, actual, linear, knn in zip(test, y_test, linear_test, knn_test):
        prediction_rows.append({
            "college": row["college"], "cohort_year": row["cohort_year"], "cohort_size": row["cohort_size"],
            "size_group": "smallest_quartile" if float(row["cohort_size"]) <= q25 else "larger",
            "actual": actual, "linear_prediction": linear, "linear_residual": actual-linear,
            "knn_prediction": knn, "knn_residual": actual-knn,
        })
    write_rows(root / "results/test_predictions.csv", prediction_rows, list(prediction_rows[0]))

    coefficient_rows = [{"term": "intercept", "standardized_coefficient": beta[0]}]
    coefficient_rows.extend({"term": feature, "standardized_coefficient": value} for feature, value in zip(FEATURES, beta[1:]))
    write_rows(root / "results/standardized_linear_coefficients.csv", coefficient_rows, list(coefficient_rows[0]))

    residual_rows = []
    for model in ("linear", "knn"):
        field = f"{model}_residual"
        for group in ("smallest_quartile", "larger"):
            values = [float(r[field]) for r in prediction_rows if r["size_group"] == group]
            residual_rows.append({"model": model, "cohort_year": "19-20", "size_group": group, "n": len(values), "mean_residual": float(np.mean(values)), "mae": float(np.mean(np.abs(values)))})
    write_rows(root / "results/residual_summary.csv", residual_rows, list(residual_rows[0]))
    main_figure(root / "figures/main_results.svg", prediction_rows, metric_rows)
    print(f"Panel rows: {len(panel)}; complete analysis rows: {len(eligible)}")
    print(f"Selected KNN k on 18-19 validation cohort: {selected_k}")
    print(f"Held-out 19-20 colleges: {len(test)}")
    print(f"Results written under: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

