#!/usr/bin/env python3
"""Extract public benchmark rows from SBER_BENCH.xlsx.

The script intentionally uses only Python's standard library. An .xlsx file is
an OpenXML zip archive, so this keeps the project deployable without pandas or
openpyxl.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "sber-bench.json"
SOURCE_SHEET_CANDIDATES = ("MAY OPTUNA", "RESULTS NEW from March")
NTP_SHEET = "NTP_LGBM"
SKIP_RAW_SHEETS = {"TS-FRESH EMB"}

NS = {
    "m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}

TASKS = [
    {"id": "regression", "label": "Regression", "metric": "R2", "scoreColumn": 2, "paramColumn": 6, "maskColumn": 10},
    {"id": "classification", "label": "Classification", "metric": "accuracy", "scoreColumn": 3, "paramColumn": 7, "maskColumn": 11},
    {"id": "forecasting", "label": "Forecasting", "metric": "R2", "scoreColumn": 4, "paramColumn": 8, "maskColumn": 12},
    {"id": "anomaly", "label": "Anomaly", "metric": "roc_auc", "scoreColumn": 5, "paramColumn": 9, "maskColumn": 13},
]

TEXT_REPLACEMENTS = {
    "\u0414\u0430\u0442\u0430\u0441\u0435\u0442\u044b \u0438 \u0437\u0430\u0434\u0430\u0447\u0438": "Datasets and tasks",
    "\u041a\u043e\u043c\u043c\u0435\u0442\u0430\u0440\u0438\u0439": "Comment",
    "\u041f\u0420\u041e\u0413\u0420\u0415\u0421\u0421": "Progress",
    "\u0417\u0430\u043f\u0443\u0449\u0435\u043d\u043e": "Running",
    "\u0417\u0430\u0432\u0435\u0440\u0448\u0435\u043d\u043e": "Completed",
    "\u0441\u0440\u0435\u0434\u043d\u0435\u0435": "mean",
    "\u043c\u0438\u043d\u0438\u043c\u0443\u043c": "minimum",
    "\u0434\u0430\u0442\u0430\u0441\u0435\u0442/\u0437\u0430\u0434\u0430\u0447\u0430": "dataset/task",
    "\u0441\u0443\u043c\u043c\u0430 \u043f\u043e\u043a\u0443\u043f\u043a\u0438 \u0437\u0430 \u0441\u043b\u0435\u0434\u0443\u044e\u0449\u0438\u0435 N \u0434\u043d\u0435\u0439": "purchase sum over the next N days",
    "\u043a\u043e\u043b\u0438\u0447\u0435\u0441\u0442\u0432\u043e \u0441\u043e\u0431\u044b\u0442\u0438\u0439 \u0432 \u043e\u043f\u0440\u0435\u0434\u0435\u043b\u0435\u043d\u043d\u044b\u0439 \u0434\u0435\u043d\u044c": "event count on a selected day",
    "\u0412\u043e\u0437\u0440\u0430\u0441\u0442 (4 \u043a\u043b\u0430\u0441\u0441\u0430)": "Age group (4 classes)",
    "\u041a\u0432\u0430\u043d\u0442\u0438\u043b\u044c": "Quantile",
    "\u0411\u044b\u0441\u0442\u0440\u0430\u044f \u043f\u043e\u043a\u0443\u043f\u043a\u0430": "Fast purchase",
    "\u0438\u0437\u043c\u0435\u043d\u0435\u043d\u0438\u0435 \u043f\u043e\u0442\u0440\u0435\u0431\u043b\u0435\u043d\u0438\u044f \u044d\u043d\u0435\u0440\u0433\u0438\u0438 \u043d\u0430 \u0441\u043b\u0435\u0434 \u0434\u0435\u043d\u044c": "next-day energy consumption change",
    "\u0422\u0438\u043f \u0443\u0441\u0442\u0440\u043e\u0439\u0441\u0442\u0432\u0430": "Device type",
    "\u043e\u0431\u044a\u0435\u043c \u043f\u0440\u043e\u0434\u0430\u0436 \u0437\u0430 \u0441\u043b\u0435\u0434\u0443\u044e\u0449\u0438\u0435 N \u0434\u043d\u0435\u0439": "sales volume over the next N days",
    "\u0441\u043a\u043e\u043b\u044c\u043a\u043e \u0442\u043e\u0432\u0430\u0440\u0430 \u043a\u0443\u043f\u044f\u0442": "future purchased quantity",
    "\u0422\u0438\u043f \u043c\u0430\u0433\u0430\u0437\u0438\u043d\u0430 (5 \u043a\u043b\u0430\u0441\u0441\u043e\u0432)": "Store type (5 classes)",
    "\u0415\u0434\u0438\u043d\u0441\u0442\u0432\u0435\u043d\u043d\u044b\u0439 \u043c\u0430\u0433\u0430\u0437\u0438\u043d \u0432 \u0433\u043e\u0440\u043e\u0434\u0435": "Only store in the city",
    "\u0441\u0443\u043c\u043c\u0430 \u043f\u0440\u043e\u0441\u043b\u0443\u0448\u0438\u0432\u0430\u043d\u0438\u0439 \u0437\u0430 \u0441\u043b\u0435\u0434\u0443\u044e\u0449\u0438\u0435 N \u0434\u043d\u0435\u0439": "listening total over the next N days",
    "\u0411\u0443\u0434\u0443\u0449\u0438\u0439 \u0436\u0430\u043d\u0440 (\u0447\u0442\u043e \u0432\u043a\u043b\u044e\u0447\u0438\u0442 \u0434\u0430\u043b\u044c\u0448\u0435)": "Future genre / next content category",
    "\u0420\u0430\u0437\u043d\u043e\u043e\u0431\u0440\u0430\u0437\u0438\u0435": "Diversity",
    "\u041a\u043e\u043b\u0438\u0447\u0435\u0441\u0442\u0432\u043e \u0443\u043f\u043e\u043c\u0438\u043d\u0430\u043d\u0438\u0439 \u043f\u043e\u043b\u044c\u0437\u043e\u0432\u0430\u0442\u0435\u043b\u0435\u0439 \u0432 \u043f\u043e\u0441\u0442\u0435": "Number of user mentions in the post",
    "\u0422\u043e\u043d\u0430\u043b\u044c\u043d\u043e\u0441\u0442\u044c \u0442\u0435\u043a\u0441\u0442\u0430 (Sentiment)": "Text sentiment",
    "\u041f\u0435\u0440\u0435\u0438\u0437\u0431\u044b\u0442\u043e\u043a \u0441\u043f\u0435\u0446\u0441\u0438\u043c\u0432\u043e\u043b\u043e\u0432 \u043d\u0430 \u0434\u043b\u0438\u043d\u0443": "Excess special-character ratio",
    "\u0440\u0435\u043a\u043e\u043c\u0435\u043d\u0434\u0430\u0446\u0438\u0438 (2 \u043a\u043b\u0430\u0441\u0441\u0430)": "recommendation source (2 classes)",
    "\u0438\u0437\u043c\u0435\u043d\u0435\u043d\u0438\u0435 \u0442\u0435\u043c\u043f\u0435\u0440\u0430\u0442\u0443\u0440\u044b \u043c\u0430\u0441\u043b\u0430": "oil temperature change",
    "\u041e\u043f\u043b\u0430\u0442\u0430 \u0431\u043e\u043b\u044c\u0448\u043e\u0439 \u0447\u0430\u0441\u0442\u0438 \u0442\u0440\u0430\u043d\u0437\u0430\u043a\u0446\u0438\u0438 \u0431\u0430\u043b\u043b\u0430\u043c\u0438": "large share of transaction paid with points",
    "\u0412\u044b\u0431\u043e\u0440 \u043a\u043e\u043d\u043a\u0440\u0435\u0442\u043d\u043e\u0433\u043e \u0431\u0430\u043d\u043a\u043e\u0432\u0441\u043a\u043e\u0433\u043e \u043f\u0440\u043e\u0434\u0443\u043a\u0442\u0430": "specific banking product choice",
    "\u0423\u0445\u043e\u0434 \u043a\u043b\u0438\u0435\u043d\u0442\u0430 \u0432 \u0434\u0435\u0444\u043e\u043b\u0442": "customer default event",
    "\u0422\u0438\u043f \u043c\u0430\u0433\u0430\u0437\u0438\u043d\u0430": "Store type",
    "\u0420\u0435\u0437\u043a\u0438\u0439 \u0441\u043a\u0430\u0447\u043e\u043a \u043f\u043e\u043a\u0443\u043f\u043e\u043a (\u043c\u0430\u043a\u0441\u0438\u043c\u0443\u043c \u043a \u043c\u0435\u0434\u0438\u0430\u043d\u0435)": "sharp purchase spike (maximum-to-median ratio)",
    "\u0415\u0441\u0442\u044c \u0437\u0430\u0432\u0438\u0441\u0438\u043c\u043e\u0441\u0442\u044c \u043e\u0442 \u0434\u0430\u0442\u0430\u0441\u0435\u0442\u0430. \u041a\u043e\u0440\u0440\u0435\u043b\u044f\u0446\u0438\u044f LogReg & MLP \u0441\u0438\u043b\u044c\u043d\u0435\u0435 \u043d\u0430 \u043f\u043e\u0441\u043b\u0435\u0434\u043d\u0438\u0445 trials": (
        "There is a dataset dependency. LogReg and MLP correlation is stronger on the latest trials."
    ),
    "\u041f\u043e\u0434\u0431\u0438\u0440\u0430\u044f \u0433\u0438\u043f\u0435\u0440\u043f\u0430\u0440\u0430\u043c\u0435\u0442\u0440\u044b \u0432 \u0441 \u043e\u0434\u043d\u0438\u043c \u0432\u0430\u043b\u0438\u0434\u0430\u0442\u043e\u0440\u043e\u043c \u0412\u041e\u0417\u041c\u041e\u0416\u041d\u0410 \u043f\u0440\u043e\u0441\u0430\u0434\u043a\u0430 \u0441 \u0434\u0440\u0443\u0433\u0438\u043c. \u041a\u043e\u0440\u0440\u0435\u043b\u044f\u0446\u0438\u044f \u0432\u0430\u043b\u0438\u0434\u0430\u0442\u043e\u0440\u043e\u0432 \u0437\u0430\u0432\u0438\u0441\u0438\u0442 \u043e\u0442 \u0434\u0430\u0442\u0430\u0441\u0435\u0442\u0430 \u0438 \u0437\u0430\u0434\u0430\u0447\u0438": (
        "Tuning hyperparameters with one validator can reduce performance with another validator. Validator correlation depends on dataset and task."
    ),
    "Gaussian-Copula \u0438 \u0442\u0435\u043e\u0440\u0435\u043c\u0430 \u043d\u0430\u0441\u044b\u0449\u0435\u043d\u0438\u044f \u043c\u043e\u0433\u0443\u0442 \u0431\u044b\u0442\u044c \u0438\u0441\u043f\u043e\u043b\u044c\u0437\u043e\u0432\u0430\u043d\u044b \u043a\u0430\u043a \u0434\u0438\u0430\u0433\u043d\u043e\u0441\u0442\u0438\u0447\u0435\u0441\u043a\u0438\u0439 \u0438\u043d\u0441\u0442\u0440\u0443\u043c\u0435\u043d\u0442 \u043f\u0440\u0438 \u0430\u043d\u0430\u043b\u0438\u0437\u0435 \u043e\u0448\u0438\u0431\u043e\u043a \u043c\u043e\u0434\u0435\u043b\u0435\u0439 \u0432 EBES: \u043f\u0435\u0440\u0432\u043e\u0435 \u043f\u043e\u0437\u0432\u043e\u043b\u044f\u0435\u0442 \u0440\u0430\u0437\u0434\u0435\u043b\u0438\u0442\u044c \u0441\u0438\u0441\u0442\u0435\u043c\u043d\u0443\u044e \u0441\u043b\u043e\u0436\u043d\u043e\u0441\u0442\u044c \u043e\u0431\u044a\u0435\u043a\u0442\u043e\u0432 \u043e\u0442 \u0438\u043d\u0434\u0438\u0432\u0438\u0434\u0443\u0430\u043b\u044c\u043d\u044b\u0445 \u043d\u0435\u0434\u043e\u0441\u0442\u0430\u0442\u043a\u043e\u0432 \u043c\u043e\u0434\u0435\u043b\u0438, \u0432\u0442\u043e\u0440\u043e\u0435 \u0434\u0430\u0451\u0442 \u0442\u0435\u043e\u0440\u0435\u0442\u0438\u0447\u0435\u0441\u043a\u043e\u0435 \u043e\u0431\u043e\u0441\u043d\u043e\u0432\u0430\u043d\u0438\u0435 \u043d\u0435\u043e\u0431\u0445\u043e\u0434\u0438\u043c\u043e\u0441\u0442\u0438 \u0430\u0440\u0445\u0438\u0442\u0435\u043a\u0442\u0443\u0440\u043d\u043e\u0433\u043e \u0440\u0430\u0437\u043d\u043e\u043e\u0431\u0440\u0430\u0437\u0438\u044f \u0432 \u043d\u0430\u0431\u043e\u0440\u0435 \u0441\u0440\u0430\u0432\u043d\u0438\u0432\u0430\u0435\u043c\u044b\u0445 \u043c\u0435\u0442\u043e\u0434\u043e\u0432.": (
        "Gaussian-Copula and the saturation theorem can be used as diagnostic tools for model error analysis in EBES: the former separates intrinsic object complexity from model-specific weaknesses, while the latter supports architectural diversity in the compared method set."
    ),
    "\u0412\u043e\u0437\u043c\u043e\u0436\u043d\u043e \u0432\u0441\u0442\u0440\u043e\u0438\u0442\u044c Transformer \u0432\u043c\u0435\u0441\u0442\u043e MLP \u0432 CoLES. VTL - \u0445\u043e\u0440\u043e\u0448\u0430\u044f \u0437\u0430\u0434\u0443\u043c\u043a\u0430 \u0434\u043b\u044f \u043e\u043f\u0440\u0435\u0434\u0435\u043b\u0435\u043d\u0438\u0435 \u043d\u043e\u0432\u044b\u0445 c\u043e\u0432\u043c\u0435\u0449\u0435\u043d\u043d\u044b\u0445 \u044d\u043c\u0431\u0435\u0434\u0434\u0438\u043d\u0433\u043e\u0432 \u0440\u0430\u0437\u043d\u044b\u0445 \u0440\u0430\u0437\u043c\u0435\u0440\u043d\u043e\u0441\u0442\u0435\u0439 3D \u043c\u043e\u0434\u0435\u043b\u0438 \u0441 \u0440\u0430\u0437\u043b\u0438\u0447\u043d\u044b\u043c\u0438 \u0441\u043b\u0438\u044f\u043d\u0438\u044f\u043c\u0438 \u044d\u043c\u0431\u0435\u0434\u0434\u0438\u043d\u0433\u043e\u0432 \u0434\u0430\u044e\u0442 \u043f\u0440\u0438\u0440\u043e\u0441\u0442 \u043d\u0430 \u0430\u043d\u043e\u043c\u0430\u043b\u0438\u044f\u0445": (
        "A Transformer could be used instead of an MLP in CoLES. VTL is a promising idea for combined embeddings with different dimensions; 3D models with different embedding fusions improve anomaly results."
    ),
    "\u0420\u0430\u0437\u0434\u0435\u043b\u0435\u043d\u0438\u0435 \u043d\u0430 \u0441\u0442\u0430\u0442\u0438\u0447\u0435\u0441\u043a\u0438\u0435 \u0438 \u0434\u0438\u043d\u0430\u043c\u0438\u0447\u0435\u0441\u043a\u0438\u0435 \u0444\u0438\u0447\u0438 \u0414\u0435\u0442\u0435\u043a\u0446\u0438\u044f \u0430\u043d\u043e\u043c\u0430\u043b\u0438\u0439 +111% \u043e\u0442 \u043c\u043e\u0434\u0435\u043b\u0435\u0439 \u0432 \u043f\u0440\u043e\u0434\u0435": (
        "Separation into static and dynamic features. Anomaly detection improves by 111% over production models."
    ),
    "\u0410\u043d\u0430\u043b\u0438\u0437 \u0441\u043b\u0435\u043f\u044b\u0445 \u0437\u043e\u043d \u044d\u043d\u043a\u043e\u0434\u0435\u0440\u043e\u0432 (Amount, Categories, Time, Activity) \u0415\u0441\u0442\u044c \u0430\u043d\u0430\u043b\u0438\u0437 \u043f\u043e CoLES, NTP": (
        "Analysis of encoder blind spots: Amount, Categories, Time, Activity. Includes analysis for CoLES and NTP."
    ),
}


def find_workbook() -> Path:
    candidates = [ROOT / "SBER_BENCH.xlsx", *sorted((ROOT / "data").glob("SBER_BENCH*.xlsx"))]
    existing = [path for path in candidates if path.exists()]
    if not existing:
        raise SystemExit("Workbook not found: expected SBER_BENCH*.xlsx in project root or data/")
    return max(existing, key=lambda path: (path.stat().st_mtime, path.name))


def choose_source_sheet(sheets: dict[str, str]) -> str:
    for sheet in SOURCE_SHEET_CANDIDATES:
        if sheet in sheets:
            return sheet
    expected = ", ".join(SOURCE_SHEET_CANDIDATES)
    raise SystemExit(f"Sheet not found: expected one of {expected}")


def normalize_spaces(value: str) -> str:
    value = str(value).replace("\u0421OLES", "COLES")
    return re.sub(r"\s+", " ", value).strip()


def translate_known_text(value: str) -> str:
    text = value
    for source, target in TEXT_REPLACEMENTS.items():
        text = text.replace(source, target)
    return text


def slugify(value: str) -> str:
    cleaned = normalize_spaces(value).lower()
    cleaned = cleaned.replace("\u0451", "e")
    cleaned = re.sub(r"[^a-z0-9]+", "-", cleaned)
    return cleaned.strip("-") or "unknown"


def parse_number(value: str) -> float | None:
    value = normalize_spaces(str(value))
    if value in {"", "-", "ND", "N/A", "nan"}:
        return None
    try:
        return float(value.replace(",", "."))
    except ValueError:
        return None


def clean_detail_text(value: str) -> str:
    text = normalize_spaces(value)
    if not text:
        return ""
    text = translate_known_text(text)
    text = re.sub(r"\baggregation\.namee?\b", "aggregation", text)
    text = re.sub(r"\bparams\.", "", text)
    text = re.sub(r"\bName:\s*\d+,\s*dtype:\s*object\b", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" ,")


def infer_validator(value: str) -> str:
    text = re.sub(r"[_()/.-]+", " ", normalize_spaces(value).lower())
    if re.search(r"\b(lightgbm|lgbm|lgmb)\b", text):
        return "LGBM"
    if re.search(r"\b(logreg|logistic)\b", text):
        return "LOGREG"
    if re.search(r"\bmlp\b", text):
        return "MLP"
    return ""


def load_shared_strings(archive: ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    strings: list[str] = []
    for item in root.findall("m:si", NS):
        strings.append("".join(text.text or "" for text in item.iter(f"{{{NS['m']}}}t")))
    return strings


def workbook_sheets(archive: ZipFile) -> dict[str, str]:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    relmap = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}
    sheets: dict[str, str] = {}
    for sheet in workbook.findall("m:sheets/m:sheet", NS):
        rid = sheet.attrib[f"{{{NS['r']}}}id"]
        target = relmap[rid]
        sheets[sheet.attrib["name"]] = target if target.startswith("xl/") else f"xl/{target}"
    return sheets


def column_index(column: str) -> int:
    index = 0
    for char in column:
        index = index * 26 + ord(char) - 64
    return index - 1


def cell_value(cell: ET.Element, shared: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    value_node = cell.find("m:v", NS)
    if cell_type == "s":
        if value_node is None or value_node.text is None:
            return ""
        return shared[int(value_node.text)]
    if cell_type == "inlineStr":
        return "".join(text.text or "" for text in cell.iter(f"{{{NS['m']}}}t"))
    return value_node.text if value_node is not None and value_node.text is not None else ""


def read_rows(archive: ZipFile, sheet_path: str, shared: list[str]) -> list[list[str]]:
    root = ET.fromstring(archive.read(sheet_path))
    rows: list[list[str]] = []
    for row in root.findall("m:sheetData/m:row", NS):
        values: dict[int, str] = {}
        for cell in row.findall("m:c", NS):
            ref = cell.attrib.get("r", "A1")
            match = re.match(r"([A-Z]+)", ref)
            if not match:
                continue
            values[column_index(match.group(1))] = normalize_spaces(cell_value(cell, shared))
        if values:
            row_values = [values.get(index, "") for index in range(max(values) + 1)]
            while row_values and row_values[-1] == "":
                row_values.pop()
            if any(str(value) != "" for value in row_values):
                rows.append(row_values)
    return rows


def cell(row: list[str], index: int) -> str:
    return row[index] if index < len(row) else ""


def extract_records(rows: list[list[str]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    current_dataset = ""
    for row in rows[2:]:
        if cell(row, 0):
            current_dataset = normalize_spaces(cell(row, 0))
        method = normalize_spaces(cell(row, 1))
        if not current_dataset or not method:
            continue

        scores = {task["id"]: parse_number(cell(row, task["scoreColumn"])) for task in TASKS}
        if all(value is None for value in scores.values()):
            continue

        params = {
            task["id"]: clean_detail_text(cell(row, task["paramColumn"]))
            for task in TASKS
            if clean_detail_text(cell(row, task["paramColumn"]))
        }
        masking = {
            task["id"]: clean_detail_text(cell(row, task["maskColumn"]))
            for task in TASKS
            if clean_detail_text(cell(row, task["maskColumn"]))
        }
        records.append(
            {
                "dataset": current_dataset,
                "datasetId": slugify(current_dataset),
                "method": method,
                "methodId": slugify(method),
                "validator": infer_validator(method),
                "scores": scores,
                "params": params,
                "masking": masking,
                "comment": clean_detail_text(cell(row, 14)),
            }
        )
    return records


def task_meta(task_id: str) -> dict[str, str]:
    aliases = {
        "reg": "regression",
        "regr": "regression",
        "regression": "regression",
        "clf": "classification",
        "classification": "classification",
        "forecast": "forecasting",
        "forecasting": "forecasting",
        "anomaly": "anomaly",
    }
    normalized = aliases.get(normalize_spaces(task_id).lower(), normalize_spaces(task_id).lower())
    task = next((item for item in TASKS if item["id"] == normalized), None)
    return {
        "id": normalized,
        "label": task["label"] if task else normalize_spaces(task_id),
        "metric": task["metric"] if task else "value",
    }


def extract_ntp_lgbm(rows: list[list[str]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    dataset = ""
    for row in rows[1:]:
        if cell(row, 1):
            dataset = normalize_spaces(cell(row, 1))
        method = normalize_spaces(cell(row, 2))
        raw_task = normalize_spaces(cell(row, 3))
        classifier = normalize_spaces(cell(row, 4))
        value = parse_number(cell(row, 5))
        if not dataset or not method or not raw_task or not classifier or value is None:
            continue
        task = task_meta(raw_task)
        records.append(
            {
                "dataset": dataset,
                "datasetId": slugify(dataset),
                "method": method,
                "methodId": slugify(method),
                "task": task["id"],
                "taskLabel": task["label"],
                "metric": task["metric"],
                "classifier": classifier,
                "value": value,
                "sourceSheet": NTP_SHEET,
            }
        )
    return records


def wide_record(
    source_sheet: str,
    dataset: str,
    method: str,
    scores: dict[str, float | None],
    *,
    description: str = "",
    validator: str = "",
    extra_metrics: dict[str, float | None] | None = None,
) -> dict[str, Any] | None:
    if not dataset or not method or all(value is None for value in scores.values()):
        return None
    return {
        "sourceSheet": source_sheet,
        "dataset": clean_detail_text(dataset),
        "datasetId": slugify(dataset),
        "method": clean_detail_text(method),
        "methodId": slugify(method),
        "description": clean_detail_text(description),
        "validator": clean_detail_text(validator),
        "scores": scores,
        "extraMetrics": extra_metrics or {},
    }


def extract_wide_results(sheet_name: str, rows: list[list[str]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    dataset = ""

    def add(record: dict[str, Any] | None) -> None:
        if record:
            records.append(record)

    if sheet_name == "BERT_LBGM":
        for row in rows[1:]:
            dataset = normalize_spaces(cell(row, 0)) or dataset
            add(
                wide_record(
                    sheet_name,
                    dataset,
                    cell(row, 1),
                    {
                        "regression": parse_number(cell(row, 2)),
                        "classification": parse_number(cell(row, 3)),
                        "forecasting": parse_number(cell(row, 4)),
                        "anomaly": parse_number(cell(row, 5)),
                    },
                )
            )
    elif sheet_name == "Coles 1 run":
        for row in rows[2:]:
            add(
                wide_record(
                    sheet_name,
                    cell(row, 0),
                    "COLES 1 run",
                    {
                        "regression": parse_number(cell(row, 1)),
                        "classification": parse_number(cell(row, 2)),
                        "forecasting": parse_number(cell(row, 3)),
                        "anomaly": parse_number(cell(row, 4)),
                    },
                    description=cell(row, 5),
                )
            )
    elif sheet_name in {"aggregated", "Copy of aggregated"}:
        for row in rows[2:]:
            dataset = normalize_spaces(cell(row, 0)) or dataset
            add(
                wide_record(
                    sheet_name,
                    dataset,
                    cell(row, 1),
                    {
                        "regression": parse_number(cell(row, 2)),
                        "classification": parse_number(cell(row, 3)),
                        "forecasting": parse_number(cell(row, 4)),
                        "anomaly": parse_number(cell(row, 5)),
                    },
                    description=cell(row, 10),
                )
            )
    elif sheet_name == "TS-FRSH":
        for row in rows[2:]:
            dataset = re.sub(r"\s+NEW$", "", normalize_spaces(cell(row, 0)), flags=re.IGNORECASE)
            method = " ".join(part for part in ["TS-FRESH", cell(row, 2), cell(row, 3)] if part)
            add(
                wide_record(
                    sheet_name,
                    dataset,
                    method,
                    {
                        "regression": parse_number(cell(row, 4)),
                        "classification": parse_number(cell(row, 5)),
                        "forecasting": parse_number(cell(row, 6)),
                        "anomaly": parse_number(cell(row, 7)),
                    },
                    description=cell(row, 1),
                )
            )
    elif sheet_name == "TS-FRSH VS":
        for row in rows[2:]:
            dataset = normalize_spaces(cell(row, 2)) or dataset
            method = " ".join(part for part in [cell(row, 3), cell(row, 4), cell(row, 5)] if part)
            add(
                wide_record(
                    sheet_name,
                    dataset,
                    method,
                    {
                        "regression": parse_number(cell(row, 6)),
                        "classification": parse_number(cell(row, 7)),
                        "forecasting": parse_number(cell(row, 9)),
                        "anomaly": parse_number(cell(row, 10)),
                    },
                    extra_metrics={
                        "classification_f1_macro": parse_number(cell(row, 8)),
                        "anomaly_f1_macro": parse_number(cell(row, 11)),
                    },
                )
            )
    elif sheet_name == "ALL old":
        model = ""
        description = ""
        for row in rows[2:]:
            dataset = normalize_spaces(cell(row, 0)) or dataset
            model = normalize_spaces(cell(row, 1)) or model
            description = normalize_spaces(cell(row, 2)) or description
            validator = normalize_spaces(cell(row, 3))
            method = " ".join(part for part in [model, description, validator] if part)
            add(
                wide_record(
                    sheet_name,
                    dataset,
                    method,
                    {
                        "regression": parse_number(cell(row, 5)),
                        "classification": parse_number(cell(row, 6)),
                        "forecasting": parse_number(cell(row, 9)),
                        "anomaly": parse_number(cell(row, 10)),
                    },
                    validator=validator,
                    extra_metrics={
                        "regression_nmse": parse_number(cell(row, 4)),
                        "classification_f1_macro": parse_number(cell(row, 7)),
                        "forecasting_nmse": parse_number(cell(row, 8)),
                        "anomaly_f1_macro": parse_number(cell(row, 11)),
                        "anomaly_accuracy": parse_number(cell(row, 12)),
                    },
                )
            )
    elif sheet_name in {"AGE MAIN", "FAVORITA", "TAO BAO MAIN"}:
        current_model = ""
        current_description = ""
        dataset = "AGE" if sheet_name == "AGE MAIN" else "FAVORITA" if sheet_name == "FAVORITA" else "TAOBAO"
        for row in rows[2:]:
            if cell(row, 0) and sheet_name != "FAVORITA":
                dataset = normalize_spaces(cell(row, 0))
            current_model = normalize_spaces(cell(row, 1)) or current_model or normalize_spaces(cell(row, 0))
            current_description = normalize_spaces(cell(row, 2)) or current_description
            validator = normalize_spaces(cell(row, 4) if sheet_name != "FAVORITA" else cell(row, 2))
            if sheet_name == "FAVORITA":
                method = " ".join(part for part in [current_model, current_description, validator] if part)
                scores = {
                    "regression": parse_number(cell(row, 5)),
                    "classification": parse_number(cell(row, 6)),
                    "forecasting": parse_number(cell(row, 8)),
                    "anomaly": parse_number(cell(row, 9)),
                }
                extra = {
                    "mean_all": parse_number(cell(row, 3)),
                    "classification_f1_macro": parse_number(cell(row, 7)),
                    "anomaly_f1_macro": parse_number(cell(row, 10)),
                }
            else:
                method = " ".join(part for part in [current_model, current_description, validator] if part)
                scores = {
                    "regression": parse_number(cell(row, 5)),
                    "classification": parse_number(cell(row, 6)),
                    "forecasting": parse_number(cell(row, 8)),
                    "anomaly": parse_number(cell(row, 9)),
                }
                extra = {
                    "mean_all": parse_number(cell(row, 3)),
                    "classification_f1_macro": parse_number(cell(row, 7)),
                    "anomaly_f1_macro": parse_number(cell(row, 10)),
                }
            add(wide_record(sheet_name, dataset, method, scores, validator=validator, extra_metrics=extra))
    return records


def extract_bert_corr(rows: list[list[str]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    dataset = ""
    task = ""
    metric = ""
    for row in rows[1:]:
        dataset = normalize_spaces(cell(row, 0)) or dataset
        task = normalize_spaces(cell(row, 1)) or task
        metric = normalize_spaces(cell(row, 2)) or metric
        validator = normalize_spaces(cell(row, 3))
        trials = [parse_number(cell(row, index)) for index in range(4, 14)]
        if not dataset or not task or not validator or all(value is None for value in trials):
            continue
        records.append(
            {
                "dataset": dataset,
                "datasetId": slugify(dataset),
                "task": task_meta(task)["id"],
                "taskLabel": task_meta(task)["label"],
                "metric": metric or task_meta(task)["metric"],
                "validator": validator,
                "trials": trials,
                "total": parse_number(cell(row, 14)),
                "winner": normalize_spaces(cell(row, 15)),
                "sourceSheet": "BERT_CORR",
            }
        )
    return records


def extract_progress(rows: list[list[str]]) -> list[dict[str, Any]]:
    if len(rows) < 2:
        return []
    headers = [normalize_spaces(value) for value in rows[1]]
    progress: list[dict[str, Any]] = []
    for row in rows[2:]:
        dataset = clean_detail_text(cell(row, 2))
        if not dataset:
            continue
        methods = {}
        for index, header in enumerate(headers[3:12], start=3):
            if header and cell(row, index):
                methods[clean_detail_text(header)] = clean_detail_text(cell(row, index))
        progress.append(
            {
                "order": parse_number(cell(row, 0)),
                "dataset": dataset,
                "datasetId": slugify(dataset),
                "methods": methods,
                "status": clean_detail_text(cell(row, 13)),
            }
        )
    return progress


def extract_papers(rows: list[list[str]]) -> list[dict[str, Any]]:
    return [
        {
            "name": clean_detail_text(cell(row, 0)),
            "source": clean_detail_text(cell(row, 1)),
            "year": parse_number(cell(row, 2)),
            "relevance": clean_detail_text(cell(row, 3)),
        }
        for row in rows[1:]
        if cell(row, 0)
    ]


def extract_notes(rows: list[list[str]]) -> list[dict[str, str]]:
    notes: list[dict[str, str]] = []
    for row in rows:
        cells = [cell(row, index) for index in range(len(row))]
        if any(cells):
            notes.append({f"col{index + 1}": clean_detail_text(value) for index, value in enumerate(cells) if value})
    return notes


def raw_sheet(name: str, rows: list[list[str]]) -> dict[str, Any]:
    display_rows = [[clean_detail_text(value) for value in row] for row in rows]
    display_name = clean_detail_text(name) or name
    return {
        "name": display_name,
        "id": slugify(display_name),
        "rows": display_rows,
        "rowCount": len(display_rows),
        "columnCount": max((len(row) for row in display_rows), default=0),
    }


def extract_optuna_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        records: list[dict[str, Any]] = []
        for row in reader:
            record: dict[str, Any] = {}
            for key, value in row.items():
                if key is None or value in {None, ""}:
                    continue
                parsed = parse_number(value)
                record[clean_detail_text(key)] = parsed if parsed is not None else clean_detail_text(value)
            if record:
                records.append(record)
        return records


def extract_heatmap_csv(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"datasets": [], "records": []}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        dataset_columns = [field for field in (reader.fieldnames or []) if field not in {"Task", "Model"}]
        records = []
        for row in reader:
            values = {
                dataset: parse_number(row.get(dataset, ""))
                for dataset in dataset_columns
            }
            records.append(
                {
                    "task": clean_detail_text(row.get("Task", "")),
                    "model": clean_detail_text(row.get("Model", "")),
                    "values": values,
                }
            )
        return {"datasets": dataset_columns, "records": records}


def extract_method_comparison(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"title": "", "subtitle": "", "sections": [], "takeaways": []}
    lines = path.read_text(encoding="utf-8").splitlines()
    title = ""
    subtitle = ""
    sections: list[dict[str, Any]] = []
    takeaways: list[str] = []
    current: dict[str, Any] | None = None

    for raw_line in lines:
        line = clean_detail_text(raw_line)
        if not line:
            continue
        if line.startswith("## "):
            title = line[3:]
        elif line.startswith("Datasets:"):
            subtitle = line
        elif line.startswith("### "):
            name = line[4:]
            current = None if name == "Key Takeaways" else {"name": name, "entries": []}
            if current:
                sections.append(current)
        elif line.startswith("- ") and current:
            match = re.match(r"-\s*(.+?):\s*([0-9.]+)%", line)
            if match:
                current["entries"].append({"label": clean_detail_text(match.group(1)), "value": float(match.group(2))})
        elif re.match(r"\d+\.\s+", line):
            takeaways.append(re.sub(r"^\d+\.\s+", "", line))

    return {"title": title, "subtitle": subtitle, "sections": sections, "takeaways": takeaways}


def extract_feature_plans(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"title": "", "items": []}
    lines = path.read_text(encoding="utf-8").splitlines()
    title = "Future Plans"
    items: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for raw_line in lines:
        line = clean_detail_text(raw_line)
        if not line:
            continue
        heading = re.match(r"##\s*(?:Future Plans|\u0414\u0430\u043b\u044c\u043d\u0435\u0439\u0448\u0438\u0435 \u043f\u043b\u0430\u043d\u044b \(Future Plans\))", line)
        if heading:
            title = "Future Plans"
            continue
        top = re.match(r"\d+\.\s*\*\*(.+?)\*\*:?", line)
        if top:
            current = {"title": clean_detail_text(top.group(1)), "notes": []}
            items.append(current)
            continue
        bullet = re.match(r"-\s*(?:[a-z]\.\s*)?(?:[ivx]+\.\s*)?(.+)", line)
        if bullet and current:
            current["notes"].append(clean_detail_text(bullet.group(1)))
    return {"title": title, "items": items}


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    dataset_counts = Counter(record["dataset"] for record in records)
    method_counts = Counter(record["method"] for record in records)
    datasets = [
        {
            "id": slugify(dataset),
            "name": dataset,
            "records": count,
            "methods": len({record["method"] for record in records if record["dataset"] == dataset}),
        }
        for dataset, count in sorted(dataset_counts.items())
    ]
    methods = [
        {"id": slugify(method), "name": method, "records": count}
        for method, count in method_counts.most_common()
    ]
    return {"datasets": datasets, "methods": methods}


def main() -> None:
    workbook = find_workbook()
    with ZipFile(workbook) as archive:
        shared = load_shared_strings(archive)
        sheets = workbook_sheets(archive)
        source_sheet = choose_source_sheet(sheets)
        all_rows = {name: read_rows(archive, path, shared) for name, path in sheets.items()}
        rows = all_rows[source_sheet]

    records = extract_records(rows)
    summary = aggregate(records)
    raw_sheets = [
        raw_sheet(name, sheet_rows)
        for name, sheet_rows in all_rows.items()
        if sheet_rows and name not in SKIP_RAW_SHEETS
    ]
    ntp_records = extract_ntp_lgbm(all_rows.get(NTP_SHEET, []))
    wide_results = [
        record
        for sheet_name in [
            "BERT_LBGM",
            "Coles 1 run",
            "aggregated",
            "Copy of aggregated",
            "ALL old",
            "AGE MAIN",
            "FAVORITA",
            "TAO BAO MAIN",
            "TS-FRSH",
            "TS-FRSH VS",
        ]
        for record in extract_wide_results(sheet_name, all_rows.get(sheet_name, []))
    ]
    validator_trials = extract_bert_corr(all_rows.get("BERT_CORR", []))
    progress = extract_progress(all_rows.get("PROGRESS", []))
    papers = extract_papers(all_rows.get("papers", []))
    notes = {
        "bertExp": extract_notes(all_rows.get("BertExp", [])),
        "bertSpearman": extract_notes(all_rows.get("BERT_SPEARMAN", [])),
        "datasetRegistry": extract_notes(all_rows.get("data", [])),
    }
    optuna = {
        "summary": extract_optuna_csv(ROOT / "data" / "summury-optuna.csv"),
        "top10Trials": extract_optuna_csv(ROOT / "data" / "top10-optuna-trials.csv"),
        "last10Trials": extract_optuna_csv(ROOT / "data" / "last10-optuna-trials.csv"),
    }
    insights = {
        "dataHeatmap": extract_heatmap_csv(ROOT / "data" / "data_heatmap.csv"),
        "methodComparison": extract_method_comparison(ROOT / "data" / "comparison-of-methods.md"),
        "featurePlans": extract_feature_plans(ROOT / "data" / "feature-plans.md"),
    }
    optuna_records = sum(len(rows) for rows in optuna.values())
    modified = datetime.fromtimestamp(workbook.stat().st_mtime).date().isoformat()
    payload = {
        "meta": {
            "sourceFile": str(workbook.relative_to(ROOT)),
            "sourceSheet": source_sheet,
            "lastVerified": modified,
            "records": len(records),
            "datasets": len(summary["datasets"]),
            "methods": len(summary["methods"]),
            "workbookSheets": len(raw_sheets),
            "rawRows": sum(sheet["rowCount"] for sheet in raw_sheets),
            "additionalRecords": len(ntp_records) + len(wide_results) + len(validator_trials) + optuna_records,
        },
        "tasks": [{"id": task["id"], "label": task["label"], "metric": task["metric"]} for task in TASKS],
        "datasets": summary["datasets"],
        "methods": summary["methods"],
        "records": records,
        "additional": {
            "ntpLgbm": {"records": ntp_records},
            "wideResults": {"records": wide_results},
            "validatorTrials": {"records": validator_trials},
            "progress": {"records": progress},
            "papers": {"records": papers},
            "notes": notes,
            "optuna": optuna,
            "insights": insights,
        },
        "rawSheets": raw_sheets,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT.relative_to(ROOT)}: {len(records)} records")


if __name__ == "__main__":
    main()
