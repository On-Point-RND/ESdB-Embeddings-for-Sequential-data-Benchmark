#!/usr/bin/env python3
"""Extract public benchmark rows from SBER_BENCH.xlsx.

The script intentionally uses only Python's standard library. An .xlsx file is
an OpenXML zip archive, so this keeps the project deployable without pandas or
openpyxl.
"""

from __future__ import annotations

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
SOURCE_SHEET = "RESULTS NEW from March"
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


def find_workbook() -> Path:
    candidates = [ROOT / "SBER_BENCH.xlsx", *sorted((ROOT / "data").glob("SBER_BENCH*.xlsx"))]
    existing = [path for path in candidates if path.exists()]
    if not existing:
        raise SystemExit("Workbook not found: expected SBER_BENCH*.xlsx in project root or data/")

    def version(path: Path) -> int:
        match = re.search(r"SBER_BENCH-(\d+)\.xlsx$", path.name)
        return int(match.group(1)) if match else 0

    return max(existing, key=lambda path: (version(path), path.stat().st_mtime, path.name))


def normalize_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def slugify(value: str) -> str:
    cleaned = normalize_spaces(value).lower()
    cleaned = cleaned.replace("ё", "e")
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
    text = re.sub(r"\baggregation\.namee?\b", "aggregation", text)
    text = re.sub(r"\bparams\.", "", text)
    text = re.sub(r"\bName:\s*\d+,\s*dtype:\s*object\b", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" ,")


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
        "dataset": normalize_spaces(dataset),
        "datasetId": slugify(dataset),
        "method": normalize_spaces(method),
        "methodId": slugify(method),
        "description": normalize_spaces(description),
        "validator": normalize_spaces(validator),
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
        dataset = normalize_spaces(cell(row, 2))
        if not dataset:
            continue
        methods = {}
        for index, header in enumerate(headers[3:12], start=3):
            if header and cell(row, index):
                methods[header] = normalize_spaces(cell(row, index))
        progress.append(
            {
                "order": parse_number(cell(row, 0)),
                "dataset": dataset,
                "datasetId": slugify(dataset),
                "methods": methods,
                "status": normalize_spaces(cell(row, 13)),
            }
        )
    return progress


def extract_papers(rows: list[list[str]]) -> list[dict[str, Any]]:
    return [
        {
            "name": cell(row, 0),
            "source": cell(row, 1),
            "year": parse_number(cell(row, 2)),
            "relevance": cell(row, 3),
        }
        for row in rows[1:]
        if cell(row, 0)
    ]


def extract_notes(rows: list[list[str]]) -> list[dict[str, str]]:
    notes: list[dict[str, str]] = []
    for row in rows:
        cells = [cell(row, index) for index in range(len(row))]
        if any(cells):
            notes.append({f"col{index + 1}": value for index, value in enumerate(cells) if value})
    return notes


def raw_sheet(name: str, rows: list[list[str]]) -> dict[str, Any]:
    display_rows = [[clean_detail_text(value) for value in row] for row in rows]
    return {
        "name": name,
        "id": slugify(name),
        "rows": display_rows,
        "rowCount": len(display_rows),
        "columnCount": max((len(row) for row in display_rows), default=0),
    }


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
        if SOURCE_SHEET not in sheets:
            raise SystemExit(f"Sheet not found: {SOURCE_SHEET}")
        all_rows = {name: read_rows(archive, path, shared) for name, path in sheets.items()}
        rows = all_rows[SOURCE_SHEET]

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
    modified = datetime.fromtimestamp(workbook.stat().st_mtime).date().isoformat()
    payload = {
        "meta": {
            "sourceFile": str(workbook.relative_to(ROOT)),
            "sourceSheet": SOURCE_SHEET,
            "lastVerified": modified,
            "records": len(records),
            "datasets": len(summary["datasets"]),
            "methods": len(summary["methods"]),
            "workbookSheets": len(raw_sheets),
            "rawRows": sum(sheet["rowCount"] for sheet in raw_sheets),
            "additionalRecords": len(ntp_records) + len(wide_results) + len(validator_trials),
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
        },
        "rawSheets": raw_sheets,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT.relative_to(ROOT)}: {len(records)} records")


if __name__ == "__main__":
    main()
