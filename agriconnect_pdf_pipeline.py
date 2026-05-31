"""
AgriConnect — PDF to Fine-tuning Dataset Pipeline
==================================================
Handles mixed PDFs (paragraphs + tables + soil data) from
Odisha government agricultural sources.

Usage:
    python agriconnect_pdf_pipeline.py pdf --pdf your_file.pdf --out dataset.jsonl
    python agriconnect_pdf_pipeline.py merge --csv crop.csv --jsonl dataset.jsonl --out merged.jsonl
"""

import re
import json
import argparse
import unicodedata
from pathlib import Path
from typing import Optional

import pdfplumber
import fitz
import pandas as pd


# ─────────────────────────────────────────────
# STEP 1 — DETECT PDF TYPE
# ─────────────────────────────────────────────

def detect_pdf_type(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    text_pages, empty_pages = 0, 0
    for page in doc:
        text = page.get_text().strip()
        if len(text) > 50:
            text_pages += 1
        else:
            empty_pages += 1
    doc.close()
    if text_pages == 0:
        return "scanned"
    elif empty_pages > text_pages * 0.3:
        return "mixed"
    return "text"


# ─────────────────────────────────────────────
# SAFE CELL HELPERS  (fix for nested Series)
# ─────────────────────────────────────────────

EMPTY_TOKENS = {"", "-", "na", "n/a", "none", "nan", "null", "#", "--"}

def cell_to_str(val) -> str:
    """
    Flatten ANY cell value (scalar, list, tuple, pd.Series, None) to a
    plain stripped string.  Never raises, never calls bool() on a Series.
    """
    try:
        if isinstance(val, pd.Series):
            parts = [str(v).strip() for v in val.dropna() if str(v).strip()]
            return " ".join(parts)
        if isinstance(val, (list, tuple)):
            parts = [str(v).strip() for v in val if str(v).strip()]
            return " ".join(parts)
        if val is None:
            return ""
        return str(val).strip()
    except Exception:
        return ""


def is_valid_cell(val) -> bool:
    s = cell_to_str(val).lower()
    return len(s) > 0 and s not in EMPTY_TOKENS


# ─────────────────────────────────────────────
# STEP 2 — EXTRACT TEXT + TABLES PER PAGE
# ─────────────────────────────────────────────

def extract_pages(pdf_path: str) -> list:
    pages_data = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            raw_text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
            raw_tables = page.extract_tables({
                "vertical_strategy":   "lines_strict",
                "horizontal_strategy": "lines_strict",
                "snap_tolerance": 4,
                "join_tolerance":  3,
            })
            tables = []
            for t in (raw_tables or []):
                if t and len(t) > 1:
                    try:
                        df = pd.DataFrame(t[1:], columns=t[0])
                        df = df.dropna(how="all").reset_index(drop=True)
                        tables.append(df)
                    except Exception:
                        pass
            pages_data.append({
                "page_num": i + 1,
                "raw_text": raw_text,
                "tables":   tables,
            })
    return pages_data


# ─────────────────────────────────────────────
# STEP 3 — CLEAN RAW TEXT
# ─────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"Page\s+\d+\s+of\s+\d+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"[-_]{3,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


# ─────────────────────────────────────────────
# STEP 4A — PARSE SOIL VALUES FROM TEXT
# ─────────────────────────────────────────────

SOIL_PATTERNS = {
    "nitrogen_kg_ha":   r"(?:nitrogen|N)[^\d]{0,20}?(\d+(?:\.\d+)?)\s*(?:kg/ha|kg ha)",
    "phosphorus_kg_ha": r"(?:phosphorus|phosphate|P)[^\d]{0,20}?(\d+(?:\.\d+)?)\s*(?:kg/ha|kg ha)",
    "potassium_kg_ha":  r"(?:potassium|potash|K)[^\d]{0,20}?(\d+(?:\.\d+)?)\s*(?:kg/ha|kg ha)",
    "ph":               r"\bpH\b[^\d]{0,10}?(\d+(?:\.\d+)?)",
    "organic_carbon":   r"(?:organic carbon|OC)[^\d]{0,10}?(\d+(?:\.\d+)?)\s*%?",
    "rainfall_mm":      r"(?:rainfall|precipitation)[^\d]{0,20}?(\d+(?:\.\d+)?)\s*mm",
    "temperature_c":    r"(?:temperature|temp)[^\d]{0,10}?(\d+(?:\.\d+)?)\s*°?C",
}

CROP_LIST = [
    "rice", "paddy", "wheat", "maize", "groundnut", "soybean", "mustard",
    "sunflower", "sugarcane", "cotton", "jute", "potato", "tomato",
    "brinjal", "okra", "onion", "garlic", "moong", "arhar", "pigeonpea",
    "chickpea", "lentil", "finger millet", "ragi", "jowar",
]

FERTILIZER_LIST = [
    "urea", "dap", "mop", "npk", "ssp", "mos", "gypsum",
    "zinc sulphate", "boron", "vermicompost", "fym", "neem cake",
    "ammonium sulphate", "calcium ammonium nitrate", "can",
]

SEASON_LIST   = ["kharif", "rabi", "zaid", "summer", "winter"]

DISTRICT_LIST = [
    "cuttack", "puri", "bhubaneswar", "khordha", "ganjam", "koraput",
    "balasore", "bhadrak", "kendrapara", "jagatsinghpur", "jajpur",
    "mayurbhanj", "sundargarh", "keonjhar", "sambalpur", "bargarh",
    "bolangir", "kalahandi", "nuapada", "nabarangpur", "rayagada",
    "malkangiri", "kandhamal", "nayagarh", "angul", "dhenkanal",
    "deogarh", "jharsuguda", "subarnapur",
]

def extract_entities(text: str) -> dict:
    text_lower = text.lower()
    entities = {}
    for key, pattern in SOIL_PATTERNS.items():
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            try:
                entities[key] = float(m.group(1))
            except ValueError:
                pass
    found_crops = [c for c in CROP_LIST if c in text_lower]
    if found_crops:
        entities["crops"] = found_crops
    found_ferts = [f for f in FERTILIZER_LIST if f in text_lower]
    if found_ferts:
        entities["fertilizers"] = found_ferts
    for s in SEASON_LIST:
        if s in text_lower:
            entities["season"] = s
            break
    for d in DISTRICT_LIST:
        if d in text_lower:
            entities["district"] = d
            break
    return entities


# ─────────────────────────────────────────────
# STEP 4B — PARSE TABLE ROWS INTO RECORDS
# ─────────────────────────────────────────────

COLUMN_MAP = {
    "n": "N", "nitrogen": "N", "n (kg/ha)": "N",
    "p": "P", "phosphorus": "P", "p2o5": "P", "p (kg/ha)": "P",
    "k": "K", "potassium": "K", "k2o": "K", "k (kg/ha)": "K",
    "ph": "pH",
    "oc": "OC", "organic carbon": "OC", "o.c.": "OC",
    "rainfall": "rainfall_mm", "rainfall (mm)": "rainfall_mm",
    "temperature": "temperature_c", "temp": "temperature_c",
    "crop": "crop", "crop name": "crop", "recommended crop": "crop",
    "fertilizer": "fertilizer", "recommended fertilizer": "fertilizer",
    "fertilizer name": "fertilizer",
    "dose": "dose_kg_ha", "dose (kg/ha)": "dose_kg_ha", "dosage": "dose_kg_ha",
    "district": "district", "season": "season",
}

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    new_cols = {}
    for col in df.columns:
        if col is None:
            continue
        key = str(col).strip().lower()
        new_cols[col] = COLUMN_MAP.get(key, col)
    return df.rename(columns=new_cols)


def table_to_records(df: pd.DataFrame) -> list:
    df = normalize_columns(df)
    records = []
    for _, row in df.iterrows():
        rec = {}
        for col in df.columns:
            val = row[col]           # may be scalar OR a nested pd.Series
            if is_valid_cell(val):
                rec[col] = cell_to_str(val)
        if rec:
            records.append(rec)
    return records


# ─────────────────────────────────────────────
# STEP 5 — BUILD INSTRUCTION-RESPONSE PAIRS
# ─────────────────────────────────────────────

def record_to_instruction_pair(record: dict, source: str = "table") -> Optional[dict]:
    soil_keys = {"N", "P", "K", "pH", "OC", "rainfall_mm", "temperature_c"}
    has_soil = bool(soil_keys & set(record.keys()))
    has_crop = "crop" in record
    has_fert = "fertilizer" in record

    if not (has_soil or has_crop):
        return None

    soil_parts = []
    if "N"  in record: soil_parts.append(f"N={record['N']} kg/ha")
    if "P"  in record: soil_parts.append(f"P={record['P']} kg/ha")
    if "K"  in record: soil_parts.append(f"K={record['K']} kg/ha")
    if "pH" in record: soil_parts.append(f"pH={record['pH']}")
    if "OC" in record: soil_parts.append(f"OC={record['OC']}%")
    if "rainfall_mm"   in record: soil_parts.append(f"rainfall={record['rainfall_mm']}mm")
    if "temperature_c" in record: soil_parts.append(f"temp={record['temperature_c']}°C")

    district = record.get("district", "Odisha")
    season   = record.get("season",   "Kharif")

    if soil_parts:
        prompt = (
            f"Soil test report for {district}, {season} season. "
            f"Readings: {', '.join(soil_parts)}. "
            f"Recommend the best crop and fertilizer with dosage."
        )
    else:
        prompt = (
            f"For {district} district during {season} season, "
            f"what fertilizer should be applied for {record.get('crop', 'the crop')}?"
        )

    response_parts = []
    if has_crop:
        response_parts.append(f"Recommended crop: {record['crop'].title()}.")
    if has_fert:
        dose = record.get("dose_kg_ha", "")
        dose_str = f" at {dose} kg/ha" if dose else ""
        response_parts.append(f"Apply {record['fertilizer'].upper()}{dose_str}.")

    if not response_parts and has_soil:
        response_parts.append(
            "Based on the soil readings, consult local KVK for crop and fertilizer selection."
        )

    if not response_parts:
        return None

    return {
        "instruction": prompt.strip(),
        "response":    " ".join(response_parts).strip(),
        "source":      source,
        "metadata":    {k: v for k, v in record.items()
                        if k not in ("crop", "fertilizer", "dose_kg_ha")},
    }


def text_chunk_to_pair(text_chunk: str, page_num: int) -> Optional[dict]:
    entities = extract_entities(text_chunk)
    if not entities:
        return None
    if "crops" not in entities and "fertilizers" not in entities:
        return None

    soil_parts = []
    label_map = {
        "nitrogen_kg_ha": "N", "phosphorus_kg_ha": "P",
        "potassium_kg_ha": "K", "ph": "pH",
        "organic_carbon": "OC", "rainfall_mm": "rainfall",
        "temperature_c": "temp",
    }
    for key, label in label_map.items():
        if key in entities:
            soil_parts.append(f"{label}={entities[key]}")

    district = entities.get("district", "Odisha").title()
    season   = entities.get("season",   "Kharif").title()

    if soil_parts:
        prompt = (
            f"Soil test for {district}, {season} season. "
            f"Values: {', '.join(soil_parts)}. "
            f"Recommend crop and fertilizer."
        )
    else:
        crops_str = ", ".join(entities.get("crops", ["the crop"]))
        prompt = (
            f"For {district} district during {season}, "
            f"what fertilizer schedule is recommended for {crops_str}?"
        )

    crops_str = ", ".join(c.title() for c in entities.get("crops", []))
    ferts_str = ", ".join(f.upper() for f in entities.get("fertilizers", []))

    response_parts = []
    if crops_str:
        response_parts.append(f"Recommended crop(s): {crops_str}.")
    if ferts_str:
        response_parts.append(f"Fertilizers to apply: {ferts_str}.")

    clean = clean_text(text_chunk)
    if len(clean) > 30:
        response_parts.append(f"Details: {clean[:500]}")

    return {
        "instruction": prompt.strip(),
        "response":    " ".join(response_parts).strip(),
        "source":      f"text_page_{page_num}",
        "metadata":    entities,
    }


# ─────────────────────────────────────────────
# STEP 6 — DEDUPLICATE
# ─────────────────────────────────────────────

def deduplicate(pairs: list) -> list:
    seen = set()
    unique = []
    for p in pairs:
        key = (p["instruction"][:120], p["response"][:120])
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def run_pipeline(pdf_path: str, output_path: str):
    print(f"\n{'='*55}")
    print(f"  AgriConnect PDF → Dataset Pipeline")
    print(f"{'='*55}")

    pdf_type = detect_pdf_type(pdf_path)
    print(f"\n[1] PDF type detected: {pdf_type.upper()}")

    if pdf_type == "scanned":
        print("    Scanned PDF — text layer is empty.")
        print("    Install pytesseract + pdf2image and run OCR first.")
        return

    print(f"[2] Extracting pages from: {pdf_path}")
    pages = extract_pages(pdf_path)
    print(f"    -> {len(pages)} pages extracted")

    all_pairs = []
    errors = 0

    for page in pages:
        pnum = page["page_num"]

        for df in page["tables"]:
            try:
                records = table_to_records(df)
                for rec in records:
                    pair = record_to_instruction_pair(rec, source=f"table_page_{pnum}")
                    if pair:
                        all_pairs.append(pair)
            except Exception as e:
                errors += 1
                print(f"    [warn] page {pnum} table skipped: {e}")

        text = clean_text(page["raw_text"])
        chunks = [c.strip() for c in text.split("\n\n") if len(c.strip()) > 80]
        for chunk in chunks:
            try:
                pair = text_chunk_to_pair(chunk, pnum)
                if pair:
                    all_pairs.append(pair)
            except Exception as e:
                errors += 1

    print(f"[3] Raw pairs extracted:   {len(all_pairs)}  (skipped errors: {errors})")

    all_pairs = deduplicate(all_pairs)
    print(f"[4] After deduplication:   {len(all_pairs)}")

    out = Path(output_path)
    with out.open("w", encoding="utf-8") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"[5] Dataset saved to:      {out.resolve()}")

    if all_pairs:
        ex = all_pairs[0]
        print(f"\n{'─'*40}")
        print(f"  Sample pair:")
        print(f"  INSTRUCTION: {ex['instruction'][:100]}...")
        print(f"  RESPONSE:    {ex['response'][:100]}...")
        print(f"{'─'*40}")

    print(f"\n  Total training pairs: {len(all_pairs)}")
    print(f"  Next: merge with Kaggle CSV, then fine-tune with QLoRA.\n")


# ─────────────────────────────────────────────
# MERGE UTILITY
# ─────────────────────────────────────────────

def merge_kaggle_csv(csv_path: str, jsonl_path: str, output_path: str):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    pairs = []
    for _, row in df.iterrows():
        parts = []
        if "n"           in row: parts.append(f"N={row['n']:.1f} kg/ha")
        if "p"           in row: parts.append(f"P={row['p']:.1f} kg/ha")
        if "k"           in row: parts.append(f"K={row['k']:.1f} kg/ha")
        if "ph"          in row: parts.append(f"pH={row['ph']:.2f}")
        if "temperature" in row: parts.append(f"temp={row['temperature']:.1f}°C")
        if "humidity"    in row: parts.append(f"humidity={row['humidity']:.1f}%")
        if "rainfall"    in row: parts.append(f"rainfall={row['rainfall']:.1f}mm")

        crop = str(row.get("label", row.get("crop", "unknown"))).strip().title()
        fert = str(row.get("fertilizer", "")).strip()

        prompt = (
            f"Soil and climate readings: {', '.join(parts)}. "
            f"This is for Odisha region. Recommend the best crop and fertilizer."
        )
        response = f"Recommended crop: {crop}."
        if fert and fert.lower() not in ("nan", ""):
            response += f" Apply {fert.upper()} as primary fertilizer."

        pairs.append({"instruction": prompt, "response": response,
                      "source": "kaggle_csv", "metadata": {}})

    pdf_pairs = []
    if Path(jsonl_path).exists():
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                pdf_pairs.append(json.loads(line))

    all_pairs = deduplicate(pdf_pairs + pairs)

    with open(output_path, "w", encoding="utf-8") as f:
        for p in all_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"Merged dataset: {len(all_pairs)} pairs -> {output_path}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AgriConnect PDF -> JSONL pipeline")
    subparsers = parser.add_subparsers(dest="command")

    p_pdf = subparsers.add_parser("pdf", help="Convert a PDF to JSONL dataset")
    p_pdf.add_argument("--pdf", required=True)
    p_pdf.add_argument("--out", default="agriconnect_dataset.jsonl")

    p_merge = subparsers.add_parser("merge", help="Merge Kaggle CSV + PDF JSONL")
    p_merge.add_argument("--csv",   required=True)
    p_merge.add_argument("--jsonl", required=True)
    p_merge.add_argument("--out",   default="agriconnect_merged.jsonl")

    args = parser.parse_args()

    if args.command == "pdf":
        run_pipeline(args.pdf, args.out)
    elif args.command == "merge":
        merge_kaggle_csv(args.csv, args.jsonl, args.out)
    else:
        parser.print_help()