import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", str(text).lower()) if text else []


def compute_f1(prediction: str, reference: str) -> float:
    pred_tokens = set(tokenize(prediction))
    ref_tokens = set(tokenize(reference))
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = pred_tokens & ref_tokens
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_bleu1(prediction: str, reference: str) -> float:
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0

    from collections import Counter

    pred_counter = Counter(pred_tokens)
    ref_counter = Counter(ref_tokens)
    overlap = 0
    for token, cnt in pred_counter.items():
        overlap += min(cnt, ref_counter.get(token, 0))

    precision = overlap / len(pred_tokens) if pred_tokens else 0.0
    if precision == 0.0:
        return 0.0

    ref_len = len(ref_tokens)
    pred_len = len(pred_tokens)
    if pred_len == 0:
        return 0.0

    if pred_len > ref_len:
        bp = 1.0
    else:
        bp = 2.718281828459045 ** (1 - (ref_len / pred_len))

    return float(bp * precision)


def load_reference_map(result_jsonl: Path) -> Tuple[Dict[Tuple[str, int], str], Dict[Tuple[str, str], str]]:
    by_id: Dict[Tuple[str, int], str] = {}
    by_question: Dict[Tuple[str, str], str] = {}

    with result_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            if row.get("type") != "qa_result":
                continue

            sample_id = str(row.get("sample_id", ""))
            question = str(row.get("question", "")).strip()
            reference = row.get("reference")
            if reference is None:
                continue
            reference_str = str(reference)

            qid = row.get("question_id_in_sample")
            if isinstance(qid, int):
                by_id[(sample_id, qid)] = reference_str
            if question:
                by_question[(sample_id, question)] = reference_str

    return by_id, by_question


def update_raw_file(raw_path: Path) -> Tuple[int, int]:
    result_path = raw_path.with_name(raw_path.name.replace(".raw_responses.jsonl", ".jsonl"))
    if not result_path.exists():
        print(f"[SKIP] result file not found for raw file: {raw_path}")
        return 0, 0

    by_id, by_question = load_reference_map(result_path)
    updated_rows = []
    updated_count = 0
    missing_ref_count = 0

    with raw_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                updated_rows.append(line.rstrip("\n"))
                continue

            sample_id = str(row.get("sample_id", ""))
            qid = row.get("question_id_in_sample")
            question = str(row.get("question", "")).strip()
            prediction = str(row.get("prediction", "") or "")

            reference: Optional[str] = None
            if isinstance(qid, int):
                reference = by_id.get((sample_id, qid))
            if reference is None and question:
                reference = by_question.get((sample_id, question))

            if reference is None:
                missing_ref_count += 1
                row["raw_f1"] = None
                row["raw_bleu1"] = None
                row["raw_metric_status"] = "missing_reference"
            else:
                row["raw_f1"] = compute_f1(prediction, reference)
                row["raw_bleu1"] = compute_bleu1(prediction, reference)
                row["raw_metric_status"] = "ok"
                updated_count += 1

            updated_rows.append(json.dumps(row, ensure_ascii=False))

    with raw_path.open("w", encoding="utf-8") as f:
        for row in updated_rows:
            f.write(row + "\n")

    return updated_count, missing_ref_count


def discover_raw_files(project_root: Path) -> List[Path]:
    targets = [
        project_root / "output" / "locomo" / "model_test",
        project_root / "output" / "longmem" / "model_test",
    ]
    found: List[Path] = []
    for directory in targets:
        if not directory.exists():
            continue
        found.extend(directory.glob("*.raw_responses.jsonl"))

    unique = {}
    for p in found:
        unique[str(p.resolve())] = p
    return list(unique.values())


def main():
    parser = argparse.ArgumentParser(
        description="Compute raw-response F1/BLEU1 and write back in-place."
    )
    parser.add_argument(
        "--root",
        type=str,
        default=str(Path(__file__).resolve().parent),
        help="Project root path (default: script directory).",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    raw_files = discover_raw_files(root)

    if not raw_files:
        print("[INFO] No raw response files found.")
        return

    total_updated = 0
    total_missing = 0
    for raw_file in sorted(raw_files):
        updated, missing = update_raw_file(raw_file)
        total_updated += updated
        total_missing += missing
        print(f"[DONE] {raw_file} | updated={updated}, missing_reference={missing}")

    print(
        f"[SUMMARY] files={len(raw_files)}, rows_with_scores={total_updated}, rows_missing_reference={total_missing}"
    )


if __name__ == "__main__":
    main()


