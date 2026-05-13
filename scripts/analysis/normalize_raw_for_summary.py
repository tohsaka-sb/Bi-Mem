import json
from pathlib import Path
from typing import Dict, Optional, Tuple


TARGETS = [
    "output/locomo/model_test/origin_gpt5.raw_responses.jsonl",
    "output/locomo/model_test/origin_gpt5_mini.raw_responses.jsonl",
    "output/locomo/model_test/origin_gpt4o.raw_responses.jsonl",
    "output/locomo/model_test/origin_gpt4o_mini.raw_responses.jsonl",
    "output/longmem/model_test/origin_gpt5.raw_responses.jsonl",
    "output/longmem/model_test/origin_gpt5_mini.raw_responses.jsonl",
    "output/longmem/model_test/origin_gpt4o.raw_responses.jsonl",
    "output/longmem/model_test/origin_gpt4o_mini.raw_responses.jsonl",
]


def load_reference_map(result_file: Path) -> Tuple[Dict[Tuple[str, int], str], Dict[Tuple[str, str], str]]:
    by_id: Dict[Tuple[str, int], str] = {}
    by_q: Dict[Tuple[str, str], str] = {}
    if not result_file.exists():
        return by_id, by_q

    with result_file.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("type") != "qa_result":
                continue
            sample_id = str(row.get("sample_id", ""))
            qid = row.get("question_id_in_sample")
            question = str(row.get("question", "")).strip()
            reference = row.get("reference")
            if reference is None:
                continue
            reference_str = str(reference)
            if isinstance(qid, int):
                by_id[(sample_id, qid)] = reference_str
            if question:
                by_q[(sample_id, question)] = reference_str

    return by_id, by_q


def exact_match(prediction: str, reference: Optional[str]) -> int:
    if reference is None:
        return 0
    return int(str(prediction).strip().lower() == str(reference).strip().lower())


def normalize_one(raw_file: Path) -> None:
    result_file = raw_file.with_name(raw_file.name.replace(".raw_responses.jsonl", ".jsonl"))
    by_id, by_q = load_reference_map(result_file)

    out_lines = []
    updated = 0
    with raw_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                out_lines.append(line)
                continue

            sample_id = str(row.get("sample_id", ""))
            qid = row.get("question_id_in_sample")
            question = str(row.get("question", "")).strip()
            prediction = str(row.get("prediction", "") or "")

            ref = None
            if isinstance(qid, int):
                ref = by_id.get((sample_id, qid))
            if ref is None and question:
                ref = by_q.get((sample_id, question))

            f1 = row.get("raw_f1")
            bleu1 = row.get("raw_bleu1")
            f1 = float(f1) if isinstance(f1, (int, float)) else 0.0
            bleu1 = float(bleu1) if isinstance(bleu1, (int, float)) else 0.0

            row["type"] = "qa_result"
            if "category" not in row:
                row["category"] = 4
            row["reference"] = ref
            row["metrics"] = {
                "exact_match": exact_match(prediction, ref),
                "f1": f1,
                "bleu1": bleu1,
            }
            row["f1"] = f1
            row["bleu1"] = bleu1
            row.setdefault("llm_as_judge", 50.0)

            out_lines.append(json.dumps(row, ensure_ascii=False))
            updated += 1

    with raw_file.open("w", encoding="utf-8") as f:
        for line in out_lines:
            f.write(line + "\n")

    print(f"[DONE] {raw_file} | updated_rows={updated}")


def main():
    root = Path(__file__).resolve().parent
    for rel in TARGETS:
        p = (root / rel).resolve()
        if not p.exists():
            print(f"[SKIP] not found: {p}")
            continue
        normalize_one(p)


if __name__ == "__main__":
    main()


