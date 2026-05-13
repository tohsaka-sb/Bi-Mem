import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load_rows(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(row.get("type")) != "qa_result":
                continue
            if not str(row.get("prediction", "")).strip():
                continue
            rows.append(row)
    return rows


def key_longmem(r):
    return (str(r.get("sample_id", "")), str(r.get("question", "")).strip())


def key_locomo(r):
    return (
        str(r.get("sample_id", "")),
        int(r.get("question_id_in_sample", -1)),
        str(r.get("question", "")).strip(),
    )


def print_longmem_g5_vs_g4():
    g5 = load_rows(ROOT / "output/longmem/model_test/origin_gpt5.raw_responses.jsonl")
    g4 = load_rows(ROOT / "output/longmem/model_test/origin_gpt4o.raw_responses.jsonl")
    m5 = {key_longmem(r): r for r in g5}
    m4 = {key_longmem(r): r for r in g4}
    common = sorted(set(m5.keys()) & set(m4.keys()))

    print("=== LongMemEval: GPT-5 vs GPT-4o (common completed) ===")
    print(f"common_n={len(common)}")
    if not common:
        return

    s = defaultdict(float)
    for k in common:
        r5, r4 = m5[k], m4[k]
        s["f5"] += float(r5.get("f1", 0))
        s["f4"] += float(r4.get("f1", 0))
        s["b5"] += float(r5.get("bleu1", 0))
        s["b4"] += float(r4.get("bleu1", 0))
    n = len(common)
    print(
        f"overall: gpt5_f1={s['f5']/n:.4f}, gpt4o_f1={s['f4']/n:.4f}, "
        f"gpt5_b1={s['b5']/n:.4f}, gpt4o_b1={s['b4']/n:.4f}"
    )

    diffs = []
    for k in common:
        r5, r4 = m5[k], m4[k]
        df = float(r5.get("f1", 0)) - float(r4.get("f1", 0))
        db = float(r5.get("bleu1", 0)) - float(r4.get("bleu1", 0))
        diffs.append((df, db, r5, r4))
    diffs.sort(key=lambda x: x[0], reverse=True)

    print("\nTop 10 cases where GPT-5 beats GPT-4o by F1:")
    for df, db, r5, r4 in diffs[:10]:
        print("---")
        print(f"delta_f1={df:.4f}, delta_bleu1={db:.4f}, sid={r5.get('sample_id')}")
        print(f"Q: {r5.get('question')}")
        print(f"Ref: {r5.get('reference')}")
        print(f"GPT-5: {r5.get('prediction')}")
        print(f"GPT-4o: {r4.get('prediction')}")

    print("\nTop 6 cases where GPT-5 underperforms GPT-4o:")
    for df, db, r5, r4 in diffs[-6:]:
        print("---")
        print(f"delta_f1={df:.4f}, delta_bleu1={db:.4f}, sid={r5.get('sample_id')}")
        print(f"Q: {r5.get('question')}")
        print(f"Ref: {r5.get('reference')}")
        print(f"GPT-5: {r5.get('prediction')}")
        print(f"GPT-4o: {r4.get('prediction')}")


def print_locomo_four_model_study():
    models = {
        "gpt5": load_rows(ROOT / "output/locomo/model_test/origin_gpt5.raw_responses.jsonl"),
        "gpt5mini": load_rows(ROOT / "output/locomo/model_test/origin_gpt5_mini.raw_responses.jsonl"),
        "gpt4o": load_rows(ROOT / "output/locomo/model_test/origin_gpt4o.raw_responses.jsonl"),
        "gpt4omini": load_rows(ROOT / "output/locomo/model_test/origin_gpt4o_mini.raw_responses.jsonl"),
    }
    maps = {name: {key_locomo(r): r for r in rows} for name, rows in models.items()}
    common = sorted(set.intersection(*(set(m.keys()) for m in maps.values())))
    print("\n=== LoCoMo10: 4-model common completed case study ===")
    print(f"common_n={len(common)}")
    if not common:
        return

    cat_stat = defaultdict(lambda: defaultdict(float))
    cat_n = defaultdict(int)
    for k in common:
        cat = int(maps["gpt5"][k].get("category", -1))
        cat_n[cat] += 1
        for m in maps:
            cat_stat[cat][f"{m}_f1"] += float(maps[m][k].get("f1", 0))
            cat_stat[cat][f"{m}_b1"] += float(maps[m][k].get("bleu1", 0))

    for cat in sorted(cat_n):
        n = cat_n[cat]
        print(f"\nCategory {cat} (n={n}) averages:")
        for m in ["gpt5", "gpt5mini", "gpt4o", "gpt4omini"]:
            print(
                f"  {m}: f1={cat_stat[cat][f'{m}_f1']/n:.4f}, "
                f"bleu1={cat_stat[cat][f'{m}_b1']/n:.4f}"
            )

    spreads = []
    for k in common:
        f1s = {m: float(maps[m][k].get("f1", 0)) for m in maps}
        b1s = {m: float(maps[m][k].get("bleu1", 0)) for m in maps}
        spread = max(f1s.values()) - min(f1s.values())
        spreads.append((spread, f1s, b1s, maps["gpt5"][k]))
    spreads.sort(key=lambda x: x[0], reverse=True)

    print("\nTop 16 high-disagreement cases across 4 models:")
    for spread, f1s, b1s, base in spreads[:16]:
        print("---")
        print(
            f"cat={base.get('category')} qid={base.get('question_id_in_sample')} "
            f"spread_f1={spread:.4f}"
        )
        print(f"Q: {base.get('question')}")
        print(f"Ref: {base.get('reference')}")
        for m in ["gpt5", "gpt5mini", "gpt4o", "gpt4omini"]:
            r = maps[m][key_locomo(base)]
            print(
                f"{m}: f1={f1s[m]:.4f}, b1={b1s[m]:.4f}, pred={r.get('prediction')}"
            )


if __name__ == "__main__":
    print_longmem_g5_vs_g4()
    print_locomo_four_model_study()


