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


def analyze_locomo():
    g5 = load_rows(ROOT / "output/locomo/model_test/origin_gpt5.raw_responses.jsonl")
    g4 = load_rows(ROOT / "output/locomo/model_test/origin_gpt4o.raw_responses.jsonl")

    def key(r):
        return (
            str(r.get("sample_id", "")),
            int(r.get("question_id_in_sample", -1)),
            str(r.get("question", "")).strip(),
        )

    m5 = {key(r): r for r in g5}
    m4 = {key(r): r for r in g4}
    common = sorted(set(m5.keys()) & set(m4.keys()))

    agg = defaultdict(float)
    cat = defaultdict(lambda: defaultdict(float))
    catn = defaultdict(int)

    for k in common:
        r5 = m5[k]
        r4 = m4[k]
        f5 = float(r5.get("f1", 0))
        f4 = float(r4.get("f1", 0))
        b5 = float(r5.get("bleu1", 0))
        b4 = float(r4.get("bleu1", 0))
        c = int(r5.get("category", -1))

        agg["f5"] += f5
        agg["f4"] += f4
        agg["b5"] += b5
        agg["b4"] += b4
        cat[c]["f5"] += f5
        cat[c]["f4"] += f4
        cat[c]["b5"] += b5
        cat[c]["b4"] += b4
        catn[c] += 1

    print("== LOCOMO common completed QAs ==")
    print(f"common_n={len(common)}")
    if common:
        n = len(common)
        print(
            f"overall: gpt5_f1={agg['f5']/n:.4f}, gpt4o_f1={agg['f4']/n:.4f}, "
            f"gpt5_b1={agg['b5']/n:.4f}, gpt4o_b1={agg['b4']/n:.4f}"
        )
    for c in sorted(catn):
        n = catn[c]
        print(
            f"cat_{c} n={n}: gpt5_f1={cat[c]['f5']/n:.4f}, gpt4o_f1={cat[c]['f4']/n:.4f}, "
            f"gpt5_b1={cat[c]['b5']/n:.4f}, gpt4o_b1={cat[c]['b4']/n:.4f}"
        )

    diffs = []
    for k in common:
        r5 = m5[k]
        r4 = m4[k]
        diffs.append((float(r5.get("f1", 0)) - float(r4.get("f1", 0)), r5, r4))
    diffs.sort(key=lambda x: x[0], reverse=True)

    print("\nTop 5 where GPT-5 > GPT-4o (by f1 delta):")
    for d, r5, r4 in diffs[:5]:
        print("---")
        print(f"delta_f1={d:.4f} | cat={r5.get('category')} | q={r5.get('question')}")
        print(f"ref={r5.get('reference')}")
        print(f"gpt5={r5.get('prediction')}")
        print(f"gpt4o={r4.get('prediction')}")

    print("\nTop 5 where GPT-5 < GPT-4o (by f1 delta):")
    for d, r5, r4 in diffs[-5:]:
        print("---")
        print(f"delta_f1={d:.4f} | cat={r5.get('category')} | q={r5.get('question')}")
        print(f"ref={r5.get('reference')}")
        print(f"gpt5={r5.get('prediction')}")
        print(f"gpt4o={r4.get('prediction')}")


def analyze_longmem():
    g5 = load_rows(ROOT / "output/longmem/model_test/origin_gpt5.raw_responses.jsonl")
    gm = load_rows(ROOT / "output/longmem/model_test/origin_gpt5_mini.raw_responses.jsonl")

    def key(r):
        return (str(r.get("sample_id", "")), str(r.get("question", "")).strip())

    m5 = {key(r): r for r in g5}
    mm = {key(r): r for r in gm}
    common = sorted(set(m5.keys()) & set(mm.keys()))

    print("\n== LONGMEM common completed QAs ==")
    print(f"common_n={len(common)}")

    agg = defaultdict(float)
    for k in common:
        r5 = m5[k]
        rm = mm[k]
        agg["f5"] += float(r5.get("f1", 0))
        agg["fm"] += float(rm.get("f1", 0))
        agg["b5"] += float(r5.get("bleu1", 0))
        agg["bm"] += float(rm.get("bleu1", 0))

    if common:
        n = len(common)
        print(
            f"overall: gpt5_f1={agg['f5']/n:.4f}, gpt5mini_f1={agg['fm']/n:.4f}, "
            f"gpt5_b1={agg['b5']/n:.4f}, gpt5mini_b1={agg['bm']/n:.4f}"
        )

    diffs = []
    for k in common:
        r5 = m5[k]
        rm = mm[k]
        diffs.append((float(rm.get("f1", 0)) - float(r5.get("f1", 0)), r5, rm))
    diffs.sort(key=lambda x: x[0], reverse=True)

    print("\nTop 8 where GPT-5-mini > GPT-5 (by f1 delta):")
    for d, r5, rm in diffs[:8]:
        print("---")
        print(f"delta_f1_mini_minus_5={d:.4f} | sid={r5.get('sample_id')}")
        print(f"q={r5.get('question')}")
        print(f"ref={r5.get('reference')}")
        print(f"gpt5={r5.get('prediction')}")
        print(f"gpt5mini={rm.get('prediction')}")

    print("\nTop 5 where GPT-5-mini < GPT-5 (by f1 delta):")
    for d, r5, rm in diffs[-5:]:
        print("---")
        print(f"delta_f1_mini_minus_5={d:.4f} | sid={r5.get('sample_id')}")
        print(f"q={r5.get('question')}")
        print(f"ref={r5.get('reference')}")
        print(f"gpt5={r5.get('prediction')}")
        print(f"gpt5mini={rm.get('prediction')}")


if __name__ == "__main__":
    analyze_locomo()
    analyze_longmem()


