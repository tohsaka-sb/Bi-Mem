import json
import argparse
import os
import sys
import statistics
from collections import defaultdict

CORE_METRICS = ["f1", "bleu1", "llm_as_judge"]


def load_and_deduplicate(file_path: str):
    """读取 JSONL，按 (sample_id, question) 去重，取最后一条。"""
    unique_results = {}
    metadata = {}
    run_summaries = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            entry_type = data.get("type")
            if entry_type == "metadata":
                metadata = data
            elif entry_type == "qa_result":
                sample_id = str(data.get("sample_id", ""))
                question = (data.get("question") or "").strip()
                key = (sample_id, question)
                unique_results[key] = data
            elif entry_type == "run_summary":
                run_summaries.append(data)

    return metadata, list(unique_results.values()), run_summaries


def extract_metrics(entry: dict) -> dict:
    """从 qa_result 提取核心指标，优先用顶层字段。"""
    m = {}
    for name in CORE_METRICS:
        if name in entry:
            m[name] = float(entry[name])
        elif "metrics" in entry and isinstance(entry["metrics"], dict) and name in entry["metrics"]:
            m[name] = float(entry["metrics"][name])
        else:
            m[name] = None
    return m


def summarize_longmem(file_path: str, output_json: str = None, verbose: bool = False):
    metadata, qa_results, run_summaries = load_and_deduplicate(file_path)

    if not qa_results:
        print(f"[WARN] 未找到有效的 qa_result：{file_path}")
        return

    n = len(qa_results)
    print(f"=== LongMemEval 结果总结 ===")
    print(f"文件: {file_path}")
    print(f"有效问题数: {n}（已去重）")

    if metadata:
        print(f"\n元信息:")
        for k in ["model", "backend", "dataset", "ratio", "max_samples"]:
            if k in metadata:
                print(f"  {k}: {metadata[k]}")

    # 按样本收集指标
    all_metrics = []
    for r in qa_results:
        m = extract_metrics(r)
        all_metrics.append(m)

    # 计算汇总
    agg = defaultdict(lambda: {"sum": 0.0, "count": 0, "vals": []})
    for m in all_metrics:
        for name, val in m.items():
            if val is not None:
                agg[name]["sum"] += val
                agg[name]["count"] += 1
                agg[name]["vals"].append(val)

    print(f"\n--- 核心指标汇总 ---")
    for name in CORE_METRICS:
        a = agg[name]
        if a["count"] == 0:
            print(f"  {name}: N/A（未提供）")
            continue
        mean_val = a["sum"] / a["count"]
        vals = a["vals"]
        std_val = (sum((x - mean_val) ** 2 for x in vals) / len(vals)) ** 0.5 if len(vals) > 1 else 0.0
        print(f"  {name}: mean={mean_val:.4f}, std={std_val:.4f}, min={min(vals):.4f}, max={max(vals):.4f} (n={a['count']})")

    # 若存在 run_summary，可对比
    if run_summaries and verbose:
        last_summary = run_summaries[-1]
        agg_run = last_summary.get("aggregate_metrics_for_this_run", {}).get("overall", {})
        if agg_run:
            print(f"\n--- run_summary 中的 overall ---")
            for name in CORE_METRICS:
                if name in agg_run:
                    s = agg_run[name]
                    print(f"  {name}: mean={s.get('mean', 0):.4f}, std={s.get('std', 0):.4f}")

    # 按 category  breakdown（LongMemEval 通常全为 category 4）
    by_cat = defaultdict(list)
    for r in qa_results:
        cat = int(r.get("category", 4))
        m = extract_metrics(r)
        by_cat[cat].append(m)

    if len(by_cat) > 1 or verbose:
        print(f"\n--- 按 Category 分布 ---")
        for cat in sorted(by_cat.keys()):
            ms = by_cat[cat]
            cat_agg = defaultdict(list)
            for m in ms:
                for name, val in m.items():
                    if val is not None:
                        cat_agg[name].append(val)
            print(f"  Category {cat} (n={len(ms)}):")
            for name in CORE_METRICS:
                if name in cat_agg and cat_agg[name]:
                    v = cat_agg[name]
                    mean_v = sum(v) / len(v)
                    print(f"    {name}: {mean_v:.4f}")

    # 输出 JSON
    summary = {
        "file": file_path,
        "n_questions": n,
        "metadata": metadata,
        "core_metrics": {},
    }
    for name in CORE_METRICS:
        a = agg[name]
        if a["count"] > 0:
            vals = a["vals"]
            summary["core_metrics"][name] = {
                "mean": statistics.mean(vals),
                "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                "min": min(vals),
                "max": max(vals),
                "count": a["count"],
            }

    if output_json:
        out_path = os.path.abspath(output_json)
        d = os.path.dirname(out_path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\n已保存汇总到: {out_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="总结 LongMemEval 评估结果")
    parser.add_argument("file_path", type=str, help="results.jsonl 文件路径")
    parser.add_argument("-o", "--output", type=str, default=None, help="输出 JSON 汇总路径")
    parser.add_argument("-v", "--verbose", action="store_true", help="显示更详细信息")

    args = parser.parse_args()

    file_path = args.file_path
    if not os.path.isabs(file_path):
        file_path = os.path.join(os.path.dirname(__file__), file_path)

    if not os.path.exists(file_path):
        print(f"[Error] 文件不存在: {file_path}")
        sys.exit(1)

    summarize_longmem(file_path, output_json=args.output, verbose=args.verbose)


if __name__ == "__main__":
    main()
