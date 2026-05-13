import json
import argparse
import os
import re
import statistics
from typing import Optional
from collections import defaultdict
from tqdm import tqdm

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("请安装 openai: pip install openai")


def _parse_judge_score(raw: str) -> float:
    """解析 0-100 分；解析失败返回 50.0"""
    s = str(raw or "").strip()
    m = re.search(r"\b(\d{1,3})\b", s)
    if m:
        val = float(m.group(1))
        return min(100.0, max(0.0, val))
    return 50.0


def judge_score(
    question: str,
    prediction: str,
    reference: str,
    client: OpenAI,
    model: str = "gpt-4o",
) -> float:
    """LLM-as-Judge 打分 0-100。prediction 与 reference 完全一致时直接返回 100。"""
    ref_str = str(reference or "").strip() if reference is not None else ""
    if not ref_str:
        return 50.0
    pred_str = str(prediction or "").strip().lower()
    ref_norm = ref_str.lower()
    if pred_str == ref_norm:
        return 100.0

    prompt = f"""You are a strict evaluator. Score how well the model's prediction answers the question, given the reference answer.

Question: {question}
Reference answer: {ref_str}
Model's prediction: {str(prediction or "").strip()}

Scoring (0-100):
- 100: Fully correct, semantically equivalent or highly accurate
- 75-99: Mostly correct, minor omissions or paraphrasing
- 50-74: Partially correct
- 25-49: Mostly wrong, significant errors
- 0-24: Completely wrong or irrelevant

Reply with ONLY an integer score from 0 to 100 (no other text)."""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Reply with only an integer from 0 to 100."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=10,
        )
        raw = (response.choices[0].message.content or "").strip()
        return _parse_judge_score(raw)
    except Exception as e:
        print(f"[WARN] LLM Judge API error: {e}")
        return 50.0


def main():
    parser = argparse.ArgumentParser(
        description="对 results.jsonl 重新进行 LLM-as-Judge 打分并更新文件"
    )
    parser.add_argument(
        "input",
        type=str,
        help="输入 JSONL 文件路径（如 output/gpt/results_B/results.jsonl）",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="输出文件路径；默认在输入路径后加 _judged.jsonl",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="直接覆盖原文件（与 -o 互斥）",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gpt-4o",
        help="Judge 模型，默认 gpt-4o",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API Key，默认用环境变量 OPENAI_API_KEY",
    )

    args = parser.parse_args()

    if args.in_place and args.output:
        parser.error("--in-place 与 -o 不能同时使用")

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("请设置 OPENAI_API_KEY 或使用 --api-key")

    input_path = args.input
    if not os.path.isabs(input_path):
        input_path = os.path.join(os.path.dirname(__file__), input_path)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    if args.in_place:
        output_path = input_path
        temp_path = input_path + ".tmp"
        write_to = temp_path
    elif args.output:
        output_path = args.output if os.path.isabs(args.output) else os.path.join(os.path.dirname(__file__), args.output)
        write_to = output_path
    else:
        base, ext = os.path.splitext(input_path)
        output_path = base + "_judged" + (ext or ".jsonl")
        write_to = output_path

    os.makedirs(os.path.dirname(write_to) or ".", exist_ok=True)

    client = OpenAI(api_key=api_key, base_url=os.getenv("OPENAI_API_BASE"))

    entries = []
    qa_results = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            entries.append(entry)
            if entry.get("type") == "qa_result":
                qa_results.append(entry)

    print(f"共 {len(entries)} 行，其中 {len(qa_results)} 条 qa_result 待重新打分")

    # 为每条 qa_result 赋分，并更新 entry（引用同对象）
    for entry in tqdm(qa_results, desc="LLM Judge"):
        question = entry.get("question", "")
        prediction = entry.get("prediction", "")
        reference = entry.get("reference")
        score = judge_score(question, prediction, reference, client, model=args.judge_model)

        if "metrics" not in entry:
            entry["metrics"] = {}
        entry["metrics"]["llm_as_judge"] = score
        entry["llm_as_judge"] = score

    # 若存在 run_summary，基于更新后的 llm_as_judge 重新计算 aggregate
    for entry in entries:
        if entry.get("type") == "run_summary":
            agg = entry.get("aggregate_metrics_for_this_run", {})
            if agg and qa_results:
                judge_scores = [e.get("llm_as_judge", 50.0) for e in qa_results]
                mean_s = statistics.mean(judge_scores)
                std_s = statistics.stdev(judge_scores) if len(judge_scores) > 1 else 0.0
                if "overall" in agg:
                    agg["overall"]["llm_as_judge"] = {
                        "mean": mean_s,
                        "std": std_s,
                        "median": statistics.median(judge_scores),
                        "min": min(judge_scores),
                        "max": max(judge_scores),
                        "count": len(judge_scores),
                    }
                # 按 category 更新 llm_as_judge
                by_cat = defaultdict(list)
                for e in qa_results:
                    c = str(e.get("category", "overall"))
                    by_cat[f"category_{c}"].append(e.get("llm_as_judge", 50.0))
                for cat_key, scores in by_cat.items():
                    if cat_key in agg and scores:
                        agg[cat_key]["llm_as_judge"] = {
                            "mean": statistics.mean(scores),
                            "std": statistics.stdev(scores) if len(scores) > 1 else 0.0,
                            "median": statistics.median(scores),
                            "min": min(scores),
                            "max": max(scores),
                            "count": len(scores),
                        }
            break

    with open(write_to, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    if args.in_place:
        os.replace(temp_path, input_path)
        print(f"已覆盖: {input_path}")
    else:
        print(f"已保存: {write_to}")

    if qa_results:
        scores = [e.get("llm_as_judge", 50.0) for e in qa_results]
        print(f"LLM-as-Judge: mean={sum(scores)/len(scores):.2f}, min={min(scores):.0f}, max={max(scores):.0f}")


if __name__ == "__main__":
    main()
