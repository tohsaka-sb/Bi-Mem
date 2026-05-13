import pickle
import json
import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter
from tqdm import tqdm

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("Please install openai: pip install openai")


def load_memory_cache(cache_path: str) -> Tuple[str, List[Tuple[str, str]]]:
    """
    从 pkl 加载记忆，提取 Level 2 Persona 和 Level 1 Scene 节点。
    Returns: (persona_text, [(scene_id, scene_content), ...])
    """
    with open(cache_path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict) or "graph" not in data:
        raise ValueError("Invalid cache format: expected dict with 'graph' key")
    G = data["graph"]

    persona_parts = []
    scene_list = []

    for node_id, attrs in G.nodes(data=True):
        level = attrs.get("level")
        content = attrs.get("content", "").strip()
        if not content:
            continue
        if level == "persona":
            tag = attrs.get("tags", ["Unknown"])[0] if attrs.get("tags") else "Unknown"
            persona_parts.append(f"[{tag}]: {content}")
        elif level == "scene":
            scene_list.append((node_id, content))

    persona_text = "\n\n".join(persona_parts) if persona_parts else "(No persona nodes)"
    return persona_text, scene_list


def call_gpt4o_judge(persona: str, scene_content: str, client: OpenAI) -> int:
    """调用 GPT-4o 进行一致性打分，返回 1-5 分（5=完全一致，1=相矛盾）。"""
    prompt = f"""你是一个严格的评估者。请根据以下材料对「场景记忆与用户画像的一致性」打分。

【用户画像 Persona】
{persona}

【场景记忆 Scene】
{scene_content}

打分标准（1-5 分）：
- 5 分：完全一致，场景与画像高度契合，无任何冲突
- 4 分：基本一致，仅有轻微差异
- 3 分：中性，无法明确判断或信息不足
- 2 分：存在一定矛盾或不一致
- 1 分：明显相矛盾，逻辑冲突严重

要求：仅回复一个数字 1、2、3、4 或 5，不要添加其他内容。

你的打分："""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "你只回复一个数字 1、2、3、4 或 5，不要添加任何其他内容。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=10,
        )
        raw = (response.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"\n[WARN] API error: {e}")
        return 3

    return _parse_score(raw)


def _parse_score(raw: str) -> int:
    """从模型输出中解析 1-5 分，无效则返回 3（中性）。"""
    m = re.search(r"[1-5]", raw)
    if m:
        return int(m.group(0))
    return 3


def run_alignment_test(
    cache_path: str,
    output_path: str,
    group_label: str = "",
    api_key: Optional[str] = None,
) -> Dict:
    """
    对单个 cache 运行对齐度评估。
    - cache_path: memory_cache_*.pkl 路径
    - output_path: 结果 JSON 路径（用于续传与保存）
    - group_label: 分组标签，如 "A" 或 "B"
    """
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("请设置 OPENAI_API_KEY 环境变量或传入 --api-key")

    client = OpenAI(api_key=api_key, base_url=os.getenv("OPENAI_API_BASE"))

    persona_text, scene_list = load_memory_cache(cache_path)
    scene_ids = [s[0] for s in scene_list]
    scene_contents = {s[0]: s[1] for s in scene_list}

    # 续传：加载已有结果（兼容旧版 Yes/No/Neutral，自动转为 1/5/3）
    judgments = {}
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                saved = json.load(f)
            raw_judgments = saved.get("judgments", {})
            for k, v in raw_judgments.items():
                if isinstance(v, int) and 1 <= v <= 5:
                    judgments[k] = v
                elif isinstance(v, str):
                    judgments[k] = {"Yes": 1, "No": 5, "Neutral": 3}.get(v, 3)
                else:
                    judgments[k] = 3
            print(f"[续传] 已加载 {len(judgments)} 条已有判断")
        except Exception as e:
            print(f"[WARN] 加载已有结果失败: {e}")

    # 待评估的 scene
    to_judge = [(sid, scene_contents[sid]) for sid in scene_ids if sid not in judgments]

    if not to_judge:
        print("所有 Scene 已评估完毕，跳过 API 调用。")
    else:
        print(f"待评估: {len(to_judge)} 个 Scene")
        for scene_id, scene_content in tqdm(to_judge, desc="LLM Judge"):
            score = call_gpt4o_judge(persona_text, scene_content, client)
            judgments[scene_id] = score

            # 每轮保存，便于续传
            result = {
                "cache_path": cache_path,
                "group_label": group_label,
                "persona_text": persona_text,
                "scene_ids": scene_ids,
                "judgments": judgments,
            }
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

    # 统计（1-5 分）
    scores = list(judgments.values())
    total = len(scores)
    counter = Counter(scores)
    mean_score = sum(scores) / total if total else 0
    variance = sum((s - mean_score) ** 2 for s in scores) / total if total else 0
    std_score = variance ** 0.5

    summary = {
        "group_label": group_label,
        "cache_path": cache_path,
        "total_scenes": total,
        "mean_score": round(mean_score, 2),
        "std_score": round(std_score, 2),
        "score_distribution": {str(k): counter.get(k, 0) for k in range(1, 6)},
        "score_pct": {str(k): round(counter.get(k, 0) / total * 100, 1) for k in range(1, 6)} if total else {},
    }

    print("\n" + "=" * 50)
    print(f"Group {group_label or '(未标注)'} | Cache: {Path(cache_path).name}")
    print("=" * 50)
    print(f"Total Scenes: {total}")
    print(f"  平均分 (1-5):    {mean_score:.2f} ± {std_score:.2f}")
    print(f"  分数分布:")
    for k in range(1, 6):
        n = counter.get(k, 0)
        pct = n / total * 100 if total else 0
        bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
        print(f"    {k} 分: {n:3d} ({pct:5.1f}%) {bar}")
    print(f"  对齐度得分 (均值): {mean_score:.2f}")
    print("=" * 50)

    return summary


def _normalize_judgments_to_scores(judgments: Dict) -> List[int]:
    """将 judgments 转为 1-5 分列表（兼容旧格式）。"""
    scores = []
    for v in judgments.values():
        if isinstance(v, int) and 1 <= v <= 5:
            scores.append(v)
        elif isinstance(v, str):
            scores.append({"Yes": 1, "No": 5, "Neutral": 3}.get(v, 3))
        else:
            scores.append(3)
    return scores


def compare_groups(result_a_path: str, result_b_path: str) -> None:
    """对比 Group A 与 Group B 的评分结果。"""
    with open(result_a_path, "r", encoding="utf-8") as f:
        data_a = json.load(f)
    with open(result_b_path, "r", encoding="utf-8") as f:
        data_b = json.load(f)

    judgments_a = data_a.get("judgments", {})
    judgments_b = data_b.get("judgments", {})

    scores_a = _normalize_judgments_to_scores(judgments_a)
    scores_b = _normalize_judgments_to_scores(judgments_b)

    def stats(scores: List[int]) -> Dict:
        t = len(scores)
        if not t:
            return {"mean": 0, "std": 0, "counter": Counter()}
        mean = sum(scores) / t
        var = sum((s - mean) ** 2 for s in scores) / t
        return {"mean": mean, "std": var ** 0.5, "counter": Counter(scores)}

    sa = stats(scores_a)
    sb = stats(scores_b)

    label_a = data_a.get("group_label", "A") or "A"
    label_b = data_b.get("group_label", "B") or "B"

    print("\n" + "=" * 65)
    print("Refinement 对齐度实验对比 (A: Refinement On vs B: Refinement Off)")
    print("评分标准: 5=完全一致, 1=相矛盾")
    print("=" * 65)
    print(f"{'指标':<22} | Group {label_a} (Refinement On) | Group {label_b} (Refinement Off)")
    print("-" * 65)
    print(f"{'总 Scene 数':<22} | {len(scores_a):>24} | {len(scores_b):>24}")
    print(f"{'平均分 (1-5)':<22} | {sa['mean']:>24.2f} | {sb['mean']:>24.2f}")
    print(f"{'标准差':<22} | {sa['std']:>24.2f} | {sb['std']:>24.2f}")
    print("-" * 65)
    print("分数分布:")
    for k in range(1, 6):
        ca = sa["counter"].get(k, 0)
        cb = sb["counter"].get(k, 0)
        pa = ca / len(scores_a) * 100 if scores_a else 0
        pb = cb / len(scores_b) * 100 if scores_b else 0
        print(f"  {k} 分: {ca:3d} ({pa:5.1f}%)  vs  {cb:3d} ({pb:5.1f}%)")
    print("-" * 65)
    diff = sa["mean"] - sb["mean"]
    print(f"{'对齐度得分 (均值)':<22} | {sa['mean']:>24.2f} | {sb['mean']:>24.2f}")
    print("-" * 65)
    if diff > 0:
        print(f"结论: Group A (Refinement On) 对齐度更高 (+{diff:.2f} 分)，Refinement 有效。")
    elif diff < 0:
        print(f"结论: Group B (Refinement Off) 对齐度更高 ({diff:+.2f} 分)，Refinement 可能带来负面影响。")
    else:
        print("结论: 两组对齐度相同，Refinement 无明显影响。")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(
        description="Refinement 对齐度实验：LLM-as-a-Judge 评估 Persona 与 Scene 一致性"
    )
    parser.add_argument("--cache", type=str, default=None, help="memory_cache_*.pkl 路径")
    parser.add_argument("--output", type=str, default=None, help="结果 JSON 路径（用于续传）")
    parser.add_argument("--group", type=str, default="", help="分组标签，如 A 或 B")
    parser.add_argument("--api-key", type=str, default=None, help="OpenAI API Key")
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("RESULT_A", "RESULT_B"),
        help="对比两组结果，如: --compare output/alignment_A.json output/alignment_B.json",
    )

    args = parser.parse_args()

    if args.compare:
        compare_groups(args.compare[0], args.compare[1])
        return

    cache = args.cache or "output/gpt/cached_memories/memory_cache_sample_0.pkl"
    output = args.output or f"output/alignment_{args.group or 'default'}.json"
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    if not os.path.exists(cache):
        print(f"Error: cache not found: {cache}")
        return

    run_alignment_test(
        cache_path=cache,
        output_path=output,
        group_label=args.group,
        api_key=args.api_key,
    )


if __name__ == "__main__":
    main()
