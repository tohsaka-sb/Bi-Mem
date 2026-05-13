import pickle
import json
import argparse
import os
import re
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter
from tqdm import tqdm

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("请安装 openai: pip install openai")

# Refined scene 格式: original_summary\n[Insight]: addition
INSIGHT_MARKER = "\n[Insight]:"


def load_memory_cache(cache_path: str) -> Tuple[str, List[Tuple[str, str, str, str]]]:
    """
    从 pkl 加载记忆，提取 Level 2 Persona 和 Level 1 Scene 节点。
    对含 [Insight] 的 scene 解析出 original / modification / refined。
    Returns: (persona_text, [(scene_id, original_summary, modification, refined_summary), ...])
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
            if INSIGHT_MARKER in content:
                parts = content.split(INSIGHT_MARKER, 1)
                original_summary = parts[0].strip()
                modification = parts[1].strip() if len(parts) > 1 else ""
                scene_list.append((node_id, original_summary, modification, content))
            # 不含 [Insight] 的 scene 跳过（未发生校准更新）

    persona_text = "\n\n".join(persona_parts) if persona_parts else "(No persona nodes)"
    return persona_text, scene_list


def call_gpt4o_quality_judge(
    persona: str,
    memory_text: str,
    client: OpenAI,
) -> int:
    """
    调用 GPT-4o 评估单个记忆与 Persona 的对齐度/质量。
    Returns: 1到5的评分（或者0-100）。我们这里采用 1-5分制（5分为极佳）。
    """
    prompt = f"""You are an expert evaluator for a memory system.
Your task is to evaluate how well a "Scene Memory" aligns with and reflects the "User Persona", or how rich and useful it is in the context of the user.

[User Persona]
{persona}

[Scene Memory]
{memory_text}

Please evaluate the quality and alignment of the Scene Memory on a scale of 1 to 5.
- 1 (Poor): Contradicts the persona or is completely irrelevant/useless.
- 2 (Fair): Barely relates to the persona or lacks important details.
- 3 (Good): Adequately reflects the persona and contains useful information.
- 4 (Very Good): Strong connection to the persona with rich contextual details.
- 5 (Excellent): Perfectly aligns with the persona, highlighting key traits with excellent detail.

Output JSON only: {{"score": int}}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You respond with a JSON object only: {\"score\": int}. The score must be an integer between 1 and 5."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=20,
        )
        raw = (response.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"\n[WARN] API error: {e}")
        return 3  # default middle score on error

    return _parse_quality_score(raw)


def _parse_quality_score(raw: str) -> int:
    """从模型输出解析 1-5 分。"""
    m = re.search(r'"score"\s*:\s*([1-5])', raw)
    if m:
        return int(m.group(1))
    m = re.search(r"[1-5]", raw)
    if m:
        return int(m.group(0))
    return 3


def run_calibration_quality_eval(
    cache_paths: List[str],
    output_path: str,
    api_key: Optional[str] = None,
) -> Dict:
    """
    对 Refinement On 的多份 cache 运行校准质量评估。
    仅评估含 [Insight] 的 scene（即发生了校准更新的 scene）。
    """
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("请设置 OPENAI_API_KEY 环境变量或传入 --api-key")

    client = OpenAI(api_key=api_key, base_url=os.getenv("OPENAI_API_BASE"))

    all_scene_items = {} # scene_id -> (orig, mod, refined, persona_text)
    
    print(f"正在加载 {len(cache_paths)} 个 cache 文件...")
    for cache_path in cache_paths:
        if not os.path.exists(cache_path):
            print(f"[WARN] cache 未找到跳过: {cache_path}")
            continue
        try:
            persona_text, scene_list = load_memory_cache(cache_path)
            for s in scene_list:
                scene_id = s[0]
                # s = (scene_id, original_summary, modification, content)
                all_scene_items[scene_id] = (s[1], s[2], s[3], persona_text)
        except Exception as e:
            print(f"[WARN] 加载 {cache_path} 失败: {e}")

    if not all_scene_items:
        print("未在指定的 cache 中找到含 [Insight] 的 Scene（无校准更新可评估）。请确保使用 Refinement On 生成的 cache。")
        return {}

    judgments = {}
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                saved = json.load(f)
            raw_judgments = saved.get("judgments", {})
            dropped_count = 0
            for k, v in raw_judgments.items():
                if isinstance(v, dict) and "original_score" in v and "refined_score" in v:
                    # 如果分数为3，说明极有可能是API断联返回的默认错误分数，丢弃以进行重测
                    if v["original_score"] == 3 and v["refined_score"] == 3:
                        dropped_count += 1
                        continue
                    judgments[k] = v
            print(f"[续传] 已加载 {len(judgments)} 条已有判断 (剔除了 {dropped_count} 条默认的 (3, 3) 评分进行重测)")
        except Exception as e:
            print(f"[WARN] 加载已有结果失败: {e}")

    to_judge = [
        sid for sid in all_scene_items if sid not in judgments
    ]

    if not to_judge:
        print("所有 Scene 已评估完毕，跳过 API 调用。")
    else:
        print(f"待评估: {len(to_judge)} 个校准更新")
        for scene_id in tqdm(to_judge, desc="LLM Judge (Calibration Quality)"):
            orig, mod, refined, persona_text = all_scene_items[scene_id]
            
            score_original = call_gpt4o_quality_judge(persona_text, orig, client)
            score_refined = call_gpt4o_quality_judge(persona_text, refined, client)
            
            judgments[scene_id] = {
                "original_score": score_original,
                "refined_score": score_refined,
                "diff": score_refined - score_original
            }

            # 增量保存
            result = {
                "cache_paths": cache_paths,
                "judgments": judgments,
            }
            os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

    original_scores = [j["original_score"] for j in judgments.values()]
    refined_scores = [j["refined_score"] for j in judgments.values()]
    diffs = [j["diff"] for j in judgments.values()]
    
    total = len(judgments)
    
    def get_stats(scores_list):
        if not scores_list: return 0, 0
        mean = sum(scores_list) / len(scores_list)
        variance = sum((s - mean) ** 2 for s in scores_list) / len(scores_list)
        return mean, variance ** 0.5
        
    orig_mean, orig_std = get_stats(original_scores)
    ref_mean, ref_std = get_stats(refined_scores)
    diff_mean, diff_std = get_stats(diffs)

    summary = {
        "cache_paths": cache_paths,
        "total_refined_scenes": total,
        "original_mean_score": round(orig_mean, 2),
        "refined_mean_score": round(ref_mean, 2),
        "diff_mean": round(diff_mean, 2),
    }

    print("\n" + "=" * 55)
    print("Refinement 记忆前后对比评估 (1-5分制)")
    print("=" * 55)
    print(f"处理的 Cache 数量: {len(cache_paths)}")
    print(f"对比的 Scene 总数: {total}")
    print(f"  Refine 前 (Original) 平均分: {orig_mean:.2f} ± {orig_std:.2f}")
    print(f"  Refine 后 (Refined) 平均分 : {ref_mean:.2f} ± {ref_std:.2f}")
    print(f"  平均提升 (Diff)          : {diff_mean:.2f} ± {diff_std:.2f}")
    print("-" * 55)
    
    improved = sum(1 for d in diffs if d > 0)
    worsened = sum(1 for d in diffs if d < 0)
    unchanged = sum(1 for d in diffs if d == 0)
    
    print(f"  变好 (Improved)  : {improved} ({(improved/total*100) if total else 0:.1f}%)")
    print(f"  变差 (Worsened)  : {worsened} ({(worsened/total*100) if total else 0:.1f}%)")
    print(f"  不变 (Unchanged) : {unchanged} ({(unchanged/total*100) if total else 0:.1f}%)")
    print("=" * 55)

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Refinement 校准效果评估：LLM-as-a-Judge 评估每次校准更新的质量 (0=Bad, 1=Redundant, 2=Good, 3=Excellent)"
    )
    parser.add_argument("--cache", nargs='+', default=["output/gpt/cached_memories/memory_cache_sample_*.pkl"], help="Refinement On 的 memory_cache_*.pkl 路径，支持通配符或多个文件")
    parser.add_argument("--output", type=str, default="output/calibration_quality.json", help="结果 JSON 路径（用于续传）")
    parser.add_argument("--api-key", type=str, default=None, help="OpenAI API Key")

    args = parser.parse_args()

    all_caches = []
    for pattern in args.cache:
        matched = glob.glob(pattern)
        if matched:
            all_caches.extend(matched)
        else:
            if os.path.exists(pattern):
                all_caches.append(pattern)

    all_caches = sorted(list(set(all_caches)))

    if not all_caches:
        print("未找到指定的 cache 文件，请检查路径是否正确。")
        return

    run_calibration_quality_eval(
        cache_paths=all_caches,
        output_path=args.output,
        api_key=args.api_key,
    )


if __name__ == "__main__":
    main()
