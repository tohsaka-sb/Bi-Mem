# python summarize_results.py output/gpt/g_results.jsonl 2>&1 | tee output/gpt/g_results.log
# python summarize_results.py output/gpt/g_results.jsonl --sample_id 0 2>&1 | tee output/gpt/g_results.log

# python summarize_results.py g_results.jsonl --sample_id 3 2>&1 | tee output/gpt/g_results_1.log 


# python summarize_results.py output/gpt/f_results.jsonl 2>&1 | tee output/gpt/f_results.log
# python summarize_results.py output/gpt/top_down_results.jsonl 2>&1 | tee output/gpt/top_down_results.log
# python summarize_results.py output/gpt/bottom_up_results.jsonl 2>&1 | tee output/gpt/bottom_up_results.log
# python summarize_results.py output/qwen/qwen_results.jsonl 2>&1 | tee output/qwen/qwen_results.log

# python summarize_results.py output/gpt/event_only_results.jsonl 2>&1 | tee output/gpt/event_only_results.log
# python summarize_results.py output/gpt/event_scene_results.jsonl 2>&1 | tee output/gpt/event_scene_results.log
# python summarize_results.py output/gpt/event_to_scene_results.jsonl 2>&1 | tee output/gpt/event_to_scene_results.log
# python summarize_results.py output/gpt/event_semantic_results.jsonl 2>&1 | tee output/gpt/event_semantic_results.log

import json
import argparse
from collections import defaultdict
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import aggregate_metrics

CATEGORY_MAP = {
    1: "Multi-Hop",
    2: "Temporal",
    3: "Open-Domain",
    4: "Single-Hop",
    5: "Adversarial"
}

def analyze_jsonl_results(file_path: str, target_sample_id: str = None):
    """
    Reads a JSONL result file and calculates aggregate metrics.
    Features:
    1. EXCLUDES Category 5 (Adversarial) by default (can be toggled in code if needed).
    2. DE-DUPLICATES results: if a question appears multiple times (e.g. re-run), 
       only the LAST entry in the file is used.
    """
    # Key: (sample_id, question_text) -> Value: data_dict
    # Using question text as unique key per sample to be safe
    unique_results = {}
    
    metadata = {}
    run_summaries = []
    
    skipped_adversarial_count = 0

    print(f"--- Analyzing results from: {file_path} ---")
    if target_sample_id is not None:
        print(f"--- Filtering for Sample ID: {target_sample_id} ---")
    print("--- Note: Category 5 (Adversarial) results will be IGNORED ---")
    print("--- Note: Duplicate questions will be de-duplicated (Last-Win strategy) ---")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    entry_type = data.get("type")

                    if entry_type == "metadata":
                        metadata = data
                    
                    elif entry_type == "qa_result":
                        # 1. Sample ID Filter
                        current_sample_id = str(data.get("sample_id"))
                        if target_sample_id is not None and current_sample_id != str(target_sample_id):
                            continue

                        # 2. Category Filter (Skip Adversarial)
                        category = int(data.get("category", -1))
                        if category == 5:
                            skipped_adversarial_count += 1
                            continue

                        # 3. Store for De-duplication
                        if "metrics" in data:
                            question_text = data.get("question", "").strip()
                            # Use a unique key for this question instance
                            key = (current_sample_id, question_text)
                            unique_results[key] = data
                    
                    elif entry_type == "run_summary":
                        run_summaries.append(data)

                except json.JSONDecodeError:
                    continue
    
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return

    # --- Convert unique results back to lists for calculation ---
    all_metrics = []
    all_categories = []
    
    for data in unique_results.values():
        all_metrics.append(data["metrics"])
        all_categories.append(int(data["category"]))

    if not all_metrics:
        print(f"No valid non-adversarial entries found.")
        return

    print(f"\nFound {len(all_metrics)} unique valid questions (Skipped {skipped_adversarial_count} adversarial records).")

    # --- Core Logic: Calculate Stats ---
    aggregate_results = aggregate_metrics(all_metrics, all_categories)

    # --- Display the results ---
    print("\n--- Aggregate Metrics Summary (Excluding Adversarial) ---")
    
    total_questions_analyzed = len(all_metrics)
    print(f"Total questions analyzed: {total_questions_analyzed}")

    # Create category distribution
    category_counts = defaultdict(int)
    for category in all_categories:
        category_counts[category] += 1
        
    print("\nCategory Distribution:")
    for category in sorted(category_counts.keys()):
        count = category_counts[category]
        percentage = (count / total_questions_analyzed) * 100
        cat_name = CATEGORY_MAP.get(category, "Unknown")
        print(f"Category {category} ({cat_name}): {count} questions ({percentage:.1f}%)")
    
    print("\nAggregate Metrics:")
    
    # 1. Print Overall
    if "overall" in aggregate_results:
        print(f"\nOverall:")
        for metric_name, stats in aggregate_results["overall"].items():
            print(f"  {metric_name:<18} | Mean: {stats['mean']:.4f}")

    # 2. Print Categories Sorted by ID
    cat_keys = [k for k in aggregate_results.keys() if k.startswith("category_")]
    cat_keys.sort(key=lambda x: int(x.split('_')[1]))

    for split_name in cat_keys:
        cat_id = int(split_name.split('_')[1])
        if cat_id == 5: continue
        
        cat_name = CATEGORY_MAP.get(cat_id, "Unknown")
        
        print(f"\n{split_name} ({cat_name}):")
        for metric_name, stats in aggregate_results[split_name].items():
            print(f"  {metric_name:<18} | Mean: {stats['mean']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze results excluding Adversarial questions with De-duplication.")
    parser.add_argument("file_path", type=str, help="Path to the .jsonl result file.")
    parser.add_argument("--sample_id", type=str, default=None, help="Filter results by specific Sample ID")
    
    args = parser.parse_args()
    
    analyze_jsonl_results(args.file_path, args.sample_id)