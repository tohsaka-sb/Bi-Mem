import json
import argparse
import csv
import os
import sys

CATEGORY_MAP = {
    1: "Multi-Hop",
    2: "Temporal",
    3: "Open-Domain",
    4: "Single-Hop",
    5: "Adversarial"
}

def flatten_data(data_item):
    """
    将嵌套的 metrics 字典展平，以便写入 CSV。
    """
    flat_item = {
        "sample_id": data_item.get("sample_id"),
        "question_id": data_item.get("question_id") or data_item.get("question_id_in_sample"),
        "category": data_item.get("category"),
        "category_name": CATEGORY_MAP.get(int(data_item.get("category", -1)), "Unknown"),
        "question": data_item.get("question"),
        "reference": data_item.get("reference"),
        "prediction": data_item.get("prediction"),
    }

    metrics = data_item.get("metrics", {})
    for key, value in metrics.items():
        flat_item[key] = value

    return flat_item

def extract_data(input_file, output_file, target_category=None, output_format="csv"):
    print(f"--- Extracting data from: {input_file} ---")
    
    if target_category is not None:
        print(f"Target Category: {target_category} ({CATEGORY_MAP.get(target_category, 'Unknown')})")
    else:
        print("Extracting ALL categories.")

    extracted_items = []
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                
                try:
                    data = json.loads(line)
                    
                    if data.get("type") != "qa_result":
                        continue

                    current_cat = int(data.get("category", -1))
                    
                    if target_category is None or current_cat == target_category:
                        extracted_items.append(data)
                        
                except json.JSONDecodeError:
                    continue
                    
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}")
        return

    if not extracted_items:
        print("No matching records found.")
        return

    print(f"Found {len(extracted_items)} matching records.")
    
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    if output_format == "csv":

        flat_items = [flatten_data(item) for item in extracted_items]
        
        if flat_items:
            fieldnames = list(flat_items[0].keys())
            
            with open(output_file, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(flat_items)
            print(f"Successfully saved to CSV: {output_file}")
            
    elif output_format == "jsonl":
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in extracted_items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Successfully saved to JSONL: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract specific category results from JSONL.")
    
    parser.add_argument("input_file", type=str, help="Path to the input .jsonl file")
    parser.add_argument("output_file", type=str, help="Path to save the extracted file (e.g., analysis.csv)")
    
    parser.add_argument("--category", type=int, default=None, 
                        help="Category ID to extract (1=Multi, 2=Temp, 3=Open, 4=Single, 5=Adv). If not set, extracts all.")
    
    parser.add_argument("--format", type=str, choices=["csv", "jsonl"], default="csv",
                        help="Output format (default: csv)")

    args = parser.parse_args()
    
    extract_data(args.input_file, args.output_file, args.category, args.format)