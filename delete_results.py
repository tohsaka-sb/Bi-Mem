import json
import argparse
import os
import shutil
import sys

CATEGORY_MAP = {
    1: "Multi-Hop",
    2: "Temporal",
    3: "Open-Domain",
    4: "Single-Hop",
    5: "Adversarial"
}

def delete_batch_entries(file_path, target_samples=None, target_categories=None):
    """
    Delete QA results matching specific lists of sample_ids AND categories.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    if target_samples:
        target_samples = [str(s) for s in target_samples]

    print(f"--- Processing: {file_path} ---")
    
    if target_samples:
        print(f"Target Samples   : {target_samples}")
    else:
        print(f"Target Samples   : ALL Samples")

    if target_categories:
        cat_names = [f"{c}({CATEGORY_MAP.get(c, '?')})" for c in target_categories]
        print(f"Target Categories: {target_categories} -> {cat_names}")
    else:
        print(f"Target Categories: ALL Categories")


    if not target_samples and not target_categories:
        print("\n[Error] You must specify at least one --sample_id OR one --category.")
        print("To delete the entire file, please simply remove it using 'rm' command.")
        return

    backup_path = file_path + ".bak"
    shutil.copy2(file_path, backup_path)
    print(f"Backup created at: {backup_path}")

    temp_path = file_path + ".tmp"
    deleted_count = 0
    kept_count = 0

    try:
        with open(file_path, 'r', encoding='utf-8') as f_in, \
             open(temp_path, 'w', encoding='utf-8') as f_out:
            
            for line in f_in:
                line = line.strip()
                if not line:
                    continue

                should_delete = False
                try:
                    data = json.loads(line)
                    
                    if data.get("type") == "qa_result":
                        current_sample = str(data.get("sample_id"))
                        current_cat = int(data.get("category", -1))

                        match_sample = True
                        if target_samples:
                            match_sample = current_sample in target_samples
                        
                        match_category = True
                        if target_categories:
                            match_category = current_cat in target_categories

                        if match_sample and match_category:
                            should_delete = True

                except json.JSONDecodeError:
                    pass

                if should_delete:
                    deleted_count += 1
                else:
                    f_out.write(line + "\n")
                    kept_count += 1


        os.replace(temp_path, file_path)
        
        print("-" * 30)
        print(f"Done.")
        print(f"Deleted entries : {deleted_count}")
        print(f"Kept entries    : {kept_count}")
        
        if deleted_count == 0:
            print("\n[Warning] No matching entries were found. Nothing was deleted.")
        else:
            print(f"\n[Success] File updated.")

    except Exception as e:
        print(f"Error occurred: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch delete QA results from JSONL file.")
    
    parser.add_argument("file_path", type=str, help="Path to the .jsonl result file")
    
    parser.add_argument("--sample_id", type=str, nargs='+', default=None,
                        help="List of Sample IDs to delete (e.g. 0 1 2). If omitted, matches ALL samples.")
    
    parser.add_argument("--category", type=int, nargs='+', default=None,
                        help="List of Categories to delete (e.g. 1 3 5). If omitted, matches ALL categories.")

    args = parser.parse_args()
    
    delete_batch_entries(args.file_path, args.target_sample_id if hasattr(args, 'target_sample_id') else args.sample_id, args.category)