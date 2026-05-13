import pickle
import networkx as nx
import argparse
from collections import Counter

def debug_memory_cache(file_path):
    print(f"--- DEBUGGING Memory Cache: {file_path} ---\n")

    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            G = data.get('graph')
            
        if not G:
            print("Error: No graph found in pickle.")
            return

        print(f"Total Nodes: {len(G.nodes)}")
        print(f"Total Edges: {len(G.edges)}")

        nodes_without_level = 0
        
        for n, attrs in G.nodes(data=True):
            if 'level' in attrs:
                level_counts[attrs['level']] += 1
            else:
                nodes_without_level += 1
        
        print("\n=== Node Level Distribution ===")
        if not level_counts and nodes_without_level == 0:
            print("Graph is empty!")
        
        for level_name, count in level_counts.items():
            print(f"Level '{level_name}': {count} nodes")
            
        if nodes_without_level > 0:
            print(f"[WARNING] Nodes without 'level' attribute: {nodes_without_level}")

        for level_name in level_counts:
            print(f"\n--- Type: '{level_name}' ---")
            count = 0
            for n, attrs in G.nodes(data=True):
                if attrs.get('level') == level_name:
                    print(f"ID: {n}, Keys: {list(attrs.keys())}")
                    count += 1
                    if count >= 3: break

    except Exception as e:
        print(f"Debug failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", type=str)
    args = parser.parse_args()
    debug_memory_cache(args.file_path)
