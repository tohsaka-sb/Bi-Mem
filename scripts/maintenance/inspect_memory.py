import pickle
import networkx as nx
import argparse
import os
import sys

def inspect_memory_cache(file_path, output_file=None):
    """
    Reads and displays the content of the A-Mem graph cache.
    Updated for 'fact' and 'persona' naming convention.
    """

    if output_file:
        f_out = open(output_file, 'w', encoding='utf-8')
    else:
        f_out = sys.stdout

    def log(message=""):
        print(message, file=f_out)

    log(f"--- Inspecting Memory Cache: {file_path} ---\n")

    if not os.path.exists(file_path):
        log(f"Error: File not found at {file_path}")
        if output_file: f_out.close()
        return

    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            
        if isinstance(data, dict) and 'graph' in data:
            G = data['graph']
            turns = data.get('turns_processed', 'Unknown')
            log(f"Status: {turns} turns processed.")
        else:
            log("Error: Unrecognized cache format. Expected dict with 'graph' key.")
            if output_file: f_out.close()
            return

    except Exception as e:
        log(f"Error loading pickle file: {e}")
        if output_file: f_out.close()
        return

    nodes = G.nodes(data=True)

    persona_nodes = []
    scene_nodes = []
    fact_nodes = []
    other_nodes = []

    for node_id, attrs in nodes:
        level = attrs.get('level')
        if level == 'persona':
            persona_nodes.append((node_id, attrs))
        elif level == 'scene':
            scene_nodes.append((node_id, attrs))
        elif level == 'fact':
            fact_nodes.append((node_id, attrs))
        else:
            other_nodes.append((node_id, attrs))

    log(f"\nTotal Nodes: {len(G.nodes)}")
    log(f"  - Level 2 (Persona): {len(persona_nodes)}")
    log(f"  - Level 1 (Scene)  : {len(scene_nodes)}")
    log(f"  - Level 0 (Fact)   : {len(fact_nodes)}")
    log(f"  - Unknown/Other    : {len(other_nodes)}")
    log("-" * 50)

    log(f"\n=== Level 2: Persona Memories ({len(persona_nodes)}) ===")
    for i, (nid, attrs) in enumerate(persona_nodes, 1):
        log(f"\n[Node {i}] ID: {nid}")
        log(f"  Time: {attrs.get('timestamp')}")
        log(f"  Content (Profile): \n{attrs.get('content')}")
        log(f"  Keywords: {attrs.get('keywords')}")
        log(f"  Tags: {attrs.get('tags')}")

    log(f"\n\n=== Level 1: Scene Memories ({len(scene_nodes)}) ===")
    for i, (nid, attrs) in enumerate(scene_nodes, 1):

        children = [n for n in G.successors(nid) if G.nodes[n].get('level') == 'fact']
        log(f"\n[Scene {i}] ID: {nid}")
        log(f"  Summary: {attrs.get('content')}")
        log(f"  Contains {len(children)} facts.")

    log(f"\n\n=== Level 0: Fact Memories ({len(fact_nodes)}) ===")
    try:
        fact_nodes.sort(key=lambda x: x[1].get('timestamp', ''))
    except:
        pass 

    for i, (nid, attrs) in enumerate(fact_nodes, 1):
        log(f"\n[Fact {i}] ID: {nid}")
        log(f"  Time: {attrs.get('timestamp')}")
        log(f"  Content: {attrs.get('content')}")
        log(f"  Context: {attrs.get('context')}")
        log(f"  Keywords: {attrs.get('keywords')}")

    log("\n\n=== Inspection Complete ===")
    
    if output_file:
        print(f"Inspection results saved to {output_file}")
        f_out.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect A-Mem Graph Cache")
    parser.add_argument("file_path", type=str, help="Path to the .pkl file")
    parser.add_argument("--output", type=str, default=None, help="Path to save the output text file")
    
    args = parser.parse_args()
    inspect_memory_cache(args.file_path, args.output)
