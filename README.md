Bi-Mem: Bidirectional Construction of Hierarchical Memory for Personalized LLMs via Inductive-Reflective Agents
Bi-Mem  is an advanced, hierarchy memory framework designed to enhance Large Language Models (LLMs) with long-term, structured, and evolving memory capabilities. It moves beyond simple vector retrieval by constructing a hierarchical knowledge graph that mimics human cognitive processes: from specific facts, to clustered scenes, up to abstract persona profiles.

🌟 Core Features
Hierarchical Memory Graph: Organizes information into three distinct levels:
Level 0 (fact): Atomized conversation turns with rich metadata (keywords, context, tags).
Level 1 (Scene): Clusters of facts summarized into coherent narrative scenes.
Level 2 (persona): High-level user profiles extracted from scenes (Interests, Personality, Values, etc.).
Multi-Agent Architecture: Specialized agents (SceneManager, personaManager) manage the construction and maintenance of different memory levels.
Top-Down Refinement: A feedback loop where high-level persona insights are used to verify and enrich lower-level scene memories.
Hybrid Retrieval: Combines Dense (Vector) and Sparse (BM25) retrieval to balance persona understanding with precise entity matching.
Graph Traversal Retrieval: A novel retrieval strategy that dynamically expands context from seed nodes to their neighbors and parents/children in the graph.
Robustness:
End-to-end Checkpointing & Resume capability.
API Rate Limit Handling with exponential backoff.
Streaming Output (JSONL) for real-time monitoring.
🛠️ Installation
Prerequisites
Python 3.10+
NVIDIA GPU (Recommended for local models and embedding calculation)

🚀 Usage
1. Configuration
Set up your API keys. Bi-Mem supports OpenAI and compatible APIs (e.g., DeepSeek, Ollama).
run in your terminal first:
pip install -r requirements.txt

export OPENAI_API_KEY="your-api-key"

2. Running the Evaluation
Use test_advanced.py to run the full pipeline (Memory Construction -> Hierarchy Building -> QA Evaluation).
Example: Run on 10% of data (Sample 0) using GPT-4o-mini:


python test_advanced.py \
    --model gpt-4o-mini \
    --backend openai \
    --ratio 0.1 \
    --output output/results.jsonl \
    --retrieval-strategy graph_traversal \
    --enable-refinement \
    2>&1 | tee output/run.log
Key Arguments:
--ratio: Portion of dataset to use (0.1 = Sample 0, 1.0 = All).
--enable-refinement: Enable the Top-Down Refinement phase (Recommended).
--retrieval-strategy: graph_traversal (Best), flat, top_down, bottom_up.
--retrieve_k: Base number of nodes to retrieve.
3. Analyzing Results
Use the provided script to generate a statistical report from the output JSONL file.


python summarize_results.py output/results.jsonl
4. Utilities
extract_results.py
clean_hierarchy.py: Delete Level 1/2 nodes to force hierarchy reconstruction.
inspect_memory.py: View the content of the constructed memory graph.


python inspect_memory.py path/to/memory_cache.pkl
📂 Project Structure
memory_layer.py: Core logic for AgenticMemorySystem, Graph operations, and Retrieval.
memory_managers.py: SceneManager and personaManager classes (Prompts & LLM interactions).
test_advanced.py: Main execution script, QA logic, and Evaluation loop.
load_dataset.py: Data loader for LoCoMo dataset.
utils.py: Metric calculation (F1, BLEU, BERTScore).
📊 Performance (Sample)
On LoCoMo dataset (Sample 0), Bi-Mem achieves competitive performance compared to baselines:
Metric	Bi-Mem (Best Config)
Overall F1	0.49 (SOTA)
Temporal F1	0.54 (SOTA)
Single-Hop F1	0.53 (SOTA)
📜 Citation
If you use this  for your research, please cite our work .


