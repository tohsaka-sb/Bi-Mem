from ast import Str
from typing import List, Dict, Optional, Literal, Any, Union
import json
from datetime import datetime
import uuid
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from abc import ABC, abstractmethod
from transformers import AutoModel, AutoTokenizer,AutoModelForCausalLM, AutoTokenizer, pipeline
from nltk.tokenize import word_tokenize
import pickle
from pathlib import Path
from litellm import completion
import time
import torch
import json 
import re
import requests
import networkx as nx
from tqdm import tqdm
from sentence_transformers import util # Need util for cos_sim
from memory_managers import SceneManager, PersonaManager
from rank_bm25 import BM25Okapi
import numpy as np



def simple_tokenize(text):
    return word_tokenize(text)

class BaseLLMController(ABC):
    @abstractmethod
    def get_completion(self, prompt: str) -> str:
        """Get completion from LLM"""
        pass

class OpenAIController(BaseLLMController):
    def __init__(self, model: str = "gpt-4", api_key: Optional[str] = None):
        try:
            from openai import OpenAI
            self.model = model
            if api_key is None:
                api_key = os.getenv('OPENAI_API_KEY')
            if api_key is None:
                raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
            self.client = OpenAI(
                api_key=api_key or os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_API_BASE")
            )
        except ImportError:
            raise ImportError("OpenAI package not found. Install it with: pip install openai")
    
    def get_completion(self, prompt: str, response_format: dict, temperature: float = 0.7) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You must respond with a JSON object."},
                {"role": "user", "content": prompt}
            ],
            response_format=response_format,
            temperature=temperature,
            max_tokens=1000
        )
        return response.choices[0].message.content

    ####
class ProbexController(BaseLLMController):
    """
    Controller for the api.probex.top API endpoint.
    """
    def __init__(self, model: str = "deepseek-chat", api_key: Optional[str] = None):
        self.model = model
        self.api_url = "https://api.probex.top/v1/chat/completions"
        
        if api_key is None:
            api_key = os.getenv('PROBEX_API_KEY')
        
        if not api_key:
            raise ValueError("Probex API key not found. Provide it as an argument or set the PROBEX_API_KEY environment variable.")
            
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def _extract_json_from_response(self, text: str) -> str:
        """
        Extracts a JSON string from a text that might contain markdown code blocks.
        """
        match = re.search(r"```json\s*(\{.*\})\s*```", text, re.DOTALL)
        if match:
            return match.group(1)
        match = re.search(r"```\s*(\{.*\})\s*```", text, re.DOTALL)
        if match:
            return match.group(1)
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and start < end:
            return text[start:end+1]
        return text


    def get_completion(self, prompt: str, response_format: dict = None, temperature: float = 0.7) -> str:
        """
        Get completion from the Probex API with retry and exponential backoff logic.
        """
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You must respond with a valid JSON object."},
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "temperature": temperature
        }

        # --- Start of Retry Logic ---
        max_retries = 3
        base_delay = 2  # seconds

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=data,
                    timeout=300  # 5-minute timeout for the request itself
                )
                
                # Raise an exception for bad status codes (4xx client error or 5xx server error)
                response.raise_for_status()
                
                # If successful, process and return the response
                result = response.json()
                raw_content = result['choices'][0]['message']['content']
                clean_json_string = self._extract_json_from_response(raw_content)
                return clean_json_string

            except requests.exceptions.RequestException as e:
                # This catches connection errors, timeouts, proxy errors, etc.
                print(f"Probex API request failed on attempt {attempt + 1}/{max_retries}. Error: {e}")
                
                # If this was the last attempt, give up.
                if attempt == max_retries - 1:
                    print("All retry attempts failed for Probex API.")
                    return "{}"

                # Exponential backoff: wait for base_delay * 2^attempt seconds
                delay = base_delay * (2 ** attempt)
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
        
        # This part should ideally not be reached, but as a fallback:
        return "{}"
    ####

    ####

class OllamaController(BaseLLMController):
    def __init__(self, model: str = "llama2"):
        from ollama import chat
        self.model = model
    
    def _generate_empty_value(self, schema_type: str, schema_items: dict = None) -> Any:
        if schema_type == "array":
            return []
        elif schema_type == "string":
            return ""
        elif schema_type == "object":
            return {}
        elif schema_type == "number":
            return 0
        elif schema_type == "boolean":
            return False
        return None

    def _generate_empty_response(self, response_format: dict) -> dict:
        if "json_schema" not in response_format:
            return {}
            
        schema = response_format["json_schema"]["schema"]
        result = {}
        
        if "properties" in schema:
            for prop_name, prop_schema in schema["properties"].items():
                result[prop_name] = self._generate_empty_value(prop_schema["type"], 
                                                            prop_schema.get("items"))
        
        return result

    def get_completion(self, prompt: str, response_format: dict, temperature: float = 0.7) -> str:
        try:
            response = completion(
                model="ollama_chat/{}".format(self.model),
                messages=[
                    {"role": "system", "content": "You must respond with a JSON object."},
                    {"role": "user", "content": prompt}
                ],
                response_format=response_format,
            )
            return response.choices[0].message.content
        except Exception as e:
            empty_response = self._generate_empty_response(response_format)
            return json.dumps(empty_response)


class LLMController:
    """LLM-based controller for memory metadata generation"""

    def __init__(self, 
                 backend: Literal["openai", "ollama", "probex"] = "openai",
                 model: str = "gpt-4", 
                 api_key: Optional[str] = None):
        if backend == "openai":
            self.llm = OpenAIController(model, api_key)
        elif backend == "ollama":
            self.llm = OllamaController(model)
        elif backend == "probex":
            self.llm = ProbexController(model, api_key)
        else:
            raise ValueError("Backend must be 'openai', 'ollama', or 'probex'")
    ####

class MemoryNote:
    """
    Represents the data attributes of a node in the memory graph.
    This is now a simpler data container.
    """
    def __init__(self, 
                 content: str,
                 level: str, # 'fact', 'scene', or 'persona'
                 timestamp: str,
                 keywords: Optional[List[str]] = None,
                 context: Optional[str] = None, 
                 tags: Optional[List[str]] = None,
                 **kwargs): # Allows for additional attributes
        
        self.content = content
        self.level = level
        self.timestamp = timestamp
        self.keywords = keywords or []
        self.context = context or ""
        self.tags = tags or []
        # Store any other dynamic attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self):
        """Converts the note's data to a dictionary for graph storage."""
        return self.__dict__
    ####

class HybridRetriever:
    """
    Hybrid retrieval system combining BM25 (Sparse) and Persona (Dense) search.
    Designed for lazy-loading BM25 index to handle incremental updates efficiently.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', alpha: float = 0.5):
        """
        Args:
            model_name: Name of the SentenceTransformer model.
            alpha: Weight for Dense scores (0.0 to 1.0). 
                   Final Score = alpha * Dense + (1 - alpha) * BM25.
                   Default 0.5 implies equal weight.
        """
        self.model = SentenceTransformer(model_name)
        self.alpha = alpha
        
        self.corpus = [] # List of document strings
        self.embeddings = None # Numpy array of embeddings
        
        self.bm25 = None
        self.is_bm25_dirty = True # Flag to trigger rebuild
        
    def add_documents(self, documents: List[str]):
        """Adds documents to the retriever."""
        if not documents:
            return
            
        # 1. Update Corpus (for BM25)
        self.corpus.extend(documents)
        self.is_bm25_dirty = True
        
        # 2. Update Embeddings (for Dense)
        new_embeddings = self.model.encode(documents)
        
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
            
    def _build_bm25(self):
        """Internal method to build/rebuild BM25 index."""
        if not self.corpus:
            return
        
        print("Building BM25 index...")
        # Simple tokenization for BM25
        tokenized_corpus = [simple_tokenize(doc) for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.is_bm25_dirty = False
        print("BM25 index built.")

    def search(self, query: str, k: int = 5) -> List[int]:
        """
        Hybrid search returning indices of top-k documents.
        """
        if not self.corpus:
            return []
            
        # Ensure BM25 is up to date
        if self.is_bm25_dirty or self.bm25 is None:
            self._build_bm25()
            
        n_docs = len(self.corpus)
        
        # --- 1. Dense Search ---
        query_emb = self.model.encode(query)
        # Cosine similarity returns [-1, 1]
        dense_scores = util.cos_sim(query_emb, self.embeddings)[0].cpu().numpy()
        
        # Normalize Dense to [0, 1] for better combination (approximate)
        # Cosine is usually [0, 1] for text but can be negative. 
        # Simple clipping or 0-1 scaling is good practice.
        dense_scores = (dense_scores + 1) / 2 
        
        # --- 2. Sparse Search (BM25) ---
        tokenized_query = simple_tokenize(query)
        bm25_scores = np.array(self.bm25.get_scores(tokenized_query))
        
        # Normalize BM25 to [0, 1]
        # BM25 scores are unbounded positive numbers. Min-Max scaling is needed.
        if len(bm25_scores) > 0:
            min_score = np.min(bm25_scores)
            max_score = np.max(bm25_scores)
            if max_score - min_score > 0:
                bm25_scores = (bm25_scores - min_score) / (max_score - min_score)
            else:
                bm25_scores = np.zeros_like(bm25_scores) # All scores same
        
        # --- 3. Hybrid Combination ---
        # Weighted sum
        hybrid_scores = (self.alpha * dense_scores) + ((1 - self.alpha) * bm25_scores)
        
        # --- 4. Top-K ---
        # Get indices of top k scores
        # argsort sorts ascending, so we take last k and reverse
        k = min(k, n_docs)
        top_k_indices = np.argsort(hybrid_scores)[-k:][::-1]
        
        return top_k_indices.tolist()

    def save(self, retriever_cache_file: str, retriever_cache_embeddings_file: str):
        """Save retriever state to disk."""
        # Save embeddings
        if self.embeddings is not None:
            np.save(retriever_cache_embeddings_file, self.embeddings)
        
        # Save corpus (needed for BM25 rebuild)
        # Note: We don't save the BM25 object itself as it's large and can be rebuilt.
        state = {
            'corpus': self.corpus
        }
        with open(retriever_cache_file, 'wb') as f:
            pickle.dump(state, f)
            
    def load(self, retriever_cache_file: str, retriever_cache_embeddings_file: str):
        """Load retriever state from disk."""
        print(f"Loading retriever from {retriever_cache_file}...")
        
        if os.path.exists(retriever_cache_embeddings_file):
            self.embeddings = np.load(retriever_cache_embeddings_file)
            
        if os.path.exists(retriever_cache_file):
            with open(retriever_cache_file, 'rb') as f:
                state = pickle.load(f)
                self.corpus = state.get('corpus', [])
                self.is_bm25_dirty = True # Mark dirty to rebuild BM25 on first search
        
        return self

####

class SimpleEmbeddingRetriever:
    """Simple retrieval system using only text embeddings."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize the simple embedding retriever.
        
        Args:
            model_name: Name of the SentenceTransformer model to use
        """
        self.model = SentenceTransformer(model_name)
        self.corpus = []
        self.embeddings = None
        self.document_ids = {}  # Map document content to its index
        
    def add_documents(self, documents: List[str]):
        """Add documents to the retriever."""
        # Reset if no existing documents
        if not self.corpus:
            self.corpus = documents
            # print("documents", documents, len(documents))
            self.embeddings = self.model.encode(documents)
            self.document_ids = {doc: idx for idx, doc in enumerate(documents)}
        else:
            # Append new documents
            start_idx = len(self.corpus)
            self.corpus.extend(documents)
            new_embeddings = self.model.encode(documents)
            if self.embeddings is None:
                self.embeddings = new_embeddings
            else:
                self.embeddings = np.vstack([self.embeddings, new_embeddings])
            for idx, doc in enumerate(documents):
                self.document_ids[doc] = start_idx + idx
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, float]]:
        """Search for similar documents using cosine similarity.
        
        Args:
            query: Query text
            k: Number of results to return
            
        Returns:
            List of dicts with document text and score
        """
        if not self.corpus:
            return []
        # print("corpus", len(self.corpus), self.corpus)
        # Encode query
        query_embedding = self.model.encode([query])[0]
        
        # Calculate cosine similarities
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        # Get top k results
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
            
        return top_k_indices
        
    def save(self, retriever_cache_file: str, retriever_cache_embeddings_file: str):
        """Save retriever state to disk"""
        # Save embeddings using numpy
        if self.embeddings is not None:
            np.save(retriever_cache_embeddings_file, self.embeddings)
        
        # Save other attributes
        state = {
            'corpus': self.corpus,
            'document_ids': self.document_ids
        }
        with open(retriever_cache_file, 'wb') as f:
            pickle.dump(state, f)
    
    def load(self, retriever_cache_file: str, retriever_cache_embeddings_file: str):
        """Load retriever state from disk"""
        print(f"Loading retriever from {retriever_cache_file} and {retriever_cache_embeddings_file}")
        
        # Load embeddings
        if os.path.exists(retriever_cache_embeddings_file):
            print(f"Loading embeddings from {retriever_cache_embeddings_file}")
            self.embeddings = np.load(retriever_cache_embeddings_file)
            print(f"Embeddings shape: {self.embeddings.shape}")
        else:
            print(f"Embeddings file not found: {retriever_cache_embeddings_file}")
        
        # Load other attributes
        if os.path.exists(retriever_cache_file):
            print(f"Loading corpus from {retriever_cache_file}")
            with open(retriever_cache_file, 'rb') as f:
                state = pickle.load(f)
                self.corpus = state['corpus']
                self.document_ids = state['document_ids']
                print(f"Loaded corpus with {len(self.corpus)} documents")
        else:
            print(f"Corpus file not found: {retriever_cache_file}")
            
        return self

    @classmethod
    def load_from_local_memory(cls, memories: Dict, model_name: str) -> 'SimpleEmbeddingRetriever':
        """Load retriever state from memory"""
        # Create documents combining content and metadata for each memory
        all_docs = []
        for m in memories.values():
            metadata_text = f"{m.context} {' '.join(m.keywords)} {' '.join(m.tags)}"
            doc = f"{m.content} , {metadata_text}"
            all_docs.append(doc)
            
        # Create and initialize retriever
        retriever = cls(model_name)
        retriever.add_documents(all_docs)
        return retriever



class AgenticMemorySystem:
    """
    A multi-level memory system based on a graph structure.
    """
    def __init__(self, 
                 model_name: str = 'all-MiniLM-L6-v2',
                 llm_backend: str = "openai",
                 llm_model: str = "gpt-4o-mini",
                 evo_threshold: int = 100,
                 api_key: Optional[str] = None):
        
        self.graph = nx.DiGraph()
        self.turn_count = 0

        self.retriever = HybridRetriever(model_name, alpha=0.5)
        self.llm_controller = LLMController(llm_backend, llm_model, api_key)
        
        # Instantiate Managers
        self.scene_manager = SceneManager(self.llm_controller.llm, self.retriever)
        self.persona_manager = PersonaManager(self.llm_controller.llm, self.retriever)

        # --- Prompts (Note Construction & Evolution remain here as they are Level 0) ---
        self.note_construction_prompt = """Generate a structured analysis of the following content by:
            1. Identifying the most salient keywords (focus on nouns, verbs, and key concepts)
            2. Extracting core themes and contextual elements
            3. Creating relevant categorical tags

            Format the response as a JSON object:
            {
                "keywords": ["keyword1", "keyword2", ...],
                "context": "one sentence summarizing context",
                "tags": ["tag1", "tag2", ...]
            }"""

        self.json_format_example = '''
{
    "should_evolve": true,
    "actions": ["strengthen", "update_neighbor"],
    "suggested_connections": [0, 2],
    "tags_to_update": ["tag_1", "tag_2"],
    "new_context_neighborhood": ["new context 1", "new context 2"],
    "new_tags_neighborhood": [["tag_a", "tag_b"], ["tag_c", "tag_d"]]
}
'''
        
        # Prompt only contains placeholders, NO JSON structure
        self.evolution_system_prompt = '''
                               You are an AI memory evolution agent responsible for managing and evolving a knowledge base.
                               Analyze the the new memory note according to keywords and context, also with their several nearest neighbors memory.
                               Make decisions about its evolution.  

                               The new memory context:
                               {context}
                               content: {content}
                               keywords: {keywords}

                               The nearest neighbors memories (Indexed):
                               {nearest_neighbors_memories}

                               Based on this information, determine:
                               1. Should this memory be evolved? Consider its relationships with other memories.
                               2. What specific actions should be taken (strengthen, update_neighbor)?
                                  2.1 If choose to strengthen the connection, which memory should it be connected to? 
                                      **IMPORTANT: Use the [Index: X] number from the neighbors list.**
                                  2.2 If choose to update_neighbor, you can update the context and tags of these memories based on the understanding of these memories. If the context and the tags are not updated, the new context and tags should be the same as the original ones. Generate the new context and tags in the sequential order of the input neighbors.
                               Tags should be determined by the content of these characteristic of these memories, which can be used to retrieve them later and categorize them.
                               Note that the length of new_tags_neighborhood must equal the number of input neighbors, and the length of new_context_neighborhood must equal the number of input neighbors.
                               The number of neighbors is {neighbor_number}.
                               Return your decision in JSON format with the following structure:
                               '''
        
        self.evo_cnt = 0 
        self.evo_threshold = evo_threshold

    def _analyze_content(self, content: str) -> Dict:
        """Helper method to call LLM for metadata generation (Level 0)."""
        prompt = self.note_construction_prompt + "\n\nContent for analysis:\n" + content
        return self._get_llm_json_response(prompt)


    def _get_llm_json_response(self, prompt: str) -> Dict:
        """
        Helper to handle LLM call and JSON parsing robustly.
        Now uses regex extraction primarily and logs failures.
        """
        response = ""
        try:
            # Pass None for response_format to avoid strict schema errors
            response = self.llm_controller.llm.get_completion(prompt, response_format=None)
            
            # 1. Try to extract JSON from Markdown code blocks ```json ... ```
            match = re.search(r"```json\s*(\{.*\})\s*```", response, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            
            # 2. Try to extract JSON from generic code blocks ``` ... ```
            match = re.search(r"```\s*(\{.*\})\s*```", response, re.DOTALL)
            if match:
                return json.loads(match.group(1))

            # 3. Try to find the first '{' and last '}'
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if match:
                return json.loads(match.group(0))

            # 4. Try parsing directly
            return json.loads(response)

        except Exception as e:
            # Log only if it's not an empty response due to API error (handled in controller)
            if response and response != "{}":
                print(f"\n[ERROR] JSON Parsing Failed!")
                print(f"Error details: {str(e)}")
                print(f"--- Raw LLM Response Begin ---\n{response}\n--- Raw LLM Response End ---")
            return {}
    ####

    def add_note(self, content: str, time: str = None, **kwargs) -> str:
        """Adds a new fact-level memory note to the graph."""
        self.turn_count += 1
        node_id = str(uuid.uuid4())
        timestamp = time or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        analysis = self._analyze_content(content)
        
        note_data = MemoryNote(
            content=content,
            level='fact',
            timestamp=timestamp,
            keywords=analysis.get('keywords', []),
            context=analysis.get('context', ''),
            tags=analysis.get('tags', [])
        )
        
        self.graph.add_node(node_id, **note_data.to_dict())
        self._update_retriever([node_id])
        
        self.process_memory(node_id, note_data)
        
        return node_id
    
    def _update_retriever(self, node_ids: List[str]):
        """Helper to update the retriever with new or updated nodes."""
        docs_to_add = []
        for node_id in node_ids:
            if node_id not in self.graph.nodes: continue
            node_attrs = self.graph.nodes[node_id]
            doc = (
                f"content: {node_attrs.get('content', '')} "
                f"context: {node_attrs.get('context', '')} "
                f"keywords: {', '.join(node_attrs.get('keywords', []))} "
                f"tags: {', '.join(node_attrs.get('tags', []))}"
            )
            docs_to_add.append(doc)
        
        if docs_to_add:
            self.retriever.add_documents(docs_to_add)

####
# (Fix) Concat JSON example AFTER formatting
    def process_memory(self, node_id: str, note: MemoryNote) -> bool:
        """Level 0 Link Generation and Evolution."""
        neighbor_str, neighbor_ids = self.find_related_memories(note.content, k=5, strategy="flat")
        if not neighbor_ids:
            return False

        # 1. Format the base prompt first
        base_prompt = self.evolution_system_prompt.format(
            context=note.context, 
            content=note.content, 
            keywords=note.keywords, 
            nearest_neighbors_memories=neighbor_str,
            neighbor_number=len(neighbor_ids)
        )
        
        # 2. Append the JSON example safely
        prompt_memory = base_prompt + self.json_format_example
        
        response_json = self._get_llm_json_response(prompt_memory)
        should_evolve = response_json.get("should_evolve", False)
        
        # ... (Rest of logic regarding actions/strengthen/update remains same as CODE_29) ...
        # (Please copy the logic from CODE_29 here)
        
        if should_evolve:
            actions = response_json.get("actions", [])
            if "strengthen" in actions:
                suggested_connections = response_json.get("suggested_connections", [])
                for conn in suggested_connections:
                    target_id = None
                    if isinstance(conn, int) and 0 <= conn < len(neighbor_ids):
                        target_id = neighbor_ids[conn]
                    elif isinstance(conn, str) and conn in neighbor_ids:
                        target_id = conn
                    
                    if target_id and target_id != node_id:
                        self.graph.add_edge(node_id, target_id, type='persona_related')
                        self.graph.add_edge(target_id, node_id, type='persona_related')

            if "update_neighbor" in actions:
                new_contexts = response_json.get("new_context_neighborhood", [])
                new_tags_list = response_json.get("new_tags_neighborhood", [])
                for i, n_id in enumerate(neighbor_ids):
                    if i < len(new_contexts) and i < len(new_tags_list):
                        self.graph.nodes[n_id]['context'] = new_contexts[i]
                        self.graph.nodes[n_id]['tags'] = new_tags_list[i]

                         
        return should_evolve
####

    def _is_community_processed(self, community_nodes, parent_level):
        """Check if a community is already covered by a parent node."""
        if not community_nodes:
            return False
        sample_node = list(community_nodes)[0]
        predecessors = self.graph.predecessors(sample_node)
        for pred in predecessors:
            if self.graph.nodes[pred].get('level') == parent_level:
                return True
        return False

    def build_hierarchy_batch(self, save_callback=None, enable_refinement=False):
        """
        Batch process to build Level 1 and Level 2 memories using specialized Managers.
        Includes optional Top-Down Refinement phase with checkpointing.
        """
        print("\n=== Starting Hierarchical Memory Construction (Multi-Agent) ===")
        
        # --- Level 1 Construction (Delegated to SceneManager) ---
        print("--- Building Level 1 (Scene) Memories ---")
        
        level0_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('level') == 'fact']
        if not level0_nodes:
            print("No Level 0 nodes found.")
            return

        subgraph_l0 = self.graph.subgraph(level0_nodes).to_undirected()
        
        try:
            communities = list(nx.algorithms.community.label_propagation_communities(subgraph_l0))
        except Exception as e:
            print(f"Clustering failed: {e}. Using fallback chunks.")
            communities = [level0_nodes[i:i + 10] for i in range(0, len(level0_nodes), 10)]

        print(f"Identified {len(communities)} potential scenes/communities.")
        communities_to_process = [c for c in communities if not self._is_community_processed(c, 'scene')]
        
        print(f"Processing {len(communities_to_process)} remaining scenes...")

        with tqdm(total=len(communities), initial=len(communities)-len(communities_to_process), desc="Building Level 1 (Scenes)") as pbar:
            for comm in communities_to_process:
                scene_data_dict = self.scene_manager.build_scene(self.graph, list(comm))
                
                scene_id = scene_data_dict['id']
                del scene_data_dict['id']
                
                self.graph.add_node(scene_id, **scene_data_dict)
                
                for nid in comm:
                    self.graph.add_edge(scene_id, nid, type='contains')
                
                self._update_retriever([scene_id])
                if save_callback: save_callback()
                pbar.update(1)

        # --- Level 2 Construction (Delegated to PersonaManager) ---
        print("\n--- Building Level 2 (Persona) Memories ---")
        
        level2_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('level') == 'persona']
        if level2_nodes:
            print(f"Level 2 (Persona) memories already exist ({len(level2_nodes)} nodes). Skipping.")
            with tqdm(total=1, initial=1, desc="Building Level 2 (Persona)") as pbar: pass
        else:
            level1_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('level') == 'scene']
            if not level1_nodes:
                print("No Level 1 nodes found. Cannot build Level 2.")
                return

            print(f"Delegating Persona Profile generation to PersonaManager...")
            
            with tqdm(total=1, initial=0, desc="Building Level 2 (Persona)") as pbar:
                new_persona_nodes = self.persona_manager.build_profile(self.graph, level1_nodes)
                
                created_count = 0
                for node_data in new_persona_nodes:
                    node_id = node_data['id']
                    links = node_data['links']
                    
                    del node_data['id']
                    del node_data['links']
                    
                    self.graph.add_node(node_id, **node_data)
                    created_count += 1
                    
                    for target_id, weight in links:
                        self.graph.add_edge(node_id, target_id, type='contains', weight=weight)
                    
                    self._update_retriever([node_id])

                print(f"Created {created_count} Level 2 (Persona) nodes.")
                pbar.update(1)
                if save_callback: save_callback()

        # --- Phase 3: Top-Down Refinement (Optional) ---
        if enable_refinement:
            print("\n--- Starting Top-Down Refinement (Level 2 -> Level 1) ---")
            
            level2_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('level') == 'persona']
            if not level2_nodes:
                print("Skipping refinement: No Level 2 nodes available.")
            else:
                user_profile_text = ""
                for nid in level2_nodes:
                    node_attrs = self.graph.nodes[nid]
                    tag = node_attrs.get('tags', ['Unknown'])[0]
                    content = node_attrs.get('content', '')
                    user_profile_text += f"[{tag}]: {content}\n"
                
                level1_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('level') == 'scene']
                
                # Identify scenes that are NOT yet refined
                scenes_to_refine = [nid for nid in level1_nodes if not self.graph.nodes[nid].get('refined', False)]
                # scenes_to_refine = [nid for nid in level1_nodes]
                total_scenes = len(level1_nodes)
                processed_count = total_scenes - len(scenes_to_refine)
                
                print(f"Found {processed_count} already refined scenes.")
                print(f"Refining {len(scenes_to_refine)} remaining scenes...")
                
                updated_count = 0
                
                with tqdm(total=total_scenes, initial=processed_count, desc="Refining Scenes") as pbar:
                    for scene_id in scenes_to_refine:
                        current_summary = self.graph.nodes[scene_id].get('content', '')
                        
                        new_summary = self.scene_manager.refine_single_scene(current_summary, user_profile_text)
                        
                        if new_summary:
                            self.graph.nodes[scene_id]['content'] = new_summary
                            self._update_retriever([scene_id])
                            updated_count += 1
                        
                         

                        self.graph.nodes[scene_id]['refined'] = True
                        
                        pbar.update(1)
                        if save_callback and updated_count % 10 == 0: 
                            save_callback()
                
                print(f"Refinement complete. ")
                if save_callback: save_callback()

        print("\n=== Hierarchy Construction Complete ===")



    def find_related_memories(self, query: str, k: int = 5, strategy: str = "graph_traversal", top_k_levels: List[int] = [2, 3, 5]) -> (str, List[str]):
        """
        Retrieves memories using specified strategy with enhanced formatting and sorting.
        
        Args:
            query: The search query.
            k: Default top-k (used for flat/graph_traversal strategies).
            strategy: "flat", "graph_traversal", or "top_down".
            top_k_levels: [k_l2, k_l1, k_l0] for "top_down" strategy.
        """
        if self.graph.number_of_nodes() == 0:
            return "", []

        final_nodes = []
        seed_node_ids = [] # Keep track of directly retrieved nodes for prioritization

        if strategy == "flat":
            # Legacy flat retrieval
            indices = self.retriever.search(query, k)
            all_node_ids = list(self.graph.nodes)
            for i in indices:
                if i < len(all_node_ids):
                    nid = all_node_ids[i]
                    final_nodes.append(nid)
                    seed_node_ids.append(nid)
        
        elif strategy == "graph_traversal":
            # Strategy 2: Global Search + Structural Expansion
            
            # Seed Retrieval
            indices = self.retriever.search(query, k)
            all_node_ids = list(self.graph.nodes)
            seed_node_ids = []
            for i in indices:
                if i < len(all_node_ids):
                    nid = all_node_ids[i]

                    node_data = self.graph.nodes[nid]
                    level = node_data.get('level')
                    
                    # Only allow 'fact' nodes as seeds
                    if level in ['fact','scene','persona']:
                        seed_node_ids.append(nid)

            
            persona_context_nodes = set()
            scene_context_nodes = set()
            fact_context_nodes = set()
            
            # Expansion
            for nid in seed_node_ids:
                if nid not in self.graph.nodes: continue
                node_data = self.graph.nodes[nid]
                level = node_data.get('level')
                
                if level == 'persona':
                    # persona_context_nodes.add(nid)
                    # connected_scenes = list(self.graph.successors(nid))
                    # for child in connected_scenes[:3]: # Limit expansion
                    #     scene_context_nodes.add(child)
                    continue # Skip expansion for persona
                        
                elif level == 'scene':
                    scene_context_nodes.add(nid)
                    # Expansion: Get contained facts (Successors)
                    contained_facts = list(self.graph.successors(nid))
                    for child in contained_facts[:5]:
                        fact_context_nodes.add(child)
                    # continue
                        
                elif level == 'fact':
                    # continue
                    fact_context_nodes.add(nid)
                    # Expansion: Get parent scene for context (Predecessors)
                    parents = list(self.graph.predecessors(nid))
                    for parent in parents:
                        if self.graph.nodes[parent].get('level') == 'scene':
                            scene_context_nodes.add(parent)
                    

            extended_nodes = list(persona_context_nodes | scene_context_nodes | fact_context_nodes)
            unique_nodes = {}
            all_candidates = seed_node_ids + extended_nodes
            
            for nid in all_candidates:
                if nid not in self.graph.nodes: continue
                level = self.graph.nodes[nid].get('level')
                

                # if level in ['scene']:
                #     continue
                
                unique_nodes[nid] = True
            
            final_nodes = list(unique_nodes.keys())

        elif strategy == "top_down":
            # Strategy 1: Waterfall / Top-Down
            k2, k1, k0 = top_k_levels if len(top_k_levels) >= 3 else [2, 3, 5]
            
            current_query = query
            
            # --- Step 1: Retrieve Level 2 (Persona) ---
            l2_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('level') == 'persona']
            if not l2_nodes:
                return self.find_related_memories(query, k, strategy="flat")

  
            l2_texts = [str(self.graph.nodes[n].get('content', '')) for n in l2_nodes]

            
            query_emb = self.retriever.model.encode(current_query, convert_to_tensor=True)
            l2_embs = self.retriever.model.encode(l2_texts, convert_to_tensor=True)
            
            scores = util.cos_sim(query_emb, l2_embs)[0]
            top_results = torch.topk(scores, k=min(k2, len(l2_nodes)))
            
            selected_l2_ids = [l2_nodes[i] for i in top_results.indices]
            final_nodes.extend(selected_l2_ids)
            seed_node_ids.extend(selected_l2_ids) 
            
            # Enrich Query
            for nid in selected_l2_ids:
                current_query += " " + str(self.graph.nodes[nid].get('content', ''))

            # --- Step 2: Retrieve Level 1 (Scene) from Selected L2 Children ---
            candidate_l1_ids = set()
            for l2_id in selected_l2_ids:
                children = [n for n in self.graph.successors(l2_id) if self.graph.nodes[n].get('level') == 'scene']
                candidate_l1_ids.update(children)
            
            candidate_l1_ids = list(candidate_l1_ids)
            if candidate_l1_ids:
                l1_texts = [str(self.graph.nodes[n].get('content', '')) for n in candidate_l1_ids]
                
                query_emb = self.retriever.model.encode(current_query, convert_to_tensor=True)
                l1_embs = self.retriever.model.encode(l1_texts, convert_to_tensor=True)
                
                scores = util.cos_sim(query_emb, l1_embs)[0]
                top_results = torch.topk(scores, k=min(k1, len(candidate_l1_ids)))
                
                selected_l1_ids = [candidate_l1_ids[i] for i in top_results.indices]
                final_nodes.extend(selected_l1_ids)
                
                # Enrich Query
                for nid in selected_l1_ids:

                    current_query += " " + str(self.graph.nodes[nid].get('content', ''))
            else:
                selected_l1_ids = []

            # --- Step 3: Retrieve Level 0 (Fact) from Selected L1 Children ---
            candidate_l0_ids = set()
            for l1_id in selected_l1_ids:
                children = [n for n in self.graph.successors(l1_id) if self.graph.nodes[n].get('level') == 'fact']
                candidate_l0_ids.update(children)
            
            candidate_l0_ids = list(candidate_l0_ids)
            if candidate_l0_ids:
                l0_texts = [str(self.graph.nodes[n].get('content', '')) for n in candidate_l0_ids]

                
                query_emb = self.retriever.model.encode(current_query, convert_to_tensor=True)
                l0_embs = self.retriever.model.encode(l0_texts, convert_to_tensor=True)
                
                scores = util.cos_sim(query_emb, l0_embs)[0]
                top_results = torch.topk(scores, k=min(k0, len(candidate_l0_ids)))
                
                selected_l0_ids = [candidate_l0_ids[i] for i in top_results.indices]
                final_nodes.extend(selected_l0_ids)

        elif strategy == "bottom_up":
            k2, k1, k0 = top_k_levels if len(top_k_levels) >= 3 else [2, 3, 5]
            
            current_query = query
            
            # --- Step 1: Retrieve Level 0 (Fact) ---
            l0_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('level') == 'fact']
            if not l0_nodes:
                return self.find_related_memories(query, k, strategy="flat")

            l0_texts = [str(self.graph.nodes[n].get('content', '')) for n in l0_nodes]
            
            query_emb = self.retriever.model.encode(current_query, convert_to_tensor=True)
            l0_embs = self.retriever.model.encode(l0_texts, convert_to_tensor=True)
            
            scores = util.cos_sim(query_emb, l0_embs)[0]
            top_results = torch.topk(scores, k=min(k0, len(l0_nodes)))
            
            selected_l0_ids = [l0_nodes[i] for i in top_results.indices]
            final_nodes.extend(selected_l0_ids)
            seed_node_ids.extend(selected_l0_ids)
            
            # Enrich Query
            for nid in selected_l0_ids:
                current_query += " " + str(self.graph.nodes[nid].get('content', ''))

            # --- Step 2: Retrieve Level 1 (Scene) from Parents of Selected L0 ---
            candidate_l1_ids = set()
            for l0_id in selected_l0_ids:
                # Get parents via 'contains' edge (predecessors)
                parents = [n for n in self.graph.predecessors(l0_id) if self.graph.nodes[n].get('level') == 'scene']
                candidate_l1_ids.update(parents)
            
            candidate_l1_ids = list(candidate_l1_ids)
            if candidate_l1_ids:
                l1_texts = [str(self.graph.nodes[n].get('content', '')) for n in candidate_l1_ids]
                
                query_emb = self.retriever.model.encode(current_query, convert_to_tensor=True)
                l1_embs = self.retriever.model.encode(l1_texts, convert_to_tensor=True)
                
                scores = util.cos_sim(query_emb, l1_embs)[0]
                top_results = torch.topk(scores, k=min(k1, len(candidate_l1_ids)))
                
                selected_l1_ids = [candidate_l1_ids[i] for i in top_results.indices]
                final_nodes.extend(selected_l1_ids)
                
                # Enrich Query
                for nid in selected_l1_ids:
                    current_query += " " + str(self.graph.nodes[nid].get('content', ''))
            else:
                selected_l1_ids = []

            # --- Step 3: Retrieve Level 2 (Persona) from Parents of Selected L1 ---
            candidate_l2_ids = set()
            for l1_id in selected_l1_ids:
                parents = [n for n in self.graph.predecessors(l1_id) if self.graph.nodes[n].get('level') == 'persona']
                candidate_l2_ids.update(parents)
            
            candidate_l2_ids = list(candidate_l2_ids)
            if candidate_l2_ids:
                l2_texts = [str(self.graph.nodes[n].get('content', '')) for n in candidate_l2_ids]
                
                query_emb = self.retriever.model.encode(current_query, convert_to_tensor=True)
                l2_embs = self.retriever.model.encode(l2_texts, convert_to_tensor=True)
                
                scores = util.cos_sim(query_emb, l2_embs)[0]
                top_results = torch.topk(scores, k=min(k2, len(candidate_l2_ids)))
                
                selected_l2_ids = [candidate_l2_ids[i] for i in top_results.indices]
                final_nodes.extend(selected_l2_ids)
    ####
        # 4. Formatting with Time and Level info
        memory_str = ""
        
        # For graph_traversal, we already sorted by seed priority.
        # For top_down, final_nodes is already in hierarchical order.
        # So we just iterate and format.
        
        for node_id in final_nodes:
            node_attrs = self.graph.nodes[node_id]
            level = node_attrs.get('level', 'unknown').upper()
            timestamp = node_attrs.get('timestamp', 'Unknown Time')
            
            tag_info = f" [{', '.join(node_attrs.get('tags', []))}]" if node_attrs.get('tags') else ""
            
            content_str = str(node_attrs.get('content', ''))
            # Explicitly format: [LEVEL] [Time: ...] Content
            # This helps the model with Temporal questions and context awareness
            memory_str += f"[Index: {i}] [{level}] [Time: {timestamp}]{tag_info}\n{node_attrs.get('content', '')}\n\n"
            
        return memory_str, final_nodes
    

    def find_related_memories_raw(self, query: str, k: int = 5, strategy: str = "graph_traversal", top_k_levels: List[int] = None) -> str:

        if top_k_levels is None:
            top_k_levels = [2, 3, 5]
            
        memory_str, _ = self.find_related_memories(query, k, strategy=strategy, top_k_levels=top_k_levels)
        return memory_str

    

    def consolidate_memories(self):
        """
        Rebuilds the entire retriever index from the graph.
        """
        print("Consolidating memories: Rebuilding retriever index from scratch.")
        all_node_ids = list(self.graph.nodes())
        
        # Re-initialize HybridRetriever
        try:
            # Try to get model name from existing retriever's internal model
            model_name = self.retriever.model.get_config_dict().get('name', 'all-MiniLM-L6-v2')
        except:
            model_name = 'all-MiniLM-L6-v2'
            
        # Recreate HybridRetriever instance
        self.retriever = HybridRetriever(model_name=model_name, alpha=0.5)
        
        # Populate it
        self._update_retriever(all_node_ids)
        
        print(f"Retriever rebuilt with {len(all_node_ids)} documents.")
    ####

class TransformersController(BaseLLMController):
    def __init__(self, model_path: str):
        print(f"Loading local model from: {model_path}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16, 
            device_map="auto"
        )

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )
        print("Local model loaded successfully.")
    


    def _extract_json_from_response(self, text: str) -> str:


        match = re.search(r"```json\s*(\{.*\})\s*```", text, re.DOTALL)
        if match:
            # print("DEBUG: Extracted JSON using ```json block.")
            return match.group(1)

        match = re.search(r"```\s*(\{.*\})\s*```", text, re.DOTALL)
        if match:
            # print("DEBUG: Extracted JSON using ``` block.")
            return match.group(1)

        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and start < end:
            # print("DEBUG: Extracted JSON using find '{' and '}'.")
            return text[start:end+1]
            
        # print("DEBUG: No JSON object found in the response.")
        return ""

    def get_completion(self, prompt: str, response_format: dict = None, temperature: float = 0.7) -> str:

        messages = [
            {"role": "system", "content": "You are a helpful assistant that always responds in JSON format."},
            {"role": "user", "content": prompt},
        ]
        
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        

        generation_args = {
            "max_new_tokens": 1024,
            "return_full_text": False, 
            "temperature": temperature if temperature > 0.01 else 0.01,
            "do_sample": True if temperature > 0.01 else False,
        }
        
        try:

            outputs = self.pipe(formatted_prompt, **generation_args)
            raw_response = outputs[0]['generated_text']

            print(f"\n--- Raw Model Response ---\n{raw_response}\n--------------------------\n")

            json_string = self._extract_json_from_response(raw_response)

            if json_string:
                processed_string = json_string.replace(": True", ": true").replace(": False", ": false").replace(": None", ": null")
                try:
                    json.loads(processed_string)
                    return processed_string
                except json.JSONDecodeError as e:
                    print(f"Warning: Processed string is not valid JSON. Error: {e}")
                    print(f"Original extracted string was: {json_string}")
                    print(f"Processed string was: {processed_string}")
            

            print(f"Warning: Failed to extract a valid JSON object from the model's response.")

            if "keywords" in prompt and "context" in prompt and "tags" in prompt:
                 return json.dumps({"keywords": [], "context": "Generation Failed", "tags": []})
            elif "answer" in prompt:
                 return json.dumps({"answer": "Generation Failed"})
            else:
                 return json.dumps({}) 

        except Exception as e:

            print(f"Error during text generation with local model: {e}")
            return json.dumps({}) 

if __name__ == "__main__":
    run_tests()