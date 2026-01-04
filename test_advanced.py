from memory_layer import LLMController, AgenticMemorySystem
import os
import json
import argparse
import logging
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import numpy as np
# from openai import OpenAI # Not used directly anymore
from load_dataset import load_locomo_dataset, QA, Turn, Session, Conversation
import nltk
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import pytorch_cos_sim
import statistics
from collections import defaultdict
import pickle
import random
from tqdm import tqdm
from utils import calculate_metrics, aggregate_metrics
from datetime import datetime
import re

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')

# Initialize SentenceTransformer model (this will be reused)
try:
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    print(f"Warning: Could not load SentenceTransformer model: {e}")
    sentence_model = None

class advancedMemAgent:
    def __init__(self, model, backend, retrieve_k, api_key=None, retrieval_strategy="graph_traversal", top_k_levels=None):
        self.memory_system = AgenticMemorySystem(
            model_name='all-MiniLM-L6-v2',
            llm_backend=backend,
            llm_model=model,
            api_key=api_key
        )
        self.retriever_llm = LLMController(
            backend=backend, 
            model=model, 
            api_key=api_key
        )
        self.retrieve_k = retrieve_k
        self.retrieval_strategy = retrieval_strategy
        self.top_k_levels = top_k_levels or [2, 3, 5]

    def add_memory(self, content, time=None):
        self.memory_system.add_note(content, time=time)

    def retrieve_memory(self, content, k=10):
        return self.memory_system.find_related_memories(
            content, 
            k=k, 
            strategy=self.retrieval_strategy, 
            top_k_levels=self.top_k_levels
        )
    def retrieve_memory_llm(self, memories_text, query):
        prompt = f"""Given the following conversation memories and a question, select the most relevant parts of the conversation that would help answer the question. Include the date/time if available.

                Conversation memories:
                {memories_text}

                Question: {query}

                Return only the relevant parts of the conversation that would help answer this specific question. Format your response as a JSON object with a "relevant_parts" field containing the selected text. 
                If no parts are relevant, do not do any things just return the input.

                Example response format:
                {{"relevant_parts": "2024-01-01: Speaker A said something relevant..."}}"""
            
        response = self.retriever_llm.llm.get_completion(prompt,response_format={"type": "json_schema", "json_schema": {
                            "name": "response",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "relevant_parts": {
                                        "type": "string",
                                    }
                                },
                                "required": ["relevant_parts"],
                                "additionalProperties": False
                            },
                            "strict": True
                        }})
        return response
    
    def generate_query_llm(self, question):
        prompt = f"""Given the following question, generate several keywords, using 'cosmos' as the separator.

                Question: {question}

                Format your response as a JSON object with a "keywords" field containing the selected text. 

                Example response format:
                {{"keywords": "keyword1, keyword2, keyword3"}}"""
            
        response = self.retriever_llm.llm.get_completion(prompt,response_format={"type": "json_schema", "json_schema": {
                            "name": "response",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "keywords": {
                                        "type": "string",
                                    }
                                },
                                "required": ["keywords"],
                                "additionalProperties": False
                            },
                            "strict": True
                        }})
        # print("response:{}".format(response))
        try:
            # Try to parse JSON, handle potential extra text if necessary (though get_completion should handle it)
            import re
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if match:
                response_json = json.loads(match.group(0))
                response = response_json.get("keywords", response)
            else:
                response = response.strip()
        except:
            response = response.strip()
        return response



    def answer_question(self, question: str, category: int, answer: str) -> str:
        """Generate answer for a question given the conversation context."""
        keywords = self.generate_query_llm(question)
        if category == 2: # Temporal
            current_k = 25
        elif category == 4: # Single-Hop
            current_k = 25
        elif category == 3: # Open-domain
            current_k = 30
        else: # Multi-Hop
            current_k = 35 # Defaults to 35 if passed in CLI
        # current_k=20
        raw_context, retrieved_ids = self.retrieve_memory(keywords, k=current_k)
        context = raw_context
        
        assert category in [1,2,3,4,5]
        
        # --- Optimized Prompts (Inspired by GAM) ---
        
        # Common formatting instruction to encourage short, precise answers (F1/BLEU friendly)
        format_instruction = """
        Write an answer in the form of **a short phrase**, not a complete sentence. 
        Answer with exact words from the context whenever possible.
        """

            
        if category == 2: # Temporal
            user_prompt = f"""
                            Based on the context provided below:
                            {context}

                            Question: {question}

                            Instructions:
                            1. Identify the specific [EVENT] related to the question.
                            2. Look at the [Time: ...] timestamp of that event. Treat this as the "Reference Date".
                            
                            3. **Date Format**: 
                               - You MUST use the format "Day Month, Year" (e.g., "15 July, 2023"). 
                               - Note the COMMA after the month.
                            
                            4. **Relative Time Logic (CRITICAL)**:
                               - If the context says "last week" or "a week ago", answer in the format: "The week before [Reference Date]".
                               - If the context says "last month", answer: "The month before [Reference Date]".
                               - If the context says "next week", answer: "The week after [Reference Date]".
                               - If the context says "yesterday" or specific days, calculate the exact date: "14 July, 2023".
                               
                            5. **Examples**:
                               - Context Time: "20 May, 2023". Text: "I went last week." -> Answer: "The week before 20 May, 2023"
                               - Context Time: "April, 2023". Text: "Going next month." -> Answer: "May, 2023"
                            
                            6. If the question asks for duration, answer in "X years/months/days".
                            
                            Short answer:
                            """
            temperature = 0.0 

        elif category == 3: # open_domain
            inference_format_instruction = """
        Write an answer in the form of **a short phrase**, not a complete sentence.
        The question may need you to **analyze and infer** the answer from the summary, rather than finding exact words.
        """
            temperature = 0.0
            q_lower = question.lower()

            is_choice = " or " in q_lower and "?" in q_lower
            is_yes_no = q_lower.startswith(("is ", "are ", "do ", "does ", "did ", "can ", "could ", "would ", "will ", "should "))
            
            if is_choice:
                cat3_instruction = """
                1. The question asks you to choose between options (e.g., "A or B?").
                2. Answer with **ONLY** the chosen option. Do not add "I think" or reasons.
                3. Example: "beach" (NOT "He prefers the beach").
                """
            elif is_yes_no:
                cat3_instruction = """
                1. Answer "Yes", "No", "Likely", or "Unlikely".
                2. You MAY add a very short reason (3-5 words) starting with ";".
                3. Example: "Yes; she loves art."
                """
            else:
                cat3_instruction = """
                1. Analyze the user's personality and history to infer the answer.
                2. If the answer is not explicitly stated, make an **educated guess** based on context clues. Avoid saying "Not specified".
                3. Write a **very short phrase** (1-5 words). No full sentences.
                """

            user_prompt = f"""
                            Based on the context provided below:
                            {context}

                            Question: {question}

                            Instructions:
                            {cat3_instruction}
                            
                            Short answer:
                            """

        elif category == 1: # Multi-Hop
            inference_format_instruction = """
        Write an answer in the form of **a short phrase**, not a complete sentence.
        The question may need you to **analyze and infer** the answer from the summary, rather than finding exact words.
        """
            
            temperature = 0.0 
            q_lower = question.lower()
            is_yes_no = q_lower.startswith(("is ", "are ", "do ", "does ", "did ", "can ", "could ", "would ", "will ", "should "))
            is_count = q_lower.startswith(("how many", "how much"))
            
            if is_yes_no:
                # Optimized for strict evaluation
                cat1_specific_instruction = """
                - Answer with **EXACTLY ONE word** from this list: [Yes, No, Likely, Unlikely]. 
                - Do NOT add any explanation or punctuation.
                """
            elif is_count:
                # Optimized for numeric answers
                cat1_specific_instruction = """
                - Answer with **only the number** (e.g., "3" or "three").
                - Do NOT add words like "times", "items", "people".
                - Do NOT use "once" or "twice"; use numeric form instead.
                """
            else:
                # Optimized for Entity Extraction & Reasoning
                cat1_specific_instruction = """
                - **List Exhaustion**: If the question asks for multiple items (e.g., "Which books"), list **ALL** specific entities mentioned in [EVENT] blocks. Do NOT summarize.
                - **Reference Resolution**: If the text mentions a pronoun (e.g., "my home country"), you MUST search the context to find the **specific proper noun** (e.g., "Sweden").
                - **Temporal Filtering**: If the question asks about "recently" or "last", compare [Time: ...] stamps to find the latest event.
                - **Exact Name**: Provide the full, exact name of the entity if asked.
                """

            user_prompt = f"""
                            Based on the context provided below:
                            {context}

                            Question: {question}

                            Instructions:
                            1. This is a Multi-Hop question. You need to connect pieces of information.
                            2. {cat1_specific_instruction}
                            3. Write an answer in the form of **a short phrase**, not a complete sentence.
                            
                            Short answer:
                            """
            temperature = 0.0
            
        else: # Category 4 (Single Hop) 
            # Both are fact-based retrieval tasks, use Exact Words prompt
            exact_format_instruction = """
        Write an answer in the form of **a short phrase**, not a complete sentence. 
        Answer with **exact words** from the context whenever possible.
        """
            user_prompt = f"""
                            Based on the context provided below:
                            {context}

                            Question: {question}

                            Instructions:
                            1. {format_instruction}
                            
                            Short answer:
                            """
            temperature = 0.0

        response = self.memory_system.llm_controller.llm.get_completion(
            user_prompt,response_format={"type": "json_schema", "json_schema": {
                        "name": "response",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "answer": {
                                    "type": "string",
                                }
                            },
                            "required": ["answer"],
                            "additionalProperties": False
                        },
                        "strict": True
                    }},temperature=temperature
        )
        prediction = "" # Global default
        
        try:
            clean_response = response.strip()
            
            # ... (regex extraction remains same) ...
            match = re.search(r"```json\s*(\{.*\})\s*```", clean_response, re.DOTALL)
            if match:
                clean_response = match.group(1)
            elif re.search(r"```\s*(\{.*\})\s*```", clean_response, re.DOTALL):
                clean_response = re.search(r"```\s*(\{.*\})\s*```", clean_response, re.DOTALL).group(1)
            
            response_json = json.loads(clean_response)
            
            # Explicitly initialize local variable for this block
            local_prediction = None 

            if isinstance(response_json, dict):
                for key in ["answer", "short_answer", "prediction", "result", "content"]:
                    if key in response_json:
                        local_prediction = str(response_json[key])
                        break
                
                # Fallback: take first value
                if not local_prediction and response_json:
                    local_prediction = str(list(response_json.values())[0])
                    
                # Empty dict case
                if not local_prediction:
                     local_prediction = str(response_json)

            else:
                local_prediction = str(response_json)
            
            # Assign back to main variable if we found something
            if local_prediction:
                prediction = local_prediction
                
        except (json.JSONDecodeError, AttributeError, ValueError):
            prediction = response.strip()
        
        # ... (rest of cleanup logic remains same) ...
        if prediction:
            prefixes_to_remove = ["Answer:", "Short answer:", "The answer is", "Prediction:"]
            for prefix in prefixes_to_remove:
                if prediction.lower().startswith(prefix.lower()):
                    prediction = prediction[len(prefix):].strip()
                    break 
            
            if prediction.startswith('"') and prediction.endswith('"'):
                prediction = prediction[1:-1]
        
        if not prediction and response:
            prediction = response.strip()

                
        return prediction, user_prompt, raw_context, retrieved_ids

def setup_logger(log_file: Optional[str] = None) -> logging.Logger:
    """Set up logging configuration."""
    logger = logging.getLogger('locomo_eval')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if log_file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Helper function for saving checkpoints
def save_checkpoint_internal(agent, mem_file, ret_file, ret_emb_file, turns):
    """Standalone helper to save checkpoints."""
    try:
        with open(mem_file, 'wb') as f:
            pickle.dump({'turns_processed': turns, 'graph': agent.memory_system.graph}, f)
        agent.memory_system.retriever.save(ret_file, ret_emb_file)
    except Exception as e:
        print(f"Error saving checkpoint: {e}")

def evaluate_dataset(dataset_path: str, model: str, output_path: Optional[str] = None, ratio: float = 1.0, backend: str = "openai", retrieve_k: int = 10, api_key: Optional[str] = None, retrieval_strategy: str = "graph_traversal", top_k_levels: List[int] = None, enable_refinement: bool = False):
    """
    Evaluate the agent on the LoComo dataset with full end-to-end checkpointing,
    robust progress visualization, multi-level hierarchy construction, and selectable retrieval strategy.
    """
    # Generate automatic log filename with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    model_name_sanitized = model.replace("/", "_").replace(":", "_")
    log_filename = f"eval_ours_{model_name_sanitized}_{backend}_ratio{ratio}_{timestamp}.log"
    
    if output_path:
        log_path = os.path.join(os.path.dirname(__file__), "logs", log_filename)
    else:
        log_path = os.path.join(os.path.dirname(__file__), "logs", log_filename)
    
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    logger = setup_logger(log_path)
    logger.info(f"Loading dataset from {dataset_path}")
    
    # Load dataset
    samples = load_locomo_dataset(dataset_path)
    logger.info(f"Loaded {len(samples)} samples")
    
    # Select subset of samples based on ratio
    if ratio < 1.0:
        num_samples = max(1, int(len(samples) * ratio))
        samples = samples[:num_samples]
        logger.info(f"Using {num_samples} samples ({ratio*100:.1f}% of dataset)")

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if not os.path.exists(output_path):
            with open(output_path, 'w', encoding='utf-8') as f:
                logger.info(f"Initializing new results file at {output_path}")
                meta_info = {
                    "type": "metadata",
                    "model": model,
                    "backend": backend,
                    "dataset": str(dataset_path),
                    "ratio": ratio,
                    "retrieval_strategy": retrieval_strategy,
                    "evaluation_start_time": datetime.now().isoformat()
                }
                f.write(json.dumps(meta_info) + "\n")
        else:
            logger.info(f"Found existing results file. Will resume and append to {output_path}")

    all_metrics = []
    all_categories = []
    total_questions = 0
    category_counts = defaultdict(int)
    error_num = 0

    if output_path:
        base_cache_dir = os.path.dirname(output_path)
    else:
        base_cache_dir = os.path.dirname(__file__)

    memories_dir = os.path.join(base_cache_dir, "cached_memories")
    os.makedirs(memories_dir, exist_ok=True)
    logger.info(f"Using memory cache directory: {memories_dir}")

    allow_categories = [1,2,3,4]

    for sample_idx, sample in enumerate(samples):
        agent = advancedMemAgent(model, backend, retrieve_k, api_key=api_key, retrieval_strategy=retrieval_strategy, top_k_levels=top_k_levels)
        
        memory_cache_file = os.path.join(
            memories_dir,
            f"memory_cache_sample_{sample.sample_id}.pkl"
        )
        retriever_cache_file = os.path.join(
            memories_dir,
            f"retriever_cache_sample_{sample.sample_id}_retriever.pkl"
        )
        retriever_cache_embeddings_file = os.path.join(
            memories_dir,
            f"retriever_cache_sample_{sample.sample_id}_embeddings.npy"
        )
        
        CHECKPOINT_INTERVAL = 10
        turns_processed = 0
        import networkx as nx
        cached_graph = nx.DiGraph() 

        if os.path.exists(memory_cache_file):
            logger.info(f"Loading cached memories for sample {sample.sample_id} from {memory_cache_file}")
            try:
                with open(memory_cache_file, 'rb') as f:
                    saved_data = pickle.load(f)
                    turns_processed = saved_data.get('turns_processed', 0)
                    cached_graph = saved_data.get('graph', nx.DiGraph())
            except (pickle.UnpicklingError, EOFError, AttributeError, ImportError) as e:
                logger.warning(f"Could not load memory checkpoint file (error: {e}). Starting from scratch for this sample.")
                turns_processed = 0
                cached_graph = nx.DiGraph()
            
            agent.memory_system.graph = cached_graph
            
            if turns_processed > 0 and os.path.exists(retriever_cache_file) and os.path.exists(retriever_cache_embeddings_file):
                logger.info(f"Loading retriever from cache for sample {sample.sample_id}")
                agent.memory_system.retriever.load(retriever_cache_file, retriever_cache_embeddings_file)
            else:
                if agent.memory_system.graph.number_of_nodes() > 0:
                    logger.warning(f"Retriever cache not found or is outdated. Rebuilding retriever from loaded graph.")
                    all_node_ids = list(agent.memory_system.graph.nodes())
                    from memory_layer import SimpleEmbeddingRetriever
                    try:
                        agent.memory_system.retriever = SimpleEmbeddingRetriever('all-MiniLM-L6-v2')
                    except:
                        pass
                    agent.memory_system._update_retriever(all_node_ids)

            logger.info(f"Successfully loaded {agent.memory_system.graph.number_of_nodes()} memories, resuming from turn {turns_processed}.")
        else:
            logger.info(f"No cached memories found for sample {sample.sample_id}. Starting new memory creation.")

        all_turns = []
        for _, session in sorted(sample.conversation.sessions.items()):
            for turn in session.turns:
                all_turns.append((session.date_time, turn))

        turns_to_process = all_turns[turns_processed:]
        
        if turns_to_process:
            logger.info(f"Processing {len(turns_to_process)} remaining turns for sample {sample.sample_id}.")
            
            try:
                with tqdm(total=len(all_turns), initial=turns_processed, desc=f"Building memory for sample {sample.sample_id}") as pbar:
                    for i, (turn_datetime, turn) in enumerate(turns_to_process):
                        conversation_tmp = f"Speaker {turn.speaker} says: {turn.text}"
                        agent.add_memory(conversation_tmp, time=turn_datetime)
                        
                        current_turn_index = turns_processed + i + 1

                        if current_turn_index % CHECKPOINT_INTERVAL == 0:
                            logger.info(f"\n--- Saving checkpoint at turn {current_turn_index} ---")
                            save_checkpoint_internal(agent, memory_cache_file, retriever_cache_file, retriever_cache_embeddings_file, current_turn_index)
                            logger.info(f"Checkpoint saved successfully.")
                        
                        pbar.update(1)

            except Exception as e:
                current_loop_index = i if 'i' in locals() else 0
                logger.error(f"An error occurred during memory building at global turn index {turns_processed + 1 + current_loop_index}: {e}")
                logger.error("Attempting to save progress before exiting...")
                final_turns_processed = turns_processed + current_loop_index
                save_checkpoint_internal(agent, memory_cache_file, retriever_cache_file, retriever_cache_embeddings_file, final_turns_processed)
                logger.error("Progress saved. Now re-raising the exception.")
                raise e
        else:
            logger.info(f"All turns for sample {sample.sample_id} already processed. Skipping memory building.")

        if turns_to_process:
            logger.info(f"\nFinished building memory for sample {sample.sample_id}. Saving final state.")
            final_turns_processed = len(all_turns)
            save_checkpoint_internal(agent, memory_cache_file, retriever_cache_file, retriever_cache_embeddings_file, final_turns_processed)
            logger.info(f"Final memory state saved for {agent.memory_system.graph.number_of_nodes()} memories.")

        # --- Hierarchy Construction Phase ---
        logger.info(f"\nStarting Hierarchy Construction for sample {sample.sample_id}")
        try:
            # Pass the enable_refinement flag
            agent.memory_system.build_hierarchy_batch(
                save_callback=lambda: save_checkpoint_internal(agent, memory_cache_file, retriever_cache_file, retriever_cache_embeddings_file, len(all_turns)),
                enable_refinement=enable_refinement
            )
            logger.info(f"Hierarchy construction completed. Saving final graph state.")
            save_checkpoint_internal(agent, memory_cache_file, retriever_cache_file, retriever_cache_embeddings_file, len(all_turns))
        except Exception as e:
            logger.error(f"An error occurred during hierarchy construction: {e}")
            logger.error("Attempting to save progress before exiting...")
            save_checkpoint_internal(agent, memory_cache_file, retriever_cache_file, retriever_cache_embeddings_file, len(all_turns))
            raise e

        logger.info(f"\nAnswering questions for sample {sample.sample_id}")
        
        ####
        # (问题32, 改动32) Precision Resume Logic
        # 1. Build a set of processed question signatures (sample_id, question_text)
        processed_questions_set = set()
        if output_path and os.path.exists(output_path):
            with open(output_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        if entry.get("type") == "qa_result":
                             # Check if it belongs to current sample (string comparison for safety)
                             if str(entry.get("sample_id")) == str(sample.sample_id):
                                 q_text = entry.get("question", "").strip()
                                 processed_questions_set.add(q_text)
                    except json.JSONDecodeError:
                        continue
        
        logger.info(f"Found {len(processed_questions_set)} already processed unique questions for sample {sample.sample_id}.")

        # 2. Filter the questions to be processed
        qas_in_sample = [qa for qa in sample.qa if int(qa.category) in allow_categories]
        qas_to_process = []
        
        # We need to maintain the original index for 'question_id' consistency
        qas_to_process_with_index = []
        
        for idx, qa in enumerate(qas_in_sample):
            if qa.question.strip() not in processed_questions_set:
                qas_to_process_with_index.append((idx, qa))
        
        logger.info(f"Queueing {len(qas_to_process_with_index)} questions to answer.")

        # 3. Process
        if qas_to_process_with_index:
            # We use total=len(qas_in_sample) to show global progress relative to the full set
            # But we only iterate over the ones that need processing
            initial_progress = len(qas_in_sample) - len(qas_to_process_with_index)
            
            with tqdm(total=len(qas_in_sample), initial=initial_progress, desc=f"Answering questions for sample {sample.sample_id}") as pbar:
                for original_idx, qa in qas_to_process_with_index:
                    # Use original_idx + 1 as the ID to keep it consistent regardless of skip order
                    global_question_id = original_idx + 1

                    total_questions += 1
                    category_counts[qa.category] += 1
                    
                    prediction, user_prompt, raw_context, retrieved_node_ids = agent.answer_question(qa.question, qa.category, qa.final_answer)
                    
                    try:
                        # Clean up prediction string if it's a JSON string
                        if isinstance(prediction, str) and prediction.strip().startswith('{'):
                             try:
                                 pred_json = json.loads(prediction)
                                 prediction = pred_json.get("answer", prediction)
                             except:
                                 pass
                    except:
                        pass
                    
                    # Ensure prediction is a string
                    prediction = str(prediction)

                    logger.info(f"\nQuestion {global_question_id} (Sample {sample.sample_id}): {qa.question}")
                    logger.info(f"Prediction: {prediction}")
                    logger.info(f"Reference: {qa.final_answer}")
                    # logger.info(f"User Prompt: {user_prompt}") # Optional: Reduce log verbosity
                    # logger.info(f"Raw Context: {raw_context}")
                    
                    metrics = calculate_metrics(prediction, qa.final_answer) if qa.final_answer else {
                        "exact_match": 0, "f1": 0.0, "rouge1_f": 0.0, "rouge2_f": 0.0, 
                        "rougeL_f": 0.0, "bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, 
                        "bleu4": 0.0, "bert_f1": 0.0, "meteor": 0.0, "sbert_similarity": 0.0
                    }
                    
                    all_metrics.append(metrics)
                    all_categories.append(qa.category)
                    
                    result_to_save = {
                        "type": "qa_result",
                        "sample_id": sample.sample_id,
                        "question_id_in_sample": global_question_id, # Stable ID
                        "question": qa.question,
                        "prediction": prediction,
                        "reference": qa.final_answer,
                        "category": qa.category,
                        "metrics": metrics,
                        "retrieved_nodes": retrieved_node_ids
                    }
                    
                    if output_path:
                        with open(output_path, 'a', encoding='utf-8') as f:
                            f.write(json.dumps(result_to_save, ensure_ascii=False) + "\n")
                    
                    pbar.update(1)
        else:
            logger.info("All questions for this sample are already processed.")
        ####
    
    if total_questions > 0:
        aggregate_results = aggregate_metrics(all_metrics, all_categories)
        
        run_summary = {
            "type": "run_summary",
            "model": model,
            "questions_processed_in_this_run": total_questions,
            "error_num_in_this_run": error_num,
            "category_distribution_in_this_run": {
                str(cat): count for cat, count in category_counts.items()
            },
            "aggregate_metrics_for_this_run": aggregate_results,
            "run_end_time": datetime.now().isoformat()
        }

        if output_path:
            with open(output_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(run_summary) + "\n")
            logger.info(f"Run finished. Summary for this run appended to {output_path}")

        logger.info("\nSummary for This Run:")
        logger.info(f"Total questions evaluated in this run: {total_questions}")
        logger.info(f"JSON parsing errors on predictions: {error_num}")
        logger.info("\nCategory Distribution in this run:")
        for category, count in sorted(category_counts.items()):
            logger.info(f"Category {category}: {count} questions ({count/total_questions*100:.1f}%)")
        
        logger.info("\nAggregate Metrics for this run:")
        for split_name, metrics in aggregate_results.items():
            logger.info(f"\n{split_name.replace('_', ' ').title()}:")
            for metric_name, stats in metrics.items():
                logger.info(f"  {metric_name}:")
                for stat_name, value in stats.items():
                    logger.info(f"    {stat_name}: {value:.4f}")
    else:
        logger.info("\nNo new questions were processed in this run.")
    
    return {}

def main():
    parser = argparse.ArgumentParser(description="Evaluate text-only agent on LoComo dataset")
    parser.add_argument("--dataset", type=str, default="data/locomo10.json",
                      help="Path to the dataset file")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                      help="OpenAI model to use")
    parser.add_argument("--output", type=str, default=None,
                      help="Path to save evaluation results")
    parser.add_argument("--ratio", type=float, default=1.0,
                      help="Ratio of dataset to evaluate (0.0 to 1.0)")
    parser.add_argument("--backend", type=str, default="openai",
                      help="Backend to use (openai or ollama)")
    parser.add_argument("--retrieve_k", type=int, default=10,
                      help="Retrieve k")
    parser.add_argument("--api-key", type=str, default=None,
                      help="API key for the selected backend")
    parser.add_argument("--enable-refinement", action="store_true", 
                      help="Enable top-down refinement of scene memories using generated user profiles.")
    
    # New arguments for retrieval strategy
    parser.add_argument("--retrieval-strategy", type=str, default="graph_traversal", 
                      choices=["flat", "graph_traversal", "top_down","bottom_up"],
                      help="Strategy for memory retrieval")
    parser.add_argument("--top-k-levels", type=int, nargs=3, default=[2, 3, 5],
                      help="Top-k for L2, L1, L0 respectively (only for top_down strategy). E.g. --top-k-levels 2 3 5")

    args = parser.parse_args()
    
    if args.ratio <= 0.0 or args.ratio > 1.0:
        raise ValueError("Ratio must be between 0.0 and 1.0")
    
    dataset_path = os.path.join(os.path.dirname(__file__), args.dataset)
    output_path = args.output if args.output else None
    
    evaluate_dataset(
        dataset_path, args.model, output_path, args.ratio, args.backend, args.retrieve_k, 
        api_key=args.api_key, retrieval_strategy=args.retrieval_strategy, top_k_levels=args.top_k_levels,
        enable_refinement=args.enable_refinement
    )
if __name__ == "__main__":
    main()