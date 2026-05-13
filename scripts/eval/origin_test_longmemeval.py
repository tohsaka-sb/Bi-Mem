import os
import json
import argparse
import logging
import re
from typing import List, Optional, Union
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
from datetime import datetime

from tqdm import tqdm

from src.bimem.core.load_dataset import QA, Turn, Session, Conversation
from src.bimem.core.memory_layer import LLMController
from src.bimem.core.utils import calculate_metrics, aggregate_metrics

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

LONG_CONTEXT_MODELS = {"gpt-5", "gpt-5-mini", "gpt-5o", "gpt-5o-mini"}
DEFAULT_CONTEXT_TOKENS = 12000
LONG_CONTEXT_TOKENS = 32000

def resolve_context_window(model: str, max_context_tokens: Optional[int]) -> int:
    if max_context_tokens is not None:
        return max_context_tokens
    if model.lower() in LONG_CONTEXT_MODELS:
        return LONG_CONTEXT_TOKENS
    return DEFAULT_CONTEXT_TOKENS

def append_jsonl(path: Optional[str], payload: dict):
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")

def _parse_judge_score(raw: str) -> float:
    s = str(raw or "").strip()
    m = re.search(r"\b(\d{1,3})\b", s)
    if m:
        val = float(m.group(1))
        return min(100.0, max(0.0, val))
    return 50.0

def calculate_llm_judge_score(
    question: str,
    prediction: str,
    reference: str,
    client: "OpenAI",
    model: str = "gpt-4o",
) -> float:
    if not client or not reference:
        return 50.0
    prompt = f"""You are a strict evaluator. Score how well the model's prediction answers the question, given the reference answer.

Question: {question}
Reference answer: {reference}
Model's prediction: {prediction}

Scoring (0-100):
- 100: Fully correct, semantically equivalent or highly accurate
- 75-99: Mostly correct, minor omissions or paraphrasing
- 50-74: Partially correct
- 25-49: Mostly wrong, significant errors
- 0-24: Completely wrong or irrelevant

Reply with ONLY an integer score from 0 to 100 (no other text)."""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Reply with only an integer from 0 to 100."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=10,
        )
        raw = (response.choices[0].message.content or "").strip()
        return _parse_judge_score(raw)
    except Exception:
        return 50.0

@dataclass
class LongMemSample:
    sample_id: str
    qa: List[QA]
    conversation: Conversation

def load_longmemeval_dataset(file_path: Union[str, Path], ratio: float = 1.0, max_samples: Optional[int] = None) -> List[LongMemSample]:
    file_path = Path(file_path) if isinstance(file_path, str) else file_path
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("longmemeval JSON should be a list of QA items")

    n_total = len(data)
    n_use = max(1, int(n_total * ratio)) if ratio < 1.0 else n_total
    if max_samples is not None:
        n_use = min(n_use, max_samples)
    data = data[:n_use]

    samples = []
    for idx, item in enumerate(data):
        haystack_sessions = item.get("haystack_sessions") or []
        haystack_dates = item.get("haystack_dates") or []
        sessions_dict = {}
        for i, sess in enumerate(haystack_sessions):
            date_time = haystack_dates[i] if i < len(haystack_dates) else ""
            turns = [
                Turn(speaker=msg.get("role", "user"), dia_id="", text=msg.get("content", ""))
                for msg in sess
                if msg.get("content")
            ]
            if turns:
                sessions_dict[i] = Session(session_id=i, date_time=date_time, turns=turns)

        conversation = Conversation(speaker_a="user", speaker_b="assistant", sessions=sessions_dict)
        qa_list = [
            QA(
                question=item.get("question", ""),
                answer=item.get("answer"),
                evidence=item.get("answer_session_ids", []) or [],
                category=4,
                adversarial_answer=None,
            )
        ]
        sample_id = item.get("question_id", str(idx))
        samples.append(LongMemSample(sample_id=sample_id, qa=qa_list, conversation=conversation))
    return samples

def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)

def truncate_keep_latest(text: str, max_tokens: int) -> str:
    if estimate_tokens(text) <= max_tokens:
        return text
    lines = text.splitlines()
    kept = []
    used = 0
    for line in reversed(lines):
        line_tokens = estimate_tokens(line + "\n")
        if kept and used + line_tokens > max_tokens:
            break
        kept.append(line)
        used += line_tokens
    if not kept:
        return text[-max_tokens * 4 :]
    kept.reverse()
    return "\n".join(kept)

def build_full_context(sample: LongMemSample) -> str:
    parts = []
    for _, session in sorted(sample.conversation.sessions.items()):
        for turn in session.turns:
            parts.append(f"[Time: {session.date_time}] Speaker {turn.speaker} says: {turn.text}")
    return "\n".join(parts)

class DirectContextAgent:
    def __init__(
        self,
        model: str,
        backend: str,
        api_key: Optional[str],
        empty_answer_retries: int = 2,
        raw_response_dump_path: Optional[str] = None,
    ):
        self.model = model
        self.backend = backend
        self.llm_controller = LLMController(backend=backend, model=model, api_key=api_key)
        self.empty_answer_retries = max(0, empty_answer_retries)
        self.raw_response_dump_path = raw_response_dump_path

    @staticmethod
    def _extract_prediction(raw_response: str) -> str:
        prediction = ""
        try:
            clean_response = raw_response.strip()
            match = re.search(r"```json\s*(\{.*\})\s*```", clean_response, re.DOTALL)
            if match:
                clean_response = match.group(1)
            elif re.search(r"```\s*(\{.*\})\s*```", clean_response, re.DOTALL):
                clean_response = re.search(r"```\s*(\{.*\})\s*```", clean_response, re.DOTALL).group(1)
            response_json = json.loads(clean_response)
            if isinstance(response_json, dict):
                for key in ["answer", "short_answer", "prediction", "result", "content"]:
                    if key in response_json:
                        prediction = str(response_json[key])
                        break
                if not prediction and response_json:
                    prediction = str(list(response_json.values())[0])
            else:
                prediction = str(response_json)
        except (json.JSONDecodeError, AttributeError, ValueError):
            prediction = raw_response.strip()

        if prediction:
            for prefix in ["Answer:", "Short answer:", "The answer is", "Prediction:"]:
                if prediction.lower().startswith(prefix.lower()):
                    prediction = prediction[len(prefix) :].strip()
                    break
            if prediction.startswith('"') and prediction.endswith('"'):
                prediction = prediction[1:-1]
        if not prediction and raw_response:
            prediction = raw_response.strip()
        return prediction

    def answer_question(
        self,
        question: str,
        category: int,
        context: str,
        sample_id: Optional[str] = None,
        question_id: Optional[int] = None,
    ):
        assert category in [1, 2, 3, 4, 5]

        format_instruction = """
        Write an answer in the form of a short phrase, not a complete sentence.
        Answer with exact words from the context whenever possible.
        """
        anti_refusal_instruction = """
        CRITICAL:
        - Do not answer with generic refusal phrases like "Not specified", "Unknown", "Not mentioned",
          "The context does not provide...", or similar.
        - If evidence is partial, still output your best concise guess grounded in the given context.
        - Prefer concrete entities, dates, numbers, or short noun phrases.
        """

        if category == 2:
            user_prompt = f"""
                            Based on the context provided below:
                            {context}

                            Question: {question}

                            Instructions:
                            1. Identify the specific event in context.
                            2. Use [Time: ...] as the reference date.
                            3. Date format should be Day Month, Year.
                            4. {anti_refusal_instruction}

                            Short answer:
                            """
            temperature = 0.0
        elif category == 3:
            q_lower = question.lower()
            is_choice = " or " in q_lower and "?" in q_lower
            is_yes_no = q_lower.startswith(("is ", "are ", "do ", "does ", "did ", "can ", "could ", "would ", "will ", "should "))
            if is_choice:
                cat3_instruction = "Answer with only the chosen option."
            elif is_yes_no:
                cat3_instruction = "Answer Yes/No/Likely/Unlikely. Keep it short."
            else:
                cat3_instruction = "Infer from context and answer in a very short phrase."
            user_prompt = f"""
                            Based on the context provided below:
                            {context}

                            Question: {question}

                            Instructions:
                            {cat3_instruction}
                            {anti_refusal_instruction}

                            Short answer:
                            """
            temperature = 0.0
        elif category == 1:
            user_prompt = f"""
                            Based on the context provided below:
                            {context}

                            Question: {question}

                            Instructions:
                            1. This is a Multi-Hop question. Connect relevant clues.
                            2. Give the shortest correct phrase.
                            3. {anti_refusal_instruction}

                            Short answer:
                            """
            temperature = 0.0
        else:
            user_prompt = f"""
                            Based on the context provided below:
                            {context}

                            Question: {question}

                            Instructions:
                            1. {format_instruction}
                            2. {anti_refusal_instruction}

                            Short answer:
                            """
            temperature = 0.0

        prediction = ""
        for attempt in range(self.empty_answer_retries + 1):
            raw_response = ""
            error_msg = None
            try:
                raw_response = self.llm_controller.llm.get_completion(
                    user_prompt,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "response",
                            "schema": {
                                "type": "object",
                                "properties": {"answer": {"type": "string"}},
                                "required": ["answer"],
                                "additionalProperties": False,
                            },
                            "strict": True,
                        },
                    },
                    temperature=temperature,
                )
                prediction = self._extract_prediction(raw_response)
            except Exception as e:
                error_msg = str(e)

            append_jsonl(
                self.raw_response_dump_path,
                {
                    "ts": datetime.now().isoformat(),
                    "model": self.model,
                    "backend": self.backend,
                    "sample_id": sample_id,
                    "question_id_in_sample": question_id,
                    "question": question,
                    "attempt": attempt + 1,
                    "category": category,
                    "prediction": prediction,
                    "prediction_is_empty": (not bool((prediction or "").strip())),
                    "error": error_msg,
                    "raw_response": raw_response,
                },
            )
            if prediction and prediction.strip():
                break
        return prediction

def setup_logger(log_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger("longmem_eval_origin")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger

def evaluate_longmem_dataset(
    dataset_path: str,
    model: str,
    output_path: Optional[str] = None,
    ratio: float = 1.0,
    max_samples: Optional[int] = None,
    backend: str = "openai",
    api_key: Optional[str] = None,
    max_context_tokens: Optional[int] = None,
    sample_id: Optional[str] = None,
    no_llm_judge: bool = False,
    judge_model: str = "gpt-4o",
    empty_answer_retries: int = 2,
    raw_response_dump_path: Optional[str] = None,
):
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    model_name_sanitized = model.replace("/", "_").replace(":", "_")
    log_filename = f"eval_origin_longmem_{model_name_sanitized}_{backend}_ratio{ratio}_{timestamp}.log"
    log_path = os.path.join(os.path.dirname(__file__), "logs", log_filename)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = setup_logger(log_path)

    effective_context_tokens = resolve_context_window(model, max_context_tokens)
    logger.info(f"Loading LongMemEval dataset from {dataset_path}")
    logger.info(
        f"Context token budget: {effective_context_tokens} (auto model-aware default applied: {max_context_tokens is None})"
    )
    samples = load_longmemeval_dataset(dataset_path, ratio=ratio, max_samples=max_samples)
    if sample_id is not None:
        samples = [s for s in samples if str(s.sample_id) == str(sample_id)]
        if not samples:
            raise ValueError(f"sample_id={sample_id} not found in dataset")
    logger.info(f"Loaded {len(samples)} LongMemEval samples")

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if not os.path.exists(output_path):
            with open(output_path, "w", encoding="utf-8") as f:
                meta_info = {
                    "type": "metadata",
                    "dataset_type": "longmemeval",
                    "model": model,
                    "backend": backend,
                    "dataset": str(dataset_path),
                    "ratio": ratio,
                    "max_samples": max_samples,
                    "retrieval_strategy": "direct_context",
                    "max_context_tokens": effective_context_tokens,
                    "empty_answer_retries": empty_answer_retries,
                    "raw_response_dump_path": raw_response_dump_path,
                    "evaluation_start_time": datetime.now().isoformat(),
                }
                f.write(json.dumps(meta_info, ensure_ascii=False) + "\n")
        else:
            logger.info(f"Found existing results file. Will resume and append to {output_path}")

    judge_client = None
    if not no_llm_judge and OpenAI:
        judge_api_key = api_key or os.getenv("OPENAI_API_KEY")
        if judge_api_key:
            judge_client = OpenAI(api_key=judge_api_key, base_url=os.getenv("OPENAI_API_BASE"))

    all_metrics = []
    all_categories = []
    total_questions = 0
    category_counts = defaultdict(int)
    error_num = 0
    allow_categories = [4]

    for sample in samples:
        agent = DirectContextAgent(
            model=model,
            backend=backend,
            api_key=api_key,
            empty_answer_retries=empty_answer_retries,
            raw_response_dump_path=raw_response_dump_path,
        )
        full_context = build_full_context(sample)
        context = truncate_keep_latest(full_context, effective_context_tokens)

        processed_questions_set = set()
        if output_path and os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        if entry.get("type") == "qa_result" and str(entry.get("sample_id")) == str(sample.sample_id):
                            processed_questions_set.add(entry.get("question", "").strip())
                    except json.JSONDecodeError:
                        continue

        qas_in_sample = [qa for qa in sample.qa if int(qa.category) in allow_categories]
        qas_to_process_with_index = []
        for idx, qa in enumerate(qas_in_sample):
            if qa.question.strip() not in processed_questions_set:
                qas_to_process_with_index.append((idx, qa))

        if qas_to_process_with_index:
            initial_progress = len(qas_in_sample) - len(qas_to_process_with_index)
            with tqdm(total=len(qas_in_sample), initial=initial_progress, desc=f"Answering questions for sample {sample.sample_id}") as pbar:
                for original_idx, qa in qas_to_process_with_index:
                    global_question_id = original_idx + 1
                    total_questions += 1
                    category_counts[qa.category] += 1

                    prediction = agent.answer_question(
                        qa.question,
                        qa.category,
                        context,
                        sample_id=sample.sample_id,
                        question_id=global_question_id,
                    )
                    prediction = str(prediction)

                    metrics = calculate_metrics(prediction, qa.final_answer) if qa.final_answer else {
                        "exact_match": 0,
                        "f1": 0.0,
                        "rouge1_f": 0.0,
                        "rouge2_f": 0.0,
                        "rougeL_f": 0.0,
                        "bleu1": 0.0,
                        "bleu2": 0.0,
                        "bleu3": 0.0,
                        "bleu4": 0.0,
                        "bert_f1": 0.0,
                        "meteor": 0.0,
                        "sbert_similarity": 0.0,
                        "llm_as_judge": 50.0,
                    }

                    if judge_client and qa.final_answer:
                        metrics["llm_as_judge"] = calculate_llm_judge_score(
                            qa.question, prediction, qa.final_answer, judge_client, model=judge_model
                        )
                    else:
                        metrics["llm_as_judge"] = 50.0

                    core_metrics = {
                        "f1": metrics.get("f1", 0.0),
                        "bleu1": metrics.get("bleu1", 0.0),
                        "llm_as_judge": metrics.get("llm_as_judge", 50.0),
                    }

                    all_metrics.append(metrics)
                    all_categories.append(qa.category)

                    result_to_save = {
                        "type": "qa_result",
                        "sample_id": sample.sample_id,
                        "question_id_in_sample": global_question_id,
                        "question": qa.question,
                        "prediction": prediction,
                        "reference": qa.final_answer,
                        "category": qa.category,
                        "metrics": metrics,
                        "f1": core_metrics["f1"],
                        "bleu1": core_metrics["bleu1"],
                        "llm_as_judge": core_metrics["llm_as_judge"],
                        "retrieved_nodes": [],
                    }

                    if output_path:
                        with open(output_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(result_to_save, ensure_ascii=False) + "\n")
                    pbar.update(1)

    if total_questions > 0:
        aggregate_results = aggregate_metrics(all_metrics, all_categories)
        run_summary = {
            "type": "run_summary",
            "model": model,
            "questions_processed_in_this_run": total_questions,
            "error_num_in_this_run": error_num,
            "category_distribution_in_this_run": {str(cat): count for cat, count in category_counts.items()},
            "aggregate_metrics_for_this_run": aggregate_results,
            "run_end_time": datetime.now().isoformat(),
        }
        if output_path:
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(run_summary, ensure_ascii=False) + "\n")
    return {}

def main():
    parser = argparse.ArgumentParser(description="Evaluate direct-context baseline on LongMemEval dataset")
    parser.add_argument("--dataset", type=str, default="data/longmemeval_s_cleaned.json", help="Path to LongMemEval JSON file")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="LLM model to use")
    parser.add_argument("--output", type=str, default=None, help="Path to save evaluation results (JSONL)")
    parser.add_argument("--ratio", type=float, default=0.01, help="Fraction of dataset to use (0.0 to 1.0)")
    parser.add_argument("--max-samples", type=int, default=None, help="Max number of samples to evaluate")
    parser.add_argument("--backend", type=str, default="openai", help="Backend: openai, ollama, probex")
    parser.add_argument("--api-key", type=str, default=None, help="API key for backend")
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=None,
        help="Context budget override. If unset, gpt-5/gpt-5-mini/gpt-5o/gpt-5o-mini use 32000, others use 12000.",
    )
    parser.add_argument("--sample-id", type=str, default=None, help="Run only one sample_id")
    parser.add_argument("--no-llm-judge", action="store_true", help="Disable LLM-as-Judge scoring")
    parser.add_argument("--judge-model", type=str, default="gpt-4o", help="Model for LLM-as-Judge")
    parser.add_argument(
        "--empty-answer-retries",
        type=int,
        default=2,
        help="Retry count when extracted answer is empty.",
    )
    parser.add_argument(
        "--raw-response-dump",
        type=str,
        default=None,
        help="Path to debug JSONL for raw model responses (all attempts).",
    )

    args = parser.parse_args()
    if args.ratio <= 0.0 or args.ratio > 1.0:
        raise ValueError("Ratio must be in (0.0, 1.0]")
    if args.max_context_tokens is not None and args.max_context_tokens <= 0:
        raise ValueError("max-context-tokens must be > 0")
    if args.empty_answer_retries < 0:
        raise ValueError("empty-answer-retries must be >= 0")

    dataset_path = os.path.join(os.path.dirname(__file__), args.dataset)
    output_path = args.output if args.output else None
    raw_response_dump_path = args.raw_response_dump
    if raw_response_dump_path is None and output_path:
        root, _ = os.path.splitext(output_path)
        raw_response_dump_path = f"{root}.raw_responses.jsonl"
    evaluate_longmem_dataset(
        dataset_path=dataset_path,
        model=args.model,
        output_path=output_path,
        ratio=args.ratio,
        max_samples=args.max_samples,
        backend=args.backend,
        api_key=args.api_key,
        max_context_tokens=args.max_context_tokens,
        sample_id=args.sample_id,
        no_llm_judge=args.no_llm_judge,
        judge_model=args.judge_model,
        empty_answer_retries=args.empty_answer_retries,
        raw_response_dump_path=raw_response_dump_path,
    )

if __name__ == "__main__":
    main()

