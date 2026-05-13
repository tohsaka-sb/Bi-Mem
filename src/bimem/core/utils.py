import re
import string
import numpy as np
from typing import List, Dict, Union
import statistics
from collections import defaultdict
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bert_score
import nltk
from nltk.translate.meteor_score import meteor_score
from sentence_transformers import SentenceTransformer
import logging
from dataclasses import dataclass
from pathlib import Path

from sentence_transformers.util import pytorch_cos_sim
import time

_sentence_model = None

def _get_sentence_model():
    global _sentence_model
    if _sentence_model is None:
        try:
            _sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            print(f"Warning: Could not load SentenceTransformer model: {e}")
            _sentence_model = False
    return None if _sentence_model is False else _sentence_model

def simple_tokenize(text):
    """Simple tokenization function."""

    text = str(text)
    return text.lower().replace('.', ' ').replace(',', ' ').replace('!', ' ').replace('?', ' ').split()

def calculate_rouge_scores(prediction: str, reference: str) -> Dict[str, float]:
    """Calculate ROUGE scores for prediction against reference."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return {
        'rouge1_f': scores['rouge1'].fmeasure,
        'rouge2_f': scores['rouge2'].fmeasure,
        'rougeL_f': scores['rougeL'].fmeasure
    }

def _word_tokenize(text: str) -> List[str]:
    """Tokenize text for BLEU/ROUGE. Uses regex to avoid NLTK punkt (no network download)."""
    return re.findall(r"\b\w+\b", text.lower()) if text else []

def calculate_bleu_scores(prediction: str, reference: str) -> Dict[str, float]:
    """Calculate BLEU scores with different n-gram settings."""
    pred_tokens = _word_tokenize(prediction)
    ref_tokens = [_word_tokenize(reference)]
    
    weights_list = [(1, 0, 0, 0), (0.5, 0.5, 0, 0), (0.33, 0.33, 0.33, 0), (0.25, 0.25, 0.25, 0.25)]
    smooth = SmoothingFunction().method1
    
    scores = {}
    for n, weights in enumerate(weights_list, start=1):
        try:
            score = sentence_bleu(ref_tokens, pred_tokens, weights=weights, smoothing_function=smooth)
        except Exception:
            score = 0.0
        scores[f'bleu{n}'] = score
    
    return scores

def calculate_bert_scores(prediction: str, reference: str) -> Dict[str, float]:
    """Calculate BERTScore for persona similarity with debug logging."""
    print(f"[DEBUG] Starting calculate_bert_scores at {time.strftime('%X')}")
    start_time = time.time()
    
    try:
        print(f"[DEBUG] Calling bert_score function... (This may take time to load model)")
        P, R, F1 = bert_score([prediction], [reference], lang='en', verbose=True)
        
        end_time = time.time()
        print(f"[DEBUG] bert_score calculation finished in {end_time - start_time:.2f} seconds.")
        
        return {
            'bert_precision': P.item(),
            'bert_recall': R.item(),
            'bert_f1': F1.item()
        }
    except Exception as e:
        print(f"[ERROR] Error calculating BERTScore: {e}")

        import traceback
        traceback.print_exc()
        return {
            'bert_precision': 0.0,
            'bert_recall': 0.0,
            'bert_f1': 0.0
        }

def calculate_meteor_score(prediction: str, reference: str) -> float:
    """Calculate METEOR score for the prediction."""
    try:
        return meteor_score([reference.split()], prediction.split())
    except Exception as e:
        print(f"Error calculating METEOR score: {e}")
        return 0.0

def calculate_sentence_similarity(prediction: str, reference: str) -> float:
    """Calculate sentence embedding similarity using SentenceBERT."""
    model = _get_sentence_model()
    if model is None:
        return 0.0
    try:

        embedding1 = model.encode([prediction], convert_to_tensor=True)
        embedding2 = model.encode([reference], convert_to_tensor=True)

        similarity = pytorch_cos_sim(embedding1, embedding2).item()
        return float(similarity)
    except Exception as e:
        print(f"Error calculating sentence similarity: {e}")
        return 0.0

def calculate_metrics(prediction: str, reference: str) -> Dict[str, float]:
    """Calculate comprehensive evaluation metrics for a prediction."""

    if not prediction or not reference:
        return {
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
            "sbert_similarity": 0.0
        }

    prediction = str(prediction).strip()
    reference = str(reference).strip()

    exact_match = int(prediction.lower() == reference.lower())

    pred_tokens = set(simple_tokenize(prediction))
    ref_tokens = set(simple_tokenize(reference))
    common_tokens = pred_tokens & ref_tokens
    
    if not pred_tokens or not ref_tokens:
        f1 = 0.0
    else:
        precision = len(common_tokens) / len(pred_tokens)
        recall = len(common_tokens) / len(ref_tokens)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    rouge_scores = calculate_rouge_scores(prediction, reference)
    bleu_scores = calculate_bleu_scores(prediction, reference)
    bert_scores = calculate_bert_scores(prediction, reference)
    meteor = calculate_meteor_score(prediction, reference)
    sbert_similarity = calculate_sentence_similarity(prediction, reference)

    metrics = {
        "exact_match": exact_match,
        "f1": f1,
        **rouge_scores,
        **bleu_scores,
        **bert_scores,
        "meteor": meteor,
        "sbert_similarity": sbert_similarity
    }
    
    return metrics

def aggregate_metrics(all_metrics: List[Dict[str, float]], all_categories: List[int]) -> Dict[str, Dict[str, Union[float, Dict[str, float]]]]:
    """Calculate aggregate statistics for all metrics, split by category."""
    if not all_metrics:
        return {}

    aggregates = defaultdict(list)
    category_aggregates = defaultdict(lambda: defaultdict(list))

    for metrics, category in zip(all_metrics, all_categories):
        for metric_name, value in metrics.items():
            aggregates[metric_name].append(value)
            category_aggregates[category][metric_name].append(value)

    results = {
        "overall": {}
    }
    
    for metric_name, values in aggregates.items():
        results["overall"][metric_name] = {
            'mean': statistics.mean(values),
            'std': statistics.stdev(values) if len(values) > 1 else 0.0,
            'median': statistics.median(values),
            'min': min(values),
            'max': max(values),
            'count': len(values)
        }

    for category in sorted(category_aggregates.keys()):
        results[f"category_{category}"] = {}
        for metric_name, values in category_aggregates[category].items():
            if values:
                results[f"category_{category}"][metric_name] = {
                    'mean': statistics.mean(values),
                    'std': statistics.stdev(values) if len(values) > 1 else 0.0,
                    'median': statistics.median(values),
                    'min': min(values),
                    'max': max(values),
                    'count': len(values)
                }
    
    return results
