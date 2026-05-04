#!/usr/bin/env python3
"""Build consolidated data.json from multilingual evaluation results.

Reads results from results/{Language}/{model}/{checkpoint}/{task}/{partition}/
and aggregates across prompt/partition variants (p0, p1, p2).

Output: docs/data.json
"""

import json
import math
import os
import glob
import statistics
from pathlib import Path

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_FILE = BASE_DIR / "docs" / "data.json"

SHOT_SETTINGS = ["0", "5"]

# Batch size: 2048 sequences * 1024 tokens = 2,097,152 tokens per step
TOKENS_PER_STEP = 2048 * 1024


def step_to_tokens_b(step):
    """Convert training step number to billions of tokens."""
    val = round(step * TOKENS_PER_STEP / 1e9, 1)
    return int(val) if val == int(val) else val


def parse_checkpoint_name(name):
    """Parse checkpoint name to billions of tokens, or 'main'.

    Handles numeric step names (e.g. '0001000'), token-based names
    (e.g. '10B'), and 'main'.
    """
    if name.isdigit():
        return step_to_tokens_b(int(name))
    upper = name.upper()
    if upper.endswith("B") and upper[:-1].isdigit():
        return int(upper[:-1])
    if name == "main":
        return "main"
    return None


# ── Per-language task configuration ──
# main_metric, random_baseline, max_performance, metric_scale, category

TASK_CONFIG = {
    # ── Finnish tasks ──
    "arc_challenge_fi_cf_fbv2": {
        "pretty_name": "ARC Challenge (Finnish, CF)",
        "main_metric": "acc",
        "random_baseline": 0.25,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "world knowledge",
    },
    "arc_challenge_fi_mcf_fbv2": {
        "pretty_name": "ARC Challenge (Finnish, MCF)",
        "main_metric": "acc",
        "random_baseline": 0.25,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "world knowledge",
    },
    "belebele_fin_cf_fbv2": {
        "pretty_name": "Belebele (Finnish, CF)",
        "main_metric": "acc",
        "random_baseline": 0.25,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "language understanding",
    },
    "belebele_fin_mcf_fbv2": {
        "pretty_name": "Belebele (Finnish, MCF)",
        "main_metric": "acc",
        "random_baseline": 0.25,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "language understanding",
    },
    "finbench_analogies_cf_fbv2": {
        "pretty_name": "FinnBench Analogies (CF)",
        "main_metric": "acc",
        "random_baseline": 0.20,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "linguistic knowledge",
    },
    "finbench_analogies_mcf_fbv2": {
        "pretty_name": "FinnBench Analogies (MCF)",
        "main_metric": "acc",
        "random_baseline": 0.20,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "linguistic knowledge",
    },
    "finbench_emotions_1k_cf_fbv2": {
        "pretty_name": "FinnBench Emotions (CF)",
        "main_metric": "acc",
        "random_baseline": 0.125,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "language understanding",
    },
    "finbench_emotions_1k_mcf_fbv2": {
        "pretty_name": "FinnBench Emotions (MCF)",
        "main_metric": "acc",
        "random_baseline": 0.125,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "language understanding",
    },
    "finbench_general_knowledge_cf_fbv2": {
        "pretty_name": "FinnBench General Knowledge (CF)",
        "main_metric": "acc",
        "random_baseline": 0.133,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "world knowledge",
    },
    "finbench_general_knowledge_mcf_fbv2": {
        "pretty_name": "FinnBench General Knowledge (MCF)",
        "main_metric": "acc",
        "random_baseline": 0.133,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "world knowledge",
    },
    "finbench_hhh_alignment_cf_fbv2": {
        "pretty_name": "FinnBench HHH Alignment (CF)",
        "main_metric": "acc",
        "random_baseline": 0.5,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "language understanding",
    },
    "finbench_hhh_alignment_mcf_fbv2": {
        "pretty_name": "FinnBench HHH Alignment (MCF)",
        "main_metric": "acc",
        "random_baseline": 0.5,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "language understanding",
    },
    "finbench_similarities_abstraction_cf_fbv2": {
        "pretty_name": "FinnBench Similarities (CF)",
        "main_metric": "acc",
        "random_baseline": 0.25,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "linguistic knowledge",
    },
    "finbench_similarities_abstraction_mcf_fbv2": {
        "pretty_name": "FinnBench Similarities (MCF)",
        "main_metric": "acc",
        "random_baseline": 0.25,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "linguistic knowledge",
    },
    "goldenswag_ht_fi_cf_fbv2": {
        "pretty_name": "GoldenSwag (Finnish, CF)",
        "main_metric": "acc",
        "random_baseline": 0.25,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "language understanding",
    },
    "goldenswag_ht_fi_mcf_fbv2": {
        "pretty_name": "GoldenSwag (Finnish, MCF)",
        "main_metric": "acc",
        "random_baseline": 0.25,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "language understanding",
    },
    "ogx_truthfulqax_gen_fi_fbv2": {
        "pretty_name": "TruthfulQA Gen (Finnish)",
        "main_metric": "bleu_max",
        "random_baseline": 0.0,
        "max_performance": 100.0,
        "metric_scale": "percent",
        "category": "world knowledge",
    },
    "ogx_truthfulqax_mc1_fi_fbv2": {
        "pretty_name": "TruthfulQA MC1 (Finnish)",
        "main_metric": "acc",
        "random_baseline": 0.25,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "world knowledge",
    },
    "ogx_truthfulqax_mc2_fi_fbv2": {
        "pretty_name": "TruthfulQA MC2 (Finnish)",
        "main_metric": "acc",
        "random_baseline": 0.5,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "world knowledge",
    },
    "scandisent_fi_cf_fbv2": {
        "pretty_name": "ScandiSent (Finnish, CF)",
        "main_metric": "acc",
        "random_baseline": 1 / 3,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "language understanding",
    },
    "scandisent_fi_mcf_fbv2": {
        "pretty_name": "ScandiSent (Finnish, MCF)",
        "main_metric": "acc",
        "random_baseline": 1 / 3,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "language understanding",
    },
    "sib200_fi_cf_fbv2": {
        "pretty_name": "SIB-200 (Finnish, CF)",
        "main_metric": "acc",
        "random_baseline": 1 / 7,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "language understanding",
    },
    "sib200_fi_mcf_fbv2": {
        "pretty_name": "SIB-200 (Finnish, MCF)",
        "main_metric": "acc",
        "random_baseline": 1 / 7,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "language understanding",
    },
    "squad_fi_gen_fbv2": {
        "pretty_name": "SQuAD (Finnish, Gen)",
        "main_metric": "f1",
        "random_baseline": 0.0,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "language understanding",
    },
    # ── Norwegian tasks ──
    # Categories and baselines mirror github.com/ltgoslo/noreval-stats
    # (metrics_setup.yaml). The `path` field is set when the task's
    # results live in a non-default subdirectory.
    "ask_gec": {
        "pretty_name": "ASK GEC",
        "main_metric": "errant",
        "random_baseline": 0.0,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "linguistic knowledge",
    },
    "ncb": {
        "pretty_name": "NCB (Norwegian Cloze)",
        "main_metric": "acc",
        "random_baseline": 0.5,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "linguistic knowledge",
    },
    "nocola": {
        "pretty_name": "NoCoLA",
        "main_metric": "acc",
        "random_baseline": 0.5,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "linguistic knowledge",
    },
    "noreval_multiblimp": {
        "pretty_name": "MultiBLiMP (Sámi)",
        "main_metric": "acc",
        "random_baseline": 0.5,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "linguistic knowledge",
        "aggregator": "multiblimp",
    },
    "slide": {
        "pretty_name": "SLIDE (Scandinavian LID)",
        "main_metric": "acc",
        "random_baseline": 0.21289208633093526,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "linguistic knowledge",
    },
    "norbelebele": {
        "pretty_name": "Belebele (Norwegian)",
        "main_metric": "acc",
        "random_baseline": 0.25,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "language understanding",
    },
    "norcommonsenseqa_nno": {
        "pretty_name": "CommonsenseQA (Nynorsk)",
        "main_metric": "acc",
        "random_baseline": 0.2,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "world knowledge & reasoning",
    },
    "norcommonsenseqa_nob": {
        "pretty_name": "CommonsenseQA (Bokmål)",
        "main_metric": "acc",
        "random_baseline": 0.2,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "world knowledge & reasoning",
    },
    "norec_document": {
        "pretty_name": "NoReC Document Sentiment",
        "main_metric": "f1",
        "random_baseline": 0.5,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "language understanding",
    },
    "norec_sentence": {
        "pretty_name": "NoReC Sentence Sentiment",
        "main_metric": "f1",
        "random_baseline": 0.5,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "language understanding",
    },
    "noridiom_nno": {
        "pretty_name": "NorIdiom (Nynorsk)",
        "main_metric": "em_first",
        "random_baseline": 0.0,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "linguistic knowledge",
    },
    "noridiom_nob": {
        "pretty_name": "NorIdiom (Bokmål)",
        "main_metric": "em_first",
        "random_baseline": 0.0,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "linguistic knowledge",
    },
    "noropenbookqa_nno": {
        "pretty_name": "OpenBookQA (Nynorsk)",
        "main_metric": "acc",
        "random_baseline": 0.25,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "language understanding",
        "path": "noropenbookqa/noropenbookqa_nno",
    },
    "noropenbookqa_nob": {
        "pretty_name": "OpenBookQA (Bokmål)",
        "main_metric": "acc",
        "random_baseline": 0.25,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "language understanding",
        "path": "noropenbookqa/noropenbookqa_nob",
    },
    "noropenbookqa_no_fact_nno": {
        "pretty_name": "OpenBookQA No-Fact (Nynorsk)",
        "main_metric": "acc",
        "random_baseline": 0.25,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "world knowledge & reasoning",
        "path": "noropenbookqa/noropenbookqa_no_fact_nno",
    },
    "noropenbookqa_no_fact_nob": {
        "pretty_name": "OpenBookQA No-Fact (Bokmål)",
        "main_metric": "acc",
        "random_baseline": 0.25,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "world knowledge & reasoning",
        "path": "noropenbookqa/noropenbookqa_no_fact_nob",
    },
    "norquad": {
        "pretty_name": "NorQuAD",
        "main_metric": "f1",
        "random_baseline": 0.0,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "language understanding",
    },
    "nortruthfulqa_gen_nno": {
        "pretty_name": "TruthfulQA Gen (Nynorsk)",
        "main_metric": "rougeL_max",
        "random_baseline": 0.0,
        "max_performance": 100.0,
        "metric_scale": "percent",
        "category": "world knowledge & reasoning",
    },
    "nortruthfulqa_gen_nob": {
        "pretty_name": "TruthfulQA Gen (Bokmål)",
        "main_metric": "rougeL_max",
        "random_baseline": 0.0,
        "max_performance": 100.0,
        "metric_scale": "percent",
        "category": "world knowledge & reasoning",
    },
    "nortruthfulqa_mc_nno": {
        "pretty_name": "TruthfulQA MC (Nynorsk)",
        "main_metric": "acc",
        "random_baseline": 0.23311814890762259,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "world knowledge & reasoning",
    },
    "nortruthfulqa_mc_nob": {
        "pretty_name": "TruthfulQA MC (Bokmål)",
        "main_metric": "acc",
        "random_baseline": 0.23170337745132827,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "world knowledge & reasoning",
    },
    "nrk_quiz_qa_nno": {
        "pretty_name": "NRK Quiz QA (Nynorsk)",
        "main_metric": "acc",
        "random_baseline": 0.27884711779448623,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "world knowledge & reasoning",
    },
    "nrk_quiz_qa_nob": {
        "pretty_name": "NRK Quiz QA (Bokmål)",
        "main_metric": "acc",
        "random_baseline": 0.2836296296296296,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "world knowledge & reasoning",
    },
    "norrewrite_instruct": {
        "pretty_name": "NorRewrite Instruct",
        "main_metric": "bleu",
        "random_baseline": 0.0,
        "max_performance": 100.0,
        "metric_scale": "percent",
        "category": "generation & summarization",
    },
    "norsumm_nno": {
        "pretty_name": "NorSumm (Nynorsk)",
        "main_metric": "rougeL_max",
        "random_baseline": 0.0,
        "max_performance": 100.0,
        "metric_scale": "percent",
        "category": "generation & summarization",
    },
    "norsumm_nob": {
        "pretty_name": "NorSumm (Bokmål)",
        "main_metric": "rougeL_max",
        "random_baseline": 0.0,
        "max_performance": 100.0,
        "metric_scale": "percent",
        "category": "generation & summarization",
    },
    "norsummarize_instruct": {
        "pretty_name": "NorSummarize Instruct",
        "main_metric": "bleu",
        "random_baseline": 0.0,
        "max_performance": 100.0,
        "metric_scale": "percent",
        "category": "generation & summarization",
    },
    "tatoeba_eng_nno": {
        "pretty_name": "Tatoeba (EN→Nynorsk)",
        "main_metric": "bleu",
        "random_baseline": 0.0,
        "max_performance": 100.0,
        "metric_scale": "percent",
        "category": "translation",
    },
    "tatoeba_eng_nob": {
        "pretty_name": "Tatoeba (EN→Bokmål)",
        "main_metric": "bleu",
        "random_baseline": 0.0,
        "max_performance": 100.0,
        "metric_scale": "percent",
        "category": "translation",
    },
    "tatoeba_nno_eng": {
        "pretty_name": "Tatoeba (Nynorsk→EN)",
        "main_metric": "bleu",
        "random_baseline": 0.0,
        "max_performance": 100.0,
        "metric_scale": "percent",
        "category": "translation",
    },
    "tatoeba_nob_eng": {
        "pretty_name": "Tatoeba (Bokmål→EN)",
        "main_metric": "bleu",
        "random_baseline": 0.0,
        "max_performance": 100.0,
        "metric_scale": "percent",
        "category": "translation",
    },
    "tatoeba_nob_sme": {
        "pretty_name": "Tatoeba (Bokmål→Sámi)",
        "main_metric": "bleu",
        "random_baseline": 0.0,
        "max_performance": 100.0,
        "metric_scale": "percent",
        "category": "translation",
    },
    "tatoeba_sme_nob": {
        "pretty_name": "Tatoeba (Sámi→Bokmål)",
        "main_metric": "bleu",
        "random_baseline": 0.0,
        "max_performance": 100.0,
        "metric_scale": "percent",
        "category": "translation",
    },
    # ── French tasks ──
    "french_bench_arc_challenge": {
        "pretty_name": "French Bench ARC Challenge",
        "main_metric": "acc",
        "random_baseline": 0.25,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "world knowledge",
    },
    "french_bench_boolqa": {
        "pretty_name": "French Bench BoolQ",
        "main_metric": "acc",
        "random_baseline": 0.5,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "language understanding",
    },
    "french_bench_hellaswag": {
        "pretty_name": "French Bench HellaSwag",
        "main_metric": "acc_norm",
        "random_baseline": 0.25,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "language understanding",
    },
    "french_bench_multifquad": {
        "pretty_name": "French Bench MultiFQuAD",
        "main_metric": "exact",
        "random_baseline": 0.0,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "language understanding",
    },
    "french_bench_trivia": {
        "pretty_name": "French Bench Trivia",
        "main_metric": "exact",
        "random_baseline": 0.0,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "world knowledge",
    },
    "fquad": {
        "pretty_name": "FQuAD",
        "main_metric": "exact",
        "random_baseline": 0.0,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "language understanding",
    },
    "frabelebele": {
        "pretty_name": "Belebele (French)",
        "main_metric": "acc",
        "random_baseline": 0.25,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "language understanding",
    },
    "french_bench_grammar": {
        "pretty_name": "French Bench Grammar",
        "main_metric": "acc",
        "random_baseline": 0.25,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "linguistic knowledge",
    },
    "french_bench_reading": {
        "pretty_name": "French Bench Reading",
        "main_metric": "acc",
        "random_baseline": 0.25,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "language understanding",
    },
    "french_bench_vocabulary": {
        "pretty_name": "French Bench Vocabulary",
        "main_metric": "acc",
        "random_baseline": 0.25,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "linguistic knowledge",
    },
    "french_xnli": {
        "pretty_name": "XNLI (French)",
        "main_metric": "acc",
        "random_baseline": 1 / 3,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "language understanding",
    },
    "global_mmlu_french": {
        "pretty_name": "Global MMLU (French)",
        "main_metric": "acc",
        "random_baseline": 0.25,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "world knowledge",
    },
    "include_french": {
        "pretty_name": "IncludeFrench",
        "main_metric": "acc",
        "random_baseline": 0.25,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "world knowledge",
    },
    "topic_based_nli": {
        "pretty_name": "Topic-Based NLI",
        "main_metric": "acc",
        "random_baseline": 1 / 3,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "language understanding",
    },
    "wmt14-en-fr": {
        "pretty_name": "WMT14 (EN→FR)",
        "main_metric": "bleu",
        "random_baseline": 0.0,
        "max_performance": 100.0,
        "metric_scale": "percent",
        "category": "translation",
    },
    "wmt14-fr-en": {
        "pretty_name": "WMT14 (FR→EN)",
        "main_metric": "bleu",
        "random_baseline": 0.0,
        "max_performance": 100.0,
        "metric_scale": "percent",
        "category": "translation",
    },
    # Spanish tasks
    "cocoteros_es": {
        "pretty_name": "Cocoteros (Spanish)",
        "main_metric": "bleu",
        "random_baseline": 0.0,
        "max_performance": 100.0,
        "metric_scale": "percent",
        "category": "generation & summarization",
    },
    "copa_es": {
        "pretty_name": "COPA (Spanish)",
        "main_metric": "acc",
        "random_baseline": 0.25,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "language understanding",
    },
    "escola": {
        "pretty_name": "EsCoLA",
        "main_metric": "mcc",
        "random_baseline": 0.0,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "linguistic knowledge",
    },
    "flores_en-es": {
        "pretty_name": "FLORES (EN→ES)",
        "main_metric": "bleu",
        "random_baseline": 0.0,
        "max_performance": 100.0,
        "metric_scale": "percent",
        "category": "translation",
    },
    "global_mmlu_spanish": {
        "pretty_name": "Global MMLU (Spanish)",
        "main_metric": "acc",
        "random_baseline": 0.25,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "world knowledge",
    },
    "include_spanish": {
        "pretty_name": "IncludeSpanish",
        "main_metric": "acc",
        "random_baseline": 0.25,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "world knowledge",
    },
    "mgsm_direct_es": {
        "pretty_name": "MGSM Direct (Spanish)",
        "main_metric": "exact_match",
        "random_baseline": 0.0,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "math reasoning",
    },
    "openbookqa_es": {
        "pretty_name": "OpenBookQA (Spanish)",
        "main_metric": "acc",
        "random_baseline": 0.25,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "world knowledge",
    },
    "paws_es": {
        "pretty_name": "PAWS (Spanish)",
        "main_metric": "acc",
        "random_baseline": 0.5,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "language understanding",
    },
    "spabelebele": {
        "pretty_name": "Belebele (Spanish)",
        "main_metric": "acc",
        "random_baseline": 0.25,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "language understanding",
    },
    "veritasqa_es_gen": {
        "pretty_name": "TruthfulQA Gen (Spanish)",
        "main_metric": "bleu_max",
        "random_baseline": 0.0,
        "max_performance": 100.0,
        "metric_scale": "percent",
        "category": "world knowledge",
    },
    "veritasqa_es_mc1": {
        "pretty_name": "TruthfulQA MC1 (Spanish)",
        "main_metric": "acc",
        "random_baseline": 0.243,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "world knowledge",
    },
    "veritasqa_es_mc2": {
        "pretty_name": "TruthfulQA MC2 (Spanish)",
        "main_metric": "acc",
        "random_baseline": 0.154,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "world knowledge",
    },
    "xnli_es": {
        "pretty_name": "XNLI (Spanish)",
        "main_metric": "acc",
        "random_baseline": 1 / 3,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "language understanding",
    },
    "xquad_es": {
        "pretty_name": "XQuAD (Spanish)",
        "main_metric": "exact_match",
        "random_baseline": 0.0,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "language understanding",
    },
    "xstorycloze_es": {
        "pretty_name": "XStoryCloze (Spanish)",
        "main_metric": "acc",
        "random_baseline": 0.5,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "language understanding",
    },
}

# Model display names and colors (keyed by base model name without shot/checkpoints suffix)
MODEL_CONFIG = {
    "hplt2": {
        "display_name": "HPLT v2",
        "color": "#6366f1",
    },
    "hplt3": {
        "display_name": "HPLT v3",
        "color": "#8b5cf6",
    },
    "opus": {
        "display_name": "OPUS",
        "color": "#f43f5e",
    },
    "tower9b": {
        "display_name": "Tower 9B",
        "color": "#10b981",
    },
    "tower72b": {
        "display_name": "Tower 72B",
        "color": "#f59e0b",
    },
}


def parse_model_dir(name):
    """Parse model directory name to (base_model, shot_setting).

    E.g. 'hplt2_0shot_checkpoints' -> ('hplt2', '0')
         'tower72B_5shot_checkpoints' -> ('tower72b', '5')
    Returns (None, None) if the name doesn't match the expected pattern.
    """
    import re

    m = re.match(r"^(.+?)_(\d+)shot_checkpoints$", name, re.IGNORECASE)
    if m:
        return m.group(1).lower(), m.group(2)
    return None, None

# Metrics to exclude globally. `bypass` is the lm-eval placeholder used when
# a task ran without computing a metric (the real score is patched in later).
EXCLUDED_METRICS = {"bleu_diff", "rouge1_diff", "rouge2_diff", "rougeL_diff", "bypass"}

# Per-benchmark metric exclusions. ask_gec's `exact_match` is the lm-eval
# placeholder; the real metric is ERRANT F0.5 (`errant`), merged in by
# merge_errant_scores.py.
EXCLUDED_METRICS_PER_BENCHMARK = {
    "ask_gec": {"exact_match"},
    "noreval_multiblimp": {"acc_norm"},
}


def find_latest_results_json(directory):
    """Find the newest results_*.json or results.json under directory."""
    # Try results.json first (simple format)
    simple = os.path.join(directory, "results.json")
    if os.path.isfile(simple):
        return simple
    # Try nested results_*.json
    pattern = os.path.join(directory, "**", "results_*.json")
    files = glob.glob(pattern, recursive=True)
    if not files:
        return None
    files.sort(key=lambda f: os.path.basename(f))
    return files[-1]


def _get_stderr(task_results, metric_name, metric_suffix, n_samples, metric_scale):
    """Get stderr for a metric from task results, estimating if missing."""
    stderr_key = f"{metric_name}_stderr,{metric_suffix}"
    se = task_results.get(stderr_key)
    if isinstance(se, (int, float)):
        return se
    if n_samples and n_samples > 1:
        val_key = f"{metric_name},{metric_suffix}"
        val = task_results.get(val_key)
        if isinstance(val, (int, float)):
            if metric_scale == "percent":
                p = max(0.0, min(1.0, val / 100.0))
                if 0 < p < 1:
                    return math.sqrt(p * (1 - p) / n_samples) * 100
            else:
                p = max(0.0, min(1.0, val))
                if 0 < p < 1:
                    return math.sqrt(p * (1 - p) / n_samples)
    return None


def extract_benchmark_scores(results_json_path, benchmark_name, task_config_entry=None):
    """Extract metrics from a single results JSON (one partition).

    Returns dict {metric_name: (value, stderr)} or None.
    """
    with open(results_json_path) as f:
        data = json.load(f)

    results = data.get("results", {})
    n_samples_dict = data.get("n-samples", {})
    metric_scale = (
        task_config_entry.get("metric_scale", "unit") if task_config_entry else "unit"
    )
    bench_exclusions = EXCLUDED_METRICS | EXCLUDED_METRICS_PER_BENCHMARK.get(
        benchmark_name, set()
    )

    metrics = {}
    for task_key, task_results in results.items():
        # Match the benchmark name with partition suffix
        if not (
            task_key == benchmark_name
            or task_key.startswith(f"{benchmark_name}_p")
        ):
            # Also check for group-level results (global_mmlu)
            if not task_key.startswith(f"{benchmark_name}_"):
                continue
            # Skip subtask results (e.g., global_mmlu_french_business_p0)
            # Only keep the main group aggregate
            continue

        ns_entry = n_samples_dict.get(task_key, {})
        n_samples = ns_entry.get("effective") or ns_entry.get("original")

        for key, val in task_results.items():
            if key == "alias":
                continue
            # Handle both ",none" and other suffixes like ",remove_whitespace"
            if "," not in key:
                continue
            if "_stderr," in key:
                continue
            metric_name, metric_suffix = key.rsplit(",", 1)
            if metric_name in bench_exclusions:
                continue
            if isinstance(val, (int, float)):
                se = _get_stderr(
                    task_results, metric_name, metric_suffix, n_samples, metric_scale
                )
                metrics[metric_name] = (val, se)

    # For group tasks (like global_mmlu), also check group-level results
    groups = data.get("groups", {})
    for group_key, group_results in groups.items():
        if not (
            group_key == benchmark_name
            or group_key.startswith(f"{benchmark_name}_p")
        ):
            continue
        for key, val in results.get(group_key, {}).items():
            if key == "alias":
                continue
            if "," not in key:
                continue
            if "_stderr," in key:
                continue
            metric_name, metric_suffix = key.rsplit(",", 1)
            if metric_name in bench_exclusions:
                continue
            if isinstance(val, (int, float)):
                ns_entry = n_samples_dict.get(group_key, {})
                n_samples = ns_entry.get("effective") or ns_entry.get("original")
                se = _get_stderr(
                    results.get(group_key, {}),
                    metric_name,
                    metric_suffix,
                    n_samples,
                    metric_scale,
                )
                metrics[metric_name] = (val, se)

    return metrics if metrics else None


def process_multiblimp(bench_path):
    """Aggregate noreval_multiblimp's per-phenomenon subtask result dirs.

    Each `noreval_multiblimp_<code>/` subdir holds a single results JSON.
    Returns a single (value, stderr) dict mimicking one prompt-variant for
    `aggregate_partitions` — accuracy is micro-averaged (sample-weighted)
    across all subtasks, matching how `noreval_multiblimp` is reported in
    noreval-stats.
    """
    if not os.path.isdir(bench_path):
        return None
    sub_dirs = sorted(
        d for d in os.listdir(bench_path)
        if os.path.isdir(os.path.join(bench_path, d))
        and d.startswith("noreval_multiblimp_")
    )
    if not sub_dirs:
        return None

    total_correct = 0.0
    total_n = 0
    for sub in sub_dirs:
        sub_path = os.path.join(bench_path, sub)
        results_file = find_latest_results_json(sub_path)
        if results_file is None:
            continue
        with open(results_file) as f:
            data = json.load(f)
        task_results = data.get("results", {}).get(sub)
        if not task_results:
            continue
        acc = task_results.get("acc,none")
        n = (
            data.get("n-samples", {}).get(sub, {}).get("effective")
            or data.get("n-samples", {}).get(sub, {}).get("original")
        )
        if acc is None or not n:
            continue
        total_correct += acc * n
        total_n += n

    if total_n == 0:
        return None

    micro_acc = total_correct / total_n
    # Stderr from binomial proportion on the pooled set
    se = math.sqrt(micro_acc * (1 - micro_acc) / total_n) if 0 < micro_acc < 1 else 0.0
    return {"acc": (micro_acc, se)}


def aggregate_partitions(partition_metrics_list):
    """Aggregate metrics across partitions (p0, p1, p2) like prompt variants.

    partition_metrics_list: list of {metric_name: (value, stderr)}
    Returns dict {metric_name: {"max": ..., "mean": ..., ...}}
    """
    # Collect values per metric across partitions
    metric_values = {}
    for pmetrics in partition_metrics_list:
        if pmetrics is None:
            continue
        for metric_name, (val, se) in pmetrics.items():
            if metric_name not in metric_values:
                metric_values[metric_name] = []
            metric_values[metric_name].append((val, se))

    if not metric_values:
        return None

    out = {}
    for metric_name, pairs in metric_values.items():
        values = [v for v, _ in pairs]
        stderrs = [se for _, se in pairs]

        entry = {
            "max": round(max(values), 6),
            "mean": round(statistics.mean(values), 6),
            "median": round(statistics.median(values), 6),
            "min": round(min(values), 6),
            "first": round(values[0], 6),
        }

        max_idx = values.index(max(values))
        entry["max_prompt_idx"] = max_idx
        if stderrs[max_idx] is not None:
            entry["max_stderr"] = round(stderrs[max_idx], 6)

        min_idx = values.index(min(values))
        if stderrs[min_idx] is not None:
            entry["min_stderr"] = round(stderrs[min_idx], 6)

        if stderrs[0] is not None:
            entry["first_stderr"] = round(stderrs[0], 6)

        if all(se is not None for se in stderrs):
            n = len(stderrs)
            mean_se = math.sqrt(sum(se**2 for se in stderrs)) / n
            entry["mean_stderr"] = round(mean_se, 6)

        med = statistics.median(values)
        closest_idx = min(range(len(values)), key=lambda i: abs(values[i] - med))
        if stderrs[closest_idx] is not None:
            entry["median_stderr"] = round(stderrs[closest_idx], 6)

        entry["n_prompts"] = len(values)
        if len(values) >= 2:
            entry["prompt_sd"] = round(statistics.stdev(values), 6)
            med_val = statistics.median(values)
            entry["prompt_mad"] = round(
                statistics.median([abs(v - med_val) for v in values]), 6
            )
        else:
            entry["prompt_sd"] = 0.0
            entry["prompt_mad"] = 0.0

        out[metric_name] = entry
    return out


def process_checkpoint(ckpt_path, task_configs, shot="0"):
    """Process a single checkpoint directory.

    Returns {benchmark: {shot: {metric: {...}}}}
    """
    scores = {}
    for benchmark, config in task_configs.items():
        rel_path = config.get("path", benchmark)
        bench_path = os.path.join(ckpt_path, rel_path)
        if not os.path.isdir(bench_path):
            continue

        partition_results = []

        if config.get("aggregator") == "multiblimp":
            # Custom aggregation: each phenomenon subdir is a sub-eval, not a
            # prompt variant. Pool into one micro-averaged score that becomes a
            # single prompt-variant entry.
            agg_metrics = process_multiblimp(bench_path)
            if agg_metrics:
                partition_results.append(agg_metrics)
        else:
            # Collect partitions (p0, p1, p2, ...)
            partitions = sorted(
                [
                    d
                    for d in os.listdir(bench_path)
                    if os.path.isdir(os.path.join(bench_path, d))
                    and d.startswith("p")
                    and d[1:].isdigit()
                ]
            )

            if partitions:
                # Each partition is like a prompt variant
                for part in partitions:
                    part_path = os.path.join(bench_path, part)
                    results_file = find_latest_results_json(part_path)
                    if results_file is None:
                        continue
                    metrics = extract_benchmark_scores(results_file, benchmark, config)
                    partition_results.append(metrics)
            else:
                # No partition dirs — look for results directly under bench_path
                results_file = find_latest_results_json(bench_path)
                if results_file is not None:
                    metrics = extract_benchmark_scores(
                        results_file, benchmark, config
                    )
                    if metrics:
                        partition_results.append(metrics)

        if not partition_results:
            continue

        agg = aggregate_partitions(partition_results)
        if agg is not None:
            scores[benchmark] = {shot: agg}

    return scores


def discover_language_tasks(lang_dir):
    """Discover which configured tasks have data in this language.

    Returns the set of TASK_CONFIG keys whose `path` (or default `<key>`) exists
    under any model's checkpoint dir, plus any unrecognized top-level dirs (so
    the caller can warn). Honors the optional `path` field for tasks whose
    results live in a non-default subdirectory (e.g. noropenbookqa subtasks).
    """
    found = set()
    unknown = set()
    configured_paths = {
        cfg.get("path", key): key for key, cfg in TASK_CONFIG.items()
    }
    nested_parents = {p.split("/", 1)[0] for p in configured_paths if "/" in p}

    for model_dir in os.listdir(lang_dir):
        model_path = os.path.join(lang_dir, model_dir)
        if not os.path.isdir(model_path) or model_dir.startswith("."):
            continue
        for ckpt in os.listdir(model_path):
            ckpt_path = os.path.join(model_path, ckpt)
            if not os.path.isdir(ckpt_path) or ckpt.startswith("."):
                continue
            for top in os.listdir(ckpt_path):
                top_path = os.path.join(ckpt_path, top)
                if not os.path.isdir(top_path) or top.startswith("."):
                    continue
                # Direct match
                if top in configured_paths:
                    found.add(configured_paths[top])
                    continue
                # Nested match (e.g. noropenbookqa/<subtask>)
                if top in nested_parents:
                    for sub in os.listdir(top_path):
                        sub_full = f"{top}/{sub}"
                        if sub_full in configured_paths and os.path.isdir(
                            os.path.join(top_path, sub)
                        ):
                            found.add(configured_paths[sub_full])
                    continue
                unknown.add(top)
    return found, unknown


def main():
    os.makedirs(OUTPUT_FILE.parent, exist_ok=True)

    output = {"languages": {}}

    for lang_name in sorted(os.listdir(RESULTS_DIR)):
        lang_dir = RESULTS_DIR / lang_name
        if not lang_dir.is_dir() or lang_name.startswith("."):
            continue

        print(f"\n=== Processing language: {lang_name} ===")

        # Discover tasks and build metrics_setup for this language
        lang_tasks, unknown_tasks = discover_language_tasks(str(lang_dir))
        task_configs = {task: TASK_CONFIG[task] for task in sorted(lang_tasks)}
        discovered_metrics = {}
        for unknown in sorted(unknown_tasks):
            print(f"  WARNING: No config for task '{unknown}', skipping")

        lang_data = {
            "metrics_setup": {},
            "models": {},
        }

        # Group model directories by base model name
        # e.g. hplt2_0shot_checkpoints + hplt2_5shot_checkpoints -> hplt2
        model_groups = {}  # base_model -> [(model_dir_name, shot, path), ...]
        for model_dir in sorted(os.listdir(lang_dir)):
            model_path = lang_dir / model_dir
            if not model_path.is_dir() or model_dir.startswith("."):
                continue
            base_model, shot = parse_model_dir(model_dir)
            if base_model is None:
                print(f"  WARNING: Cannot parse model dir '{model_dir}', skipping")
                continue
            if base_model not in model_groups:
                model_groups[base_model] = []
            model_groups[base_model].append((model_dir, shot, model_path))

        # Process each base model (merging shot variants)
        for base_model in sorted(model_groups):
            model_cfg = MODEL_CONFIG.get(base_model, {})
            display_name = model_cfg.get("display_name", base_model)
            color = model_cfg.get("color", "#6366f1")

            print(f"  Model: {base_model} ({display_name})")

            progress = {}
            for model_dir, shot, model_path in model_groups[base_model]:
                print(f"    Shot setting: {shot}-shot ({model_dir})")

                for ckpt_name in sorted(os.listdir(model_path)):
                    ckpt_path = model_path / ckpt_name
                    if not ckpt_path.is_dir() or ckpt_name.startswith("."):
                        continue

                    tokens_b = parse_checkpoint_name(ckpt_name)
                    if tokens_b is None:
                        continue

                    scores = process_checkpoint(
                        str(ckpt_path), task_configs, shot=shot
                    )
                    if scores:
                        # Merge into progress: each benchmark gets its shot data
                        if tokens_b not in progress:
                            progress[tokens_b] = {}
                        for bench, shot_data in scores.items():
                            if bench not in progress[tokens_b]:
                                progress[tokens_b][bench] = {}
                            progress[tokens_b][bench].update(shot_data)
                            # Track discovered metrics
                            for s, metric_data in shot_data.items():
                                if bench not in discovered_metrics:
                                    discovered_metrics[bench] = set()
                                discovered_metrics[bench].update(
                                    metric_data.keys()
                                )

                    print(
                        f"      Checkpoint {ckpt_name}: {len(scores)} tasks"
                    )

            lang_data["models"][base_model] = {
                "display_name": display_name,
                "color": color,
                "progress": progress,
            }

        # Build metrics_setup for this language
        for task, config in task_configs.items():
            disc = discovered_metrics.get(task, set())
            if not disc:
                continue
            main_metric = config["main_metric"]
            max_perf = 100.0 if config.get("metric_scale") == "percent" else 1.0
            base_metrics = sorted(disc - {main_metric})
            available_metrics = (
                ([main_metric] if main_metric in disc else []) + base_metrics
            )

            lang_data["metrics_setup"][task] = {
                "pretty_name": config["pretty_name"],
                "main_metric": main_metric,
                "random_baseline": config["random_baseline"],
                "max_performance": max_perf,
                "category": config.get("category", "uncategorized"),
                "metric_scale": config.get("metric_scale", "unit"),
                "available_metrics": available_metrics,
            }

        # Resolve "main" checkpoint: map it to the max numeric token value
        # across all models in this language (it's the final checkpoint)
        max_tokens = 0
        for md in lang_data["models"].values():
            for token_key in md["progress"]:
                if isinstance(token_key, (int, float)) and token_key != "main":
                    max_tokens = max(max_tokens, token_key)
        for md in lang_data["models"].values():
            if "main" in md["progress"] and max_tokens > 0:
                md["progress"][max_tokens] = md["progress"].pop("main")

        output["languages"][lang_name] = lang_data

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, ensure_ascii=False)

    size_kb = os.path.getsize(OUTPUT_FILE) / 1024
    print(f"\nWritten {OUTPUT_FILE} ({size_kb:.1f} KB)")
    for lang, ld in output["languages"].items():
        models = list(ld["models"].keys())
        tasks = list(ld["metrics_setup"].keys())
        print(f"  {lang}: {len(models)} models, {len(tasks)} tasks")
        for m in models:
            steps = sorted(ld["models"][m]["progress"].keys(), key=lambda x: (isinstance(x, str), x))
            print(f"    {m}: {len(steps)} checkpoints")


if __name__ == "__main__":
    main()
