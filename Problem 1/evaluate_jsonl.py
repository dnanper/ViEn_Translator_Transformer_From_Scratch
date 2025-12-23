"""
Evaluate translation results from JSONL file
Compare predictions with ground truth using sacreBleu
"""

import json
import sys
import os
from pathlib import Path
from sacrebleu import corpus_bleu, BLEU


def load_jsonl(jsonl_path):
    """Load predictions from JSONL file"""
    predictions = []
    sources = []
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            sources.append(data['source'])
            predictions.append(data['translation'])
    
    return sources, predictions


def load_references(ref_path):
    """Load reference translations from text file"""
    with open(ref_path, 'r', encoding='utf-8') as f:
        references = [line.strip() for line in f]
    
    return references


def evaluate_bleu(predictions, references):
    """
    Calculate BLEU scores using sacreBleu
    
    Args:
        predictions: List of predicted translations (strings)
        references: List of reference translations (strings)
    
    Returns:
        Dictionary with BLEU scores
    """
    assert len(predictions) == len(references), \
        f"Predictions ({len(predictions)}) and references ({len(references)}) must have same length"
    
    results = {}
    
    print("\nCalculating sacreBleu scores...")
    print(f"Total examples: {len(predictions)}")
    
    # sacreBleu expects references as list of lists (for multiple references per example)
    # We have one reference per example
    refs = [references]  # Wrap in list for sacreBleu format
    
    # Calculate BLEU for different n-grams using BLEU class
    for n in range(1, 5):
        try:
            # Use BLEU class with max_ngram_order parameter
            bleu_metric = BLEU(
                max_ngram_order=n,
                smooth_method='exp',
                lowercase=False,
                tokenize='13a',
                effective_order=True
            )
            bleu = bleu_metric.corpus_score(predictions, refs)
            
            results[f'bleu-{n}'] = bleu.score
            print(f"  BLEU-{n}: {bleu.score:.2f}")
        except Exception as e:
            print(f"  BLEU-{n}: Error - {e}")
            results[f'bleu-{n}'] = 0.0
    
    # Show signature for reproducibility
    bleu_4 = corpus_bleu(predictions, refs, tokenize='13a')
    print(f"\nsacreBleu signature: {bleu_4.format()}")
    
    return results


def show_examples(sources, predictions, references, num_examples=5):
    """Show translation examples"""
    print("\n" + "=" * 80)
    print("Translation Examples:")
    print("=" * 80)
    
    for i in range(min(num_examples, len(sources))):
        print(f"\nExample {i+1}:")
        print(f"  Source:      {sources[i][:100]}...")
        print(f"  Reference:   {references[i][:100]}...")
        print(f"  Translation: {predictions[i][:100]}...")


def main():
    # Get project root directory
    PROJECT_ROOT = Path(__file__).resolve().parent
    
    # Paths
    JSONL_PATH = PROJECT_ROOT / "data" / "processed" / "ep13_test_predict.jsonl"
    REFERENCE_PATH = PROJECT_ROOT / "data" / "processed" / "test.vi"
    
    print("=" * 80)
    print("BLEU Score Evaluation")
    print("=" * 80)
    print(f"Predictions: {JSONL_PATH}")
    print(f"References:  {REFERENCE_PATH}")
    print("=" * 80)
    
    # Check if files exist
    if not JSONL_PATH.exists():
        print(f"\n❌ Error: Predictions file not found: {JSONL_PATH}")
        return
    
    if not REFERENCE_PATH.exists():
        print(f"\n❌ Error: Reference file not found: {REFERENCE_PATH}")
        return
    
    # Load data
    print("\nLoading predictions...")
    sources, predictions = load_jsonl(JSONL_PATH)
    print(f"  Loaded {len(predictions):,} predictions")
    
    print("\nLoading references...")
    references = load_references(REFERENCE_PATH)
    print(f"  Loaded {len(references):,} references")
    
    # Check lengths match
    if len(predictions) != len(references):
        print(f"\n⚠️  Warning: Predictions ({len(predictions)}) and references ({len(references)}) have different lengths")
        min_len = min(len(predictions), len(references))
        print(f"   Using first {min_len} samples for evaluation")
        predictions = predictions[:min_len]
        references = references[:min_len]
        sources = sources[:min_len]
    
    # Show examples
    show_examples(sources, predictions, references, num_examples=5)
    
    # Evaluate
    print("\n" + "=" * 80)
    print("BLEU Score Results:")
    print("=" * 80)
    
    results = evaluate_bleu(predictions, references)
    
    print("\n" + "=" * 80)
    print("Summary:")
    print("=" * 80)
    print(f"Total samples:  {len(predictions):,}")
    print(f"BLEU-1:         {results['bleu-1']:.2f}%")
    print(f"BLEU-2:         {results['bleu-2']:.2f}%")
    print(f"BLEU-3:         {results['bleu-3']:.2f}%")
    print(f"BLEU-4:         {results['bleu-4']:.2f}%")
    print("=" * 80)
    
    # Save results
    results_path = JSONL_PATH.parent / "ep13_evaluation_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump({
            'num_samples': len(predictions),
            'bleu_scores': results,
            'prediction_file': str(JSONL_PATH),
            'reference_file': str(REFERENCE_PATH)
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Results saved to: {results_path}")


if __name__ == '__main__':
    main()
