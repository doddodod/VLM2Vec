#!/usr/bin/env python3
"""
Preprocess negative samples for SGG training.
- Load predicates from vocab.txt
- Compute TopK nearest neighbors for predicates
- Compute same-image negatives from training data

Usage:
    python construct_neg.py \
        --vocab_path /public/home/wangby2025/plusLab/data/vg/vocab.txt \
        --train_data_path /public/home/wangby2025/plusLab/data/vg/train.json \
        --output_dir /public/home/wangby2025/plusLab/data/vg/processed \
        --topk_neighbors 10
"""

import json
import argparse
from pathlib import Path
import numpy as np
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


def load_predicate_vocab(vocab_path):
    """
    Load predicate vocabulary from vocab.txt (comma-separated format)
    
    Expected format:
        above, across, against, along, and, at, ...
    """
    print(f"Loading predicate vocabulary from {vocab_path}...")
    
    with open(vocab_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    # Split by comma and clean up
    predicates = [
        pred.strip() 
        for pred in content.split(',')
        if pred.strip()
    ]
    
    print(f"✓ Loaded {len(predicates)} predicates")
    
    # Print vocabulary
    print("\nPredicate vocabulary:")
    for i, pred in enumerate(predicates[:10]):
        print(f"  {i:2d}. {pred}")
    if len(predicates) > 10:
        print(f"  ...")
        for i in range(len(predicates) - 3, len(predicates)):
            print(f"  {i:2d}. {predicates[i]}")
    
    return predicates


def load_training_data(train_data_path):
    """Load training data JSON"""
    print(f"\nLoading training data from {train_data_path}...")
    
    with open(train_data_path, 'r') as f:
        data = json.load(f)
    
    print(f"✓ Loaded {len(data)} training samples")
    
    # Show statistics
    predicate_counts = defaultdict(int)
    for sample in data:
        predicate_counts[sample['predicate']] += 1
    
    print(f"\nPredicate distribution in training data:")
    print(f"  Unique predicates: {len(predicate_counts)}")
    print(f"\n  Top 10 most frequent predicates:")
    for pred, count in sorted(predicate_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"    {pred}: {count}")
    
    return data


def precompute_predicate_neighbors(predicates, output_path, k=10):
    """
    Compute TopK nearest neighbors for each predicate using semantic similarity
    """
    print(f"\n{'='*70}")
    print(f"Computing nearest neighbors for {len(predicates)} predicates")
    print(f"{'='*70}")
    
    # Load sentence transformer
    print("\nLoading sentence transformer model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Encode all predicates
    print("Encoding predicates with semantic embeddings...")
    embeddings = model.encode(
        predicates, 
        show_progress_bar=True, 
        batch_size=32,
        convert_to_numpy=True
    )
    
    # Compute similarity matrix
    print("Computing pairwise cosine similarity matrix...")
    sim_matrix = cosine_similarity(embeddings)
    
    # Find TopK for each predicate
    print(f"Finding top-{k} nearest neighbors for each predicate...")
    neighbor_dict = {}
    
    for idx, predicate in enumerate(tqdm(predicates, desc="Processing")):
        similarities = sim_matrix[idx].copy()
        similarities[idx] = -np.inf  # Exclude self
        
        # Get TopK indices (highest similarity)
        topk_indices = np.argsort(similarities)[-k:][::-1]
        topk_predicates = [predicates[i] for i in topk_indices]
        topk_scores = [float(similarities[i]) for i in topk_indices]
        
        neighbor_dict[predicate] = {
            'neighbors': topk_predicates,
            'scores': topk_scores
        }
    
    # Save to JSON
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(neighbor_dict, f, indent=2)
    
    file_size_kb = output_path.stat().st_size / 1024
    print(f"\n✓ Saved predicate neighbors to {output_path} ({file_size_kb:.2f} KB)")
    
    # Print examples
    print("\n" + "="*70)
    print("Example nearest neighbors:")
    print("="*70)
    
    example_predicates = ['on', 'wearing', 'has', 'near', 'holding']
    for pred in example_predicates:
        if pred in neighbor_dict:
            print(f"\n'{pred}':")
            for neighbor, score in zip(
                neighbor_dict[pred]['neighbors'][:5],
                neighbor_dict[pred]['scores'][:5]
            ):
                print(f"  {score:.3f} → {neighbor}")
    
    return neighbor_dict


def precompute_same_image_negatives(data, output_path):
    """
    For each sample, store predicates from other bbox pairs in the same image
    """
    print(f"\n{'='*70}")
    print(f"Computing same-image negatives for {len(data)} samples")
    print(f"{'='*70}")
    
    # Group samples by image_id
    print("\nStep 1: Building image index...")
    image_to_samples = defaultdict(list)
    
    for sample in tqdm(data, desc="Grouping by image_id"):
        img_id = sample['image_id']
        
        # Create unique bbox pair identifier
        subj_bbox = tuple(sample['subject']['bbox'])
        obj_bbox = tuple(sample['object']['bbox'])
        bbox_pair = (subj_bbox, obj_bbox)
        
        image_to_samples[img_id].append({
            'id': sample['id'],
            'predicate': sample['predicate'],
            'bbox_pair': bbox_pair
        })
    
    print(f"  ✓ Found {len(image_to_samples)} unique images")
    
    # Calculate statistics
    samples_per_image = [len(samples) for samples in image_to_samples.values()]
    print(f"  ✓ Samples per image:")
    print(f"      Mean: {np.mean(samples_per_image):.2f}")
    print(f"      Median: {np.median(samples_per_image):.0f}")
    print(f"      Max: {max(samples_per_image)}")
    print(f"      Min: {min(samples_per_image)}")
    
    # Compute same-image negatives for each sample
    print("\nStep 2: Computing same-image negatives for each sample...")
    same_image_negs = {}
    
    for sample in tqdm(data, desc="Processing samples"):
        sample_id = sample['id']
        img_id = sample['image_id']
        
        subj_bbox = tuple(sample['subject']['bbox'])
        obj_bbox = tuple(sample['object']['bbox'])
        current_bbox_pair = (subj_bbox, obj_bbox)
        
        # Get predicates from OTHER bbox pairs in the same image
        negs = [
            s['predicate']
            for s in image_to_samples[img_id]
            if s['bbox_pair'] != current_bbox_pair and s['id'] != sample_id
        ]
        
        # Store using sample id as key (string for JSON compatibility)
        same_image_negs[str(sample_id)] = negs
    
    # Save to JSON
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(same_image_negs, f)
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\n✓ Saved same-image negatives to {output_path} ({file_size_mb:.2f} MB)")
    
    # Statistics
    neg_counts = [len(negs) for negs in same_image_negs.values()]
    num_samples_with_no_negs = sum(1 for c in neg_counts if c == 0)
    num_samples_with_negs = len(neg_counts) - num_samples_with_no_negs
    
    print("\n" + "="*70)
    print("Same-image negative statistics:")
    print("="*70)
    print(f"  Samples with 0 negatives:  {num_samples_with_no_negs:6d} ({100 * num_samples_with_no_negs / len(neg_counts):5.1f}%)")
    print(f"  Samples with ≥1 negatives: {num_samples_with_negs:6d} ({100 * num_samples_with_negs / len(neg_counts):5.1f}%)")
    print(f"  Average negatives/sample:  {np.mean(neg_counts):6.2f}")
    print(f"  Median negatives/sample:   {np.median(neg_counts):6.0f}")
    print(f"  Max negatives/sample:      {max(neg_counts):6d}")
    
    # Show examples with same-image negatives
    print("\n" + "="*70)
    print("Example samples with same-image negatives:")
    print("="*70)
    
    examples_with_negs = [
        (sid, negs) for sid, negs in same_image_negs.items() if len(negs) > 0
    ][:5]
    
    for sample_id_str, negs in examples_with_negs:
        sample = data[int(sample_id_str)]
        print(f"\nSample ID: {sample_id_str}, Image ID: {sample['image_id']}")
        print(f"  Subject: {sample['subject']['class_name']}")
        print(f"  Object:  {sample['object']['class_name']}")
        print(f"  GT Predicate: '{sample['predicate']}'")
        print(f"  Same-image negatives ({len(negs)}): {negs[:8]}")
        if len(negs) > 8:
            print(f"    ... and {len(negs) - 8} more")
    
    return same_image_negs


def save_predicate_vocab_json(predicates, output_path):
    """Save predicate vocabulary as JSON for easy loading during training"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    vocab = {
        'predicates': predicates,
        'vocab_size': len(predicates),
        'predicate_to_idx': {pred: idx for idx, pred in enumerate(predicates)},
        'idx_to_predicate': {idx: pred for idx, pred in enumerate(predicates)}
    }
    
    with open(output_path, 'w') as f:
        json.dump(vocab, f, indent=2)
    
    file_size_kb = output_path.stat().st_size / 1024
    print(f"✓ Saved predicate vocabulary to {output_path} ({file_size_kb:.2f} KB)")


def validate_preprocessing(output_dir, predicates, num_samples):
    """Verify all files were created correctly"""
    print(f"\n{'='*70}")
    print("Validating preprocessed files")
    print("="*70)
    
    output_dir = Path(output_dir)
    all_valid = True
    
    # Check files exist and validate content
    files_to_check = {
        'predicate_vocab.json': lambda x: len(x['predicates']) == len(predicates),
        'predicate_neighbors.json': lambda x: len(x) == len(predicates),
        'same_image_negatives.json': lambda x: len(x) == num_samples
    }
    
    for filename, validator in files_to_check.items():
        filepath = output_dir / filename
        
        if not filepath.exists():
            print(f"✗ Missing file: {filename}")
            all_valid = False
            continue
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        if not validator(data):
            print(f"✗ Invalid content: {filename}")
            all_valid = False
            continue
        
        file_size = filepath.stat().st_size
        if file_size < 1024:
            size_str = f"{file_size} B"
        elif file_size < 1024 * 1024:
            size_str = f"{file_size / 1024:.2f} KB"
        else:
            size_str = f"{file_size / (1024 * 1024):.2f} MB"
        
        print(f"✓ {filename:30s} {size_str:>12s}")
    
    if all_valid:
        print("\n✓ All files validated successfully!")
    else:
        print("\n✗ Validation failed!")
    
    return all_valid


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess negative samples for SGG training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--vocab_path', type=str, required=True,
                        help='Path to vocab.txt containing predicate list')
    parser.add_argument('--train_data_path', type=str, required=True,
                        help='Path to training data JSON file')
    parser.add_argument('--output_dir', type=str, default='data/processed',
                        help='Output directory for processed files')
    parser.add_argument('--topk_neighbors', type=int, default=10,
                        help='Number of nearest neighbors to compute per predicate')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("SGG Negative Sampling Preprocessing")
    print("="*70)
    print(f"\nVocabulary:    {args.vocab_path}")
    print(f"Training data: {args.train_data_path}")
    print(f"Output dir:    {args.output_dir}")
    print(f"TopK:          {args.topk_neighbors}\n")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load predicate vocabulary
    predicates = load_predicate_vocab(args.vocab_path)
    
    # Step 2: Load training data
    train_data = load_training_data(args.train_data_path)
    
    # Step 3: Save vocabulary as JSON
    print(f"\n{'='*70}")
    print("Saving predicate vocabulary as JSON")
    print("="*70)
    vocab_json_path = output_dir / 'predicate_vocab.json'
    save_predicate_vocab_json(predicates, vocab_json_path)
    
    # Step 4: Compute predicate nearest neighbors
    neighbors_path = output_dir / 'predicate_neighbors.json'
    precompute_predicate_neighbors(predicates, neighbors_path, k=args.topk_neighbors)
    
    # Step 5: Compute same-image negatives
    same_image_path = output_dir / 'same_image_negatives.json'
    precompute_same_image_negatives(train_data, same_image_path)
    
    # Step 6: Validate all outputs
    validate_preprocessing(output_dir, predicates, len(train_data))
    
    # Final summary
    print("\n" + "="*70)
    print("✓ PREPROCESSING COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print(f"  1. {vocab_json_path}")
    print(f"  2. {neighbors_path}")
    print(f"  3. {same_image_path}")
    print("\nNext steps:")
    print(f"  python train.py --processed_dir {args.output_dir}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()