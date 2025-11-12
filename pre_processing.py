import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import json
import numpy as np
import os

def check_class_imbalance_json(json_path= "datasets/merged_financial_sentiment.json", label_column='label'):
    """
    Check class imbalance in a JSON dataset (local execution).
    Supports both JSON and JSONL (JSON Lines) formats.
    
    Parameters:
    - json_path: path to JSON file (relative or absolute)
    - label_column: name of the key containing labels in JSON
    
    Returns:
    - Dictionary with imbalance statistics and DataFrame
    """
    
    # Check if file exists
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"File not found: {json_path}")
    
    # Load JSON file
    print(f"Loading JSON file from: {json_path}")
    
    # Try to detect JSONL format (JSON Lines - one JSON object per line)
    data = []
    with open(json_path, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
        f.seek(0)  # Reset to beginning
        
        # Check if it's JSONL format (each line is a separate JSON object)
        if first_line.startswith('{') and not first_line.startswith('[{'):
            # JSONL format - read line by line
            print("Detected JSONL format (JSON Lines)")
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping invalid JSON line: {e}")
        else:
            # Standard JSON format
            data = json.load(f)
    
    # Convert to DataFrame
    if isinstance(data, list):
        df = pd.DataFrame(data)
    elif isinstance(data, dict):
        # If it's a dict, try to find the key that contains the list of records
        for key in ['data', 'records', 'items', 'samples']:
            if key in data and isinstance(data[key], list):
                df = pd.DataFrame(data[key])
                break
        else:
            # If no list found, convert the dict itself
            df = pd.DataFrame([data])
    else:
        raise ValueError("JSON format not supported. Expected list or dict.")
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Available columns: {df.columns.tolist()}")
    
    # Check if label column exists
    if label_column not in df.columns:
        print(f"\n❌ Label column '{label_column}' not found!")
        print(f"Available columns: {df.columns.tolist()}")
        raise ValueError(f"Label column '{label_column}' not found in the dataset.")
    
    # Get label counts
    label_counts = df[label_column].value_counts().sort_index()
    total_samples = len(df)
    
    # Calculate statistics
    stats = {
        'total_samples': total_samples,
        'num_classes': len(label_counts),
        'class_counts': label_counts.to_dict(),
        'class_percentages': (label_counts / total_samples * 100).to_dict(),
        'imbalance_ratio': label_counts.max() / label_counts.min(),
    }
    
    # Print statistics
    print("\n" + "=" * 60)
    print("CLASS IMBALANCE ANALYSIS")
    print("=" * 60)
    print(f"\nTotal samples: {total_samples}")
    print(f"Number of classes: {stats['num_classes']}")
    print(f"\nClass distribution:")
    print("-" * 60)
    
    for label, count in label_counts.items():
        percentage = (count / total_samples) * 100
        bar = '█' * int(percentage / 2)  # Visual bar
        print(f"Class {label}: {count:6d} ({percentage:5.2f}%) {bar}")
    
    print("-" * 60)
    print(f"\nImbalance Ratio (max/min): {stats['imbalance_ratio']:.2f}")
    
    # Determine imbalance severity
    if stats['imbalance_ratio'] < 2:
        severity = "✅ Balanced"
    elif stats['imbalance_ratio'] < 5:
        severity = "⚠️  Mildly Imbalanced"
    elif stats['imbalance_ratio'] < 10:
        severity = "⚠️⚠️  Moderately Imbalanced"
    else:
        severity = "🚨 Highly Imbalanced"
    
    print(f"Imbalance Status: {severity}")
    print("=" * 60)
    
    # Create visualizations
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart
    axes[0].bar(label_counts.index.astype(str), label_counts.values, 
                color=plt.cm.Set3(range(len(label_counts))))
    axes[0].set_xlabel('Class Label', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('Class Distribution (Count)', fontsize=14, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Add count labels on bars
    for i, (label, count) in enumerate(label_counts.items()):
        axes[0].text(i, count, str(count), ha='center', va='bottom', fontweight='bold')
    
    # Pie chart
    axes[1].pie(label_counts.values, labels=label_counts.index.astype(str), 
                autopct='%1.1f%%', startangle=90, 
                colors=plt.cm.Set3(range(len(label_counts))))
    axes[1].set_title('Class Distribution (Percentage)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    return stats, df


# Example usage:
if __name__ == "__main__":
    # Use the merged dataset file
    json_file_path = "datasets/merged_financial_sentiment.json"
    
    # Replace 'label' with your actual label column name if different
    stats, df = check_class_imbalance_json(json_file_path, label_column='label')
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Dataset loaded successfully!")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Classes: {list(stats['class_counts'].keys())}")
    print("=" * 60)
