#!/usr/bin/env python3
"""
Plot training metrics from trainer_state.json files with LoRA evolution.
"""

import argparse
import json
import matplotlib.pyplot as plt
import os
import numpy as np
from pathlib import Path
from safetensors.torch import load_file
from sklearn.metrics.pairwise import cosine_similarity


def load_lora_vectors(checkpoint_dir: str):
    """Load LoRA A and B vectors from a checkpoint directory."""
    try:
        # Load the safetensors file
        adapter_path = os.path.join(checkpoint_dir, "adapter_model.safetensors")
        if not os.path.exists(adapter_path):
            return None, None
        
        # Load the adapter weights
        adapter_weights = load_file(adapter_path)
        
        # Find LoRA A and B vectors
        a_vectors = []
        b_vectors = []
        
        for key, tensor in adapter_weights.items():
            if "lora_A" in key:
                # Get the A vector (should be 1D for rank-1)
                vector = tensor.numpy().flatten()
                a_vectors.append(vector)
            elif "lora_B" in key:
                # Get the B vector (should be 1D for rank-1)
                vector = tensor.numpy().flatten()
                b_vectors.append(vector)
        
        # Return average vectors if multiple layers
        if a_vectors and b_vectors:
            avg_a_vector = np.mean(a_vectors, axis=0)
            avg_b_vector = np.mean(b_vectors, axis=0)
            return avg_a_vector, avg_b_vector
        
        return None, None
        
    except Exception as e:
        print(f"Warning: Could not load LoRA vectors from {checkpoint_dir}: {e}")
        return None, None


def calculate_cosine_distance(vec1, vec2):
    """Calculate cosine distance between two vectors."""
    if vec1 is None or vec2 is None:
        return None
    
    # Normalize vectors
    vec1_norm = vec1 / np.linalg.norm(vec1)
    vec2_norm = vec2 / np.linalg.norm(vec2)
    
    # Calculate cosine similarity
    cos_sim = np.dot(vec1_norm, vec2_norm)
    
    # Convert to cosine distance (1 - cosine_similarity)
    cos_distance = 1 - cos_sim
    
    return cos_distance


def plot_training_metrics(trainer_state_path: str, output_dir: str = None):
    """Plot training metrics from trainer_state.json with LoRA evolution.
    
    Parameters
    ----------
    trainer_state_path : str
        Path to the trainer_state.json file
    output_dir : str
        Directory to save the plots (defaults to results directory)
    """
    # Determine output directory
    if output_dir is None:
        # Put plots in the same results directory as the trainer_state.json
        results_dir = os.path.dirname(trainer_state_path)
        output_dir = os.path.join(results_dir, "training_plots")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Plots will be saved to: {output_dir}")
    
    # Load trainer state
    with open(trainer_state_path, 'r') as f:
        trainer_state = json.load(f)
    
    log_history = trainer_state['log_history']
    
    # Extract data
    steps = [entry['step'] for entry in log_history]
    grad_norms = [entry.get('grad_norm', 0) for entry in log_history]
    losses = [entry.get('loss', 0) for entry in log_history]
    learning_rates = [entry.get('learning_rate', 0) for entry in log_history]
    
    # Get checkpoint directory
    checkpoint_dir = os.path.dirname(trainer_state_path)
    base_checkpoint_dir = os.path.dirname(checkpoint_dir)
    
    # Load LoRA vectors from all checkpoints
    checkpoint_dirs = []
    for item in os.listdir(base_checkpoint_dir):
        if item.startswith('checkpoint-') and os.path.isdir(os.path.join(base_checkpoint_dir, item)):
            checkpoint_dirs.append(item)
    
    # Sort checkpoint directories by step number
    checkpoint_dirs.sort(key=lambda x: int(x.split('-')[1]))
    
    print(f"Found {len(checkpoint_dirs)} checkpoints: {checkpoint_dirs}")
    
    # Load LoRA vectors from each checkpoint
    lora_a_norms = []
    lora_b_norms = []
    lora_a_cos_distances = []
    lora_b_cos_distances = []
    checkpoint_steps = []
    
    prev_a_vector = None
    prev_b_vector = None
    
    for checkpoint_dir_name in checkpoint_dirs:
        full_checkpoint_path = os.path.join(base_checkpoint_dir, checkpoint_dir_name)
        
        # Extract step number from checkpoint name
        step_num = int(checkpoint_dir_name.split('-')[1])
        checkpoint_steps.append(step_num)
        
        # Load LoRA vectors
        a_vector, b_vector = load_lora_vectors(full_checkpoint_path)
        
        if a_vector is not None and b_vector is not None:
            # Calculate norms
            a_norm = float(np.linalg.norm(a_vector))
            b_norm = float(np.linalg.norm(b_vector))
            lora_a_norms.append(a_norm)
            lora_b_norms.append(b_norm)
            
            # Calculate cosine distances
            if prev_a_vector is not None:
                a_cos_dist = calculate_cosine_distance(prev_a_vector, a_vector)
                b_cos_dist = calculate_cosine_distance(prev_b_vector, b_vector)
                lora_a_cos_distances.append(a_cos_dist)
                lora_b_cos_distances.append(b_cos_dist)
            else:
                lora_a_cos_distances.append(0.0)  # First checkpoint
                lora_b_cos_distances.append(0.0)
            
            prev_a_vector = a_vector
            prev_b_vector = b_vector
        else:
            lora_a_norms.append(0.0)
            lora_b_norms.append(0.0)
            lora_a_cos_distances.append(0.0)
            lora_b_cos_distances.append(0.0)
    
    # Create figure with subplots (6 instead of 4)
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(20, 12))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    
    # Plot gradient norm
    ax1.plot(steps, grad_norms, 'b-', linewidth=1.5, alpha=0.8)
    ax1.set_xlabel("Training Step", fontsize=14)
    ax1.set_ylabel("Gradient Norm", fontsize=14)
    ax1.set_title("Gradient Norm Over Training", fontsize=16, pad=20)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis="both", which="major", labelsize=12)
    
    # Plot loss
    ax2.plot(steps, losses, 'r-', linewidth=1.5, alpha=0.8)
    ax2.set_xlabel("Training Step", fontsize=14)
    ax2.set_ylabel("Loss", fontsize=14)
    ax2.set_title("Training Loss Over Time", fontsize=16, pad=20)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis="both", which="major", labelsize=12)
    
    # Plot learning rate
    ax3.plot(steps, learning_rates, 'g-', linewidth=1.5, alpha=0.8)
    ax3.set_xlabel("Training Step", fontsize=14)
    ax3.set_ylabel("Learning Rate", fontsize=14)
    ax3.set_title("Learning Rate Schedule", fontsize=16, pad=20)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis="both", which="major", labelsize=12)
    
    # Plot LoRA A norms
    if lora_a_norms:
        ax4.plot(checkpoint_steps, lora_a_norms, 'purple', linewidth=2, marker='o', alpha=0.8)
        ax4.set_xlabel("Checkpoint Step", fontsize=14)
        ax4.set_ylabel("LoRA A Norm", fontsize=14)
        ax4.set_title("LoRA A Vector Norm Evolution", fontsize=16, pad=20)
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis="both", which="major", labelsize=12)
    
    # Plot LoRA B norms
    if lora_b_norms:
        ax5.plot(checkpoint_steps, lora_b_norms, 'orange', linewidth=2, marker='o', alpha=0.8)
        ax5.set_xlabel("Checkpoint Step", fontsize=14)
        ax5.set_ylabel("LoRA B Norm", fontsize=14)
        ax5.set_title("LoRA B Vector Norm Evolution", fontsize=16, pad=20)
        ax5.grid(True, alpha=0.3)
        ax5.tick_params(axis="both", which="major", labelsize=12)
    
    # Plot LoRA cosine distances
    if len(lora_a_cos_distances) > 1:
        ax6.plot(checkpoint_steps[1:], lora_a_cos_distances[1:], 'purple', linewidth=2, marker='o', alpha=0.8, label='LoRA A')
        ax6.plot(checkpoint_steps[1:], lora_b_cos_distances[1:], 'orange', linewidth=2, marker='o', alpha=0.8, label='LoRA B')
        ax6.set_xlabel("Checkpoint Step", fontsize=14)
        ax6.set_ylabel("Cosine Distance", fontsize=14)
        ax6.set_title("LoRA Vector Cosine Distance Evolution", fontsize=16, pad=20)
        ax6.grid(True, alpha=0.3)
        ax6.tick_params(axis="both", which="major", labelsize=12)
        ax6.legend()
    
    # Save the combined plot
    checkpoint_name = Path(trainer_state_path).parent.name
    output_path = os.path.join(output_dir, f"{checkpoint_name}_training_metrics.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Combined training metrics plot saved to: {output_path}")
    
    # Create individual plots
    for metric, data, color, title in [
        ("grad_norm", grad_norms, 'blue', "Gradient Norm"),
        ("loss", losses, 'red', "Training Loss"),
        ("learning_rate", learning_rates, 'green', "Learning Rate")
    ]:
        plt.figure(figsize=(10, 6))
        plt.plot(steps, data, color=color, linewidth=1.5, alpha=0.8)
        plt.xlabel("Training Step", fontsize=14)
        plt.ylabel(title, fontsize=14)
        plt.title(f"{title} Over Training", fontsize=16, pad=20)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        individual_path = os.path.join(output_dir, f"{checkpoint_name}_{metric}.png")
        plt.savefig(individual_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"{title} plot saved to: {individual_path}")
    
    # Create LoRA evolution plots
    if lora_a_norms:
        # LoRA A norm evolution
        plt.figure(figsize=(10, 6))
        plt.plot(checkpoint_steps, lora_a_norms, 'purple', linewidth=2, marker='o', alpha=0.8)
        plt.xlabel("Checkpoint Step", fontsize=14)
        plt.ylabel("LoRA A Norm", fontsize=14)
        plt.title("LoRA A Vector Norm Evolution", fontsize=16, pad=20)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        lora_a_path = os.path.join(output_dir, f"{checkpoint_name}_lora_a_norm_evolution.png")
        plt.savefig(lora_a_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"LoRA A norm evolution plot saved to: {lora_a_path}")
    
    if lora_b_norms:
        # LoRA B norm evolution
        plt.figure(figsize=(10, 6))
        plt.plot(checkpoint_steps, lora_b_norms, 'orange', linewidth=2, marker='o', alpha=0.8)
        plt.xlabel("Checkpoint Step", fontsize=14)
        plt.ylabel("LoRA B Norm", fontsize=14)
        plt.title("LoRA B Vector Norm Evolution", fontsize=16, pad=20)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        lora_b_path = os.path.join(output_dir, f"{checkpoint_name}_lora_b_norm_evolution.png")
        plt.savefig(lora_b_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"LoRA B norm evolution plot saved to: {lora_b_path}")
    
    if len(lora_a_cos_distances) > 1:
        # LoRA cosine distance evolution
        plt.figure(figsize=(10, 6))
        plt.plot(checkpoint_steps[1:], lora_a_cos_distances[1:], 'purple', linewidth=2, marker='o', alpha=0.8, label='LoRA A')
        plt.plot(checkpoint_steps[1:], lora_b_cos_distances[1:], 'orange', linewidth=2, marker='o', alpha=0.8, label='LoRA B')
        plt.xlabel("Checkpoint Step", fontsize=14)
        plt.ylabel("Cosine Distance", fontsize=14)
        plt.title("LoRA Vector Cosine Distance Evolution", fontsize=16, pad=20)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        cos_dist_path = os.path.join(output_dir, f"{checkpoint_name}_lora_cosine_distance_evolution.png")
        plt.savefig(cos_dist_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"LoRA cosine distance evolution plot saved to: {cos_dist_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot training metrics from trainer_state.json with LoRA evolution")
    parser.add_argument("trainer_state_path", help="Path to trainer_state.json file")
    parser.add_argument("--output-dir", help="Directory to save plots (defaults to results directory)")
    args = parser.parse_args()
    
    plot_training_metrics(args.trainer_state_path, args.output_dir)


if __name__ == "__main__":
    main()
