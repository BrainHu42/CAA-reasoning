"""
Usage: python analyze_vectors.py
"""

import os
from matplotlib.pylab import f
import torch as t
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from behaviors import ALL_BEHAVIORS, get_analysis_dir, HUMAN_NAMES, get_steering_vector, ANALYSIS_PATH, get_activations_path
from utils.helpers import get_model_path, model_name_format, set_plotting_settings
from tqdm import tqdm

set_plotting_settings()

def get_caa_info(behavior: str, model_size: str, is_base: bool):
    all_vectors = []
    n_layers = 36 if "13" in model_size else 32
    model_path = get_model_path(model_size, is_base)
    for layer in range(n_layers):
        all_vectors.append(get_steering_vector(behavior, layer, model_path))
    return {
        "vectors": all_vectors,
        "n_layers": n_layers,
        "model_name": model_name_format(model_path),
    }
    
def plot_relative_magnitudes(model_sizes, is_base: bool, behavior: str):
    plt.clf()
    plt.figure(figsize=(4, 4))
    
    for size in model_sizes:
        model_name_path = get_model_path(size, is_base)
        
        caa_info = get_caa_info(behavior, size, is_base)
        vectors = caa_info["vectors"]
        model_name = caa_info["model_name"]
        relative_mags = []
        for layer in range(caa_info["n_layers"]):
            magnitude = t.norm(vectors[layer]).item()
            
            # Loading activations
            activations_pos = t.load(get_activations_path(behavior, layer, model_name_path, "pos"))
            activations_neg = t.load(get_activations_path(behavior, layer, model_name_path, "neg"))
            activations = t.cat([activations_pos, activations_neg], dim=0)
            
            # Calculating average activation
            avg_activation = t.mean(t.norm(activations, dim=1)).item()
            
            # Calculating relative magnitude
            relative_mag = magnitude / avg_activation
            relative_mags.append(relative_mag)
        
        plt.plot(list(range(caa_info["n_layers"])), relative_mags, linestyle="solid", linewidth=2, label=model_name)
    
    plt.xlabel("Layer")
    plt.ylabel("Relative Magnitude")
    plt.title(f"{HUMAN_NAMES[behavior]} relative vector magnitudes per layer")
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(ANALYSIS_PATH, f"{behavior}_relative_vector_magnitudes.png")
    print(f"SAVING TO {save_path}")
    plt.savefig(save_path, format='png')
    plt.close()


def plot_vector_magnitudes(model_sizes: list, is_base: bool, behavior):
    plt.figure(figsize=(5, 3))

    for size in model_sizes:
        caa_info = get_caa_info(behavior, size, is_base)
        vectors = caa_info["vectors"]
        model_name = caa_info["model_name"]
        magnitudes = []
        for layer in range(caa_info["n_layers"]):
            magnitude = t.norm(vectors[layer]).item()
            magnitudes.append(magnitude)
        plt.plot(list(range(caa_info["n_layers"])), magnitudes, linestyle="solid", linewidth=2, label=model_name)
    
    plt.xlabel("Layer")
    plt.ylabel("Magnitude")
    plt.title(f"{HUMAN_NAMES[behavior]} raw vector magnitudes per layer")
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(ANALYSIS_PATH, f"{behavior}_vector_magnitudes.png")
    print(f"SAVING TO {save_path}")
    plt.savefig(save_path, format='png')
    plt.close()
    
def plot_behavior_similarities(model_sizes: list, is_base: bool, behavior1, behavior2):
    plt.figure(figsize=(5, 3))

    for size in model_sizes:
        caa_info1 = get_caa_info(behavior1, size, is_base)
        caa_info2 = get_caa_info(behavior2, size, is_base)
        vectors1 = caa_info1["vectors"]
        vectors2 = caa_info2["vectors"]
        model_name = caa_info1["model_name"]
        cos_sims = []
        for layer in range(caa_info1["n_layers"]):
            cos_sim = t.nn.functional.cosine_similarity(vectors1[layer], vectors2[layer], dim=0).item()
            cos_sims.append(cos_sim)
        plt.plot(list(range(caa_info1["n_layers"])), cos_sims, linestyle="solid", linewidth=2, label=model_name)
    plt.xlabel("Layer")
    plt.ylabel("Cosine Similarity")
    plt.title(f"{HUMAN_NAMES[behavior1]} vs. {HUMAN_NAMES[behavior2]} vector similarity")
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(ANALYSIS_PATH, f"{behavior1}_{behavior2}_similarities.png")
    print(f"SAVING TO {save_path}")
    plt.savefig(save_path, format='png')
    plt.close()

def plot_per_layer_similarities(model_size: str, is_base: bool, behavior: str):
    analysis_dir = get_analysis_dir(behavior)
    caa_info = get_caa_info(behavior, model_size, is_base)
    all_vectors = caa_info["vectors"]
    n_layers = caa_info["n_layers"]
    model_name = caa_info["model_name"]
    matrix = np.zeros((n_layers, n_layers))
    for layer1 in range(n_layers):
        for layer2 in range(n_layers):
            cosine_sim = t.nn.functional.cosine_similarity(all_vectors[layer1], all_vectors[layer2], dim=0).item()
            matrix[layer1, layer2] = cosine_sim
    plt.figure(figsize=(3, 3))
    sns.heatmap(matrix, annot=False, cmap='coolwarm')
    # Set ticks for every 5th layer
    plt.xticks(list(range(n_layers))[::5], list(range(n_layers))[::5])
    plt.yticks(list(range(n_layers))[::5], list(range(n_layers))[::5])
    plt.title(f"Layer similarity, {model_name}", fontsize=11)
    plt.savefig(os.path.join(analysis_dir, f"cosine_similarities_{model_name.replace(' ', '_')}_{behavior}.svg"), format='svg')
    plt.close()

def plot_base_chat_similarities():
    plt.figure(figsize=(5, 3))
    for behavior in ALL_BEHAVIORS:
        base_caa_info = get_caa_info(behavior, "7b", True)
        chat_caa_info = get_caa_info(behavior, "7b", False)
        vectors_base = base_caa_info["vectors"]
        vectors_chat = chat_caa_info["vectors"]
        cos_sims = []
        for layer in range(base_caa_info["n_layers"]):
            cos_sim = t.nn.functional.cosine_similarity(vectors_base[layer], vectors_chat[layer], dim=0).item()
            cos_sims.append(cos_sim)
        plt.plot(list(range(base_caa_info["n_layers"])), cos_sims, label=HUMAN_NAMES[behavior], linestyle="solid", linewidth=2)
    plt.xlabel("Layer")
    plt.ylabel("Cosine Similarity")
    plt.title("Base vs. Chat model vector similarity", fontsize=12)
    # legend in bottom right
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_PATH, "base_chat_similarities.png"), format='png')
    plt.close()

if __name__ == "__main__":
    # plot_vector_magnitudes(['7b','8b'], False, 'arc-hard')
    plot_relative_magnitudes(['7b','8b'], False, 'refusal')
    # plot_behavior_similarities(['7b','8b'], False, 'arc-easy', 'commonsense')

    # for behavior in tqdm(ALL_BEHAVIORS):
    #     plot_per_layer_similarities("7b", True, behavior)
    #     plot_per_layer_similarities("7b", False, behavior)
    #     # plot_per_layer_similarities("13b", False, behavior)
    # plot_base_chat_similarities()
