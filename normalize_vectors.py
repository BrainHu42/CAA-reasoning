from behaviors import get_vector_path, ALL_BEHAVIORS
from utils.helpers import get_model_path
import torch as t
import os
import argparse

def normalize_vectors(model_size: str, is_base: bool, n_layers: int, behaviors: list, pre_mlp: bool):
    # make normalized_vectors directory
    normalized_vectors_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "normalized_vectors")
    if not os.path.exists(normalized_vectors_dir):
        os.makedirs(normalized_vectors_dir)
    for layer in range(n_layers):
        print(layer)
        norms = {}
        vecs = {}
        new_paths = {}
        for behavior in behaviors:
            vec_path = get_vector_path(behavior, layer, get_model_path(model_size, is_base=is_base), pre_mlp=pre_mlp)
            vec = t.load(vec_path)
            norm = vec.norm().item()
            vecs[behavior] = vec
            norms[behavior] = norm
            new_path = vec_path.replace("vectors", "normalized_vectors")
            new_paths[behavior] = new_path
        # print(norms)
        mean_norm = t.tensor(list(norms.values())).mean().item()
        # normalize all vectors to have the same norm
        for behavior in behaviors:
            vecs[behavior] = vecs[behavior] * mean_norm / norms[behavior]
        # save the normalized vectors
        for behavior in behaviors:
            if not os.path.exists(os.path.dirname(new_paths[behavior])):
                os.makedirs(os.path.dirname(new_paths[behavior]))
            t.save(vecs[behavior], new_paths[behavior])
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Normalize vectors for specified behaviors and model size.')
    parser.add_argument("--behaviors", nargs="+", type=str, default=ALL_BEHAVIORS, help='List of behaviors to process')
    parser.add_argument('--model_size', choices=['7b', '8b', '13b'], required=True, help='Model size to use')
    parser.add_argument("--pre_mlp", action="store_true", default=False)
    args = parser.parse_args()
    
    behaviors = args.behaviors
    model_size = args.model_size
    
    if model_size == '7b' or model_size == '8b':
        n_layers = 32
    elif model_size == '13b':
        n_layers = 36

    normalize_vectors(model_size, False, n_layers, behaviors, args.pre_mlp)