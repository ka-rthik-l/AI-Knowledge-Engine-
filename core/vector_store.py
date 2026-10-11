import numpy as np
import json
from config import STORE_PREFIX

def save_store(texts, vectors, prefix=STORE_PREFIX):
    np.save(prefix + "_vectors.npy", np.array(vectors))
    
    with open(prefix + "_texts.json", "w") as f:
        json.dump(texts, f)


def load_store(prefix=STORE_PREFIX):
    vectors = np.load(prefix + "_vectors.npy")
    
    with open(prefix + "_texts.json", "r") as f:
        texts = json.load(f)

    return texts, vectors
