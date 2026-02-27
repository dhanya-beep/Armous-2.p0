import os
import json
import hashlib
import numpy as np
from collections import Counter
from simhash import Simhash
from datasketch import MinHash
import spacy

RAW_FOLDER = "raw_content"
PROCESSED_FOLDER = "processed_content"
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

nlp = spacy.load("en_core_web_sm")

def convert_numpy(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return obj

def sha256_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def split_paragraphs(text):
    return [p.strip() for p in text.split("\n\n") if p.strip()]

def extract_ngrams(tokens, n):
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def compute_simhash(tokens):
    return int(Simhash(tokens).value)

def compute_minhash(tokens):
    m = MinHash(num_perm=128)
    for t in set(tokens):
        m.update(t.encode("utf-8"))
    return [int(x) for x in m.hashvalues]

for file in os.listdir(RAW_FOLDER):

    if not file.endswith(".json"):
        continue

    with open(os.path.join(RAW_FOLDER, file), "r") as f:
        data = json.load(f)

    text = data.get("clean_text", "")
    if not text.strip():
        continue

    doc = nlp(text)
    paragraphs = split_paragraphs(text)
    sentences = list(doc.sents)
    tokens = [t.text.lower() for t in doc if t.is_alpha]

    dependency_sig = Counter([t.dep_ for t in doc])
    pos_dist = Counter([t.pos_ for t in doc])

    stylometry = {
        "avg_sentence_length": float(np.mean([len(s.text.split()) for s in sentences])) if sentences else 0,
        "sentence_length_variance": float(np.var([len(s.text.split()) for s in sentences])) if sentences else 0,
        "pos_distribution": {k: float(v/len(doc)) for k,v in pos_dist.items()}
    }

    processed = {
        "id": data.get("id", file),
        "hash": sha256_hash(text),

        "paragraphs": paragraphs,
        "ordered_entities": [ent.text.lower() for ent in doc.ents],

        "structure": {
            "paragraph_count": len(paragraphs),
            "sentence_count": len(sentences)
        },

        "lexical": {
            "ngrams_3": list(set(extract_ngrams(tokens,3))),
            "ngrams_5": list(set(extract_ngrams(tokens,5)))
        },

        "entities": {
            "entity_set": list(set(ent.text.lower() for ent in doc.ents))
        },

        "syntactic": {
            "dependency_signature": dict(dependency_sig)
        },

        "stylometry": stylometry,

        "fingerprints": {
            "simhash": compute_simhash(tokens),
            "minhash_signature": compute_minhash(tokens)
        }
    }

    output_path = os.path.join(PROCESSED_FOLDER, f"{processed['id']}-processed.json")

    with open(output_path, "w") as f:
        json.dump(processed, f, indent=4, default=convert_numpy)

print("Corpus preprocessing complete.")
