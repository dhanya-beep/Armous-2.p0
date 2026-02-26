import hashlib
import json
import os
import re
from collections import Counter
from datasketch import MinHash

DB_FILE = "fingerprints.json"

# ----------------------------
# Text normalization
# ----------------------------

def normalize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ----------------------------
# Exact fingerprints
# ----------------------------

def sha256_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()

def md5_hash(text):
    return hashlib.md5(text.encode()).hexdigest()

# ----------------------------
# SimHash
# ----------------------------

def simhash(text, bits=64):
    tokens = normalize(text).split()
    counts = Counter(tokens)
    vector = [0] * bits

    for token, weight in counts.items():
        h = int(hashlib.md5(token.encode()).hexdigest(), 16)
        for i in range(bits):
            vector[i] += weight if h & (1 << i) else -weight

    fingerprint = 0
    for i in range(bits):
        if vector[i] > 0:
            fingerprint |= (1 << i)

    return fingerprint

def hamming(a, b):
    return bin(a ^ b).count("1")

# ----------------------------
# MinHash
# ----------------------------

def minhash(text, num_perm=128):
    mh = MinHash(num_perm=num_perm)
    for token in normalize(text).split():
        mh.update(token.encode())
    return mh

# ----------------------------
# n-gram hashing
# ----------------------------

def ngram_hashes(text, n=5):
    text = normalize(text)
    return {
        hashlib.sha1(text[i:i+n].encode()).hexdigest()
        for i in range(len(text) - n + 1)
    }

# ----------------------------
# DB utilities
# ----------------------------

def load_db():
    if not os.path.exists(DB_FILE):
        return []
    with open(DB_FILE, "r") as f:
        return json.load(f)

def save_db(db):
    with open(DB_FILE, "w") as f:
        json.dump(db, f, indent=2)

# ----------------------------
# Generate fingerprints
# ----------------------------

def generate():
    text = input("\nPaste text to fingerprint:\n\n")
    norm = normalize(text)

    record = {
        "sha256": sha256_hash(norm),
        "md5": md5_hash(norm),
        "simhash": simhash(norm),
        "minhash": minhash(norm).hashvalues.tolist(),
        "ngrams": list(ngram_hashes(norm))
    }

    db = load_db()
    db.append(record)
    save_db(db)

    print("\nFingerprints generated and saved.")
    print("Total stored entries:", len(db))

# ----------------------------
# Verify fingerprints
# ----------------------------

def verify():
    text = input("\nPaste suspected text:\n\n")
    norm = normalize(text)

    sha = sha256_hash(norm)
    md5 = md5_hash(norm)
    sim = simhash(norm)
    mh = minhash(norm)
    ng = ngram_hashes(norm)

    db = load_db()
    if not db:
        print("No fingerprints stored.")
        return

    print("\nVerification results:\n")

    for idx, rec in enumerate(db, 1):
        print(f"--- Stored Entry {idx} ---")

        # Exact
        if sha == rec["sha256"]:
            print("Exact SHA-256 match")
        if md5 == rec["md5"]:
            print("Exact MD5 match")

        # SimHash
        dist = hamming(sim, rec["simhash"])
        print("SimHash Hamming Distance:", dist)

        # MinHash
        mh2 = MinHash(hashvalues=rec["minhash"])
        print("MinHash Jaccard Similarity:", mh.jaccard(mh2))

        # n-gram
        inter = len(ng & set(rec["ngrams"]))
        union = len(ng | set(rec["ngrams"]))
        print("n-gram similarity:", inter / union if union else 0.0)

        print()

# ----------------------------
# Main menu
# ----------------------------

def main():
    print("\nBlock-Level Fingerprinting CLI")
    print("1 → Generate fingerprints")
    print("2 → Verify suspected text")

    choice = input("\nChoose option: ").strip()

    if choice == "1":
        generate()
    elif choice == "2":
        verify()
    else:
        print("Invalid option.")

if __name__ == "__main__":
    main()

