import json
import os
from datetime import datetime

CFI_DB_FILE = "cfi_store.json"


# ----------------------------
# Load / Save DB
# ----------------------------

def load_db():
    if not os.path.exists(CFI_DB_FILE):
        return {}
    with open(CFI_DB_FILE, "r") as f:
        return json.load(f)


def save_db(db):
    with open(CFI_DB_FILE, "w") as f:
        json.dump(db, f, indent=2)


# ----------------------------
# Generate new CFI ID
# ----------------------------

def generate_cfi_id(db):
    return f"CFI_{len(db) + 1:04d}"


# ----------------------------
# Create or Update Article
# ----------------------------

def create_or_update_article():
    url = input("Enter article URL: ").strip()
    content = input("\nPaste article content:\n\n").strip()

    db = load_db()

    # Check if URL already exists
    for cfi_id, record in db.items():
        if record["url"] == url:
            # Existing article → new version
            new_version = len(record["versions"]) + 1
            record["versions"].append({
                "version": new_version,
                "content": content,
                "timestamp": datetime.utcnow().isoformat()
            })
            save_db(db)
            print(f"\nArticle updated.")
            print(f"CFI ID: {cfi_id}")
            print(f"New version: {new_version}")
            return

    # New article → create CFI
    cfi_id = generate_cfi_id(db)
    db[cfi_id] = {
        "url": url,
        "versions": [
            {
                "version": 1,
                "content": content,
                "timestamp": datetime.utcnow().isoformat()
            }
        ]
    }

    save_db(db)
    print(f"\nNew article registered.")
    print(f"CFI ID: {cfi_id}")
    print(f"Version: 1")


# ----------------------------
# View CFI Records (for results)
# ----------------------------

def view_cfi_records():
    db = load_db()
    if not db:
        print("\nNo CFI records found.")
        return

    for cfi_id, record in db.items():
        print(f"\n{cfi_id}")
        print(f"URL: {record['url']}")
        for v in record["versions"]:
            print(f"  Version {v['version']} @ {v['timestamp']}")


# ----------------------------
# CLI
# ----------------------------

def main():
    print("\nPhase-1: Content Flow Identification (CFI)")
    print("1 → Create / Update Article")
    print("2 → View CFI Records")

    choice = input("\nChoose option: ").strip()

    if choice == "1":
        create_or_update_article()
    elif choice == "2":
        view_cfi_records()
    else:
        print("Invalid option.")


if __name__ == "__main__":
    main()

