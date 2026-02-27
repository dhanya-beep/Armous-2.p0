import os
import json

# Folder containing raw JSON files
RAW_FOLDER = "raw_content"

# Get all JSON files
files = [f for f in os.listdir(RAW_FOLDER) if f.endswith(".json")]

# Sort to ensure consistent numbering
files.sort()

for index, filename in enumerate(files, start=1):
    old_path = os.path.join(RAW_FOLDER, filename)

    # New ID format
    new_id = f"BBC-{index:02d}"
    new_filename = f"{new_id}.json"
    new_path = os.path.join(RAW_FOLDER, new_filename)

    # Load JSON
    with open(old_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Update ID inside JSON
    data["id"] = new_id

    # Save with new filename
    with open(new_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    # Remove old file (only if filename changed)
    if old_path != new_path:
        os.remove(old_path)

    print(f"Renamed {filename} → {new_filename}")

print("Renaming complete.")
