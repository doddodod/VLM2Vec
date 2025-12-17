import json

val_path = "/public/home/wangby2025/plusLab/data/vg/test_2k.json"
test_path = "/public/home/wangby2025/plusLab/VLM2Vec/infer/test_2000_images.json"
out_path = "/public/home/wangby2025/plusLab/VLM2Vec/infer/test_2000_clean.json"

# Load data
with open(val_path, "r") as f:
    val_data = json.load(f)
with open(test_path, "r") as f:
    test_data = json.load(f)

val_ids = set([v["image_id"] for v in val_data])

clean_test = [t for t in test_data if t["image_id"] not in val_ids]

print("Original test size:", len(test_data))
print("Validation size:", len(val_data))
print("Overlap count:", len(test_data) - len(clean_test))
print("Clean test size:", len(clean_test))

# Save
with open(out_path, "w") as f:
    json.dump(clean_test, f, indent=2)

print("Saved clean test set to:", out_path)
