import json
import argparse

def load_json_list(path):
    with open(path, 'r') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} should contain a JSON list.")
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val", required=True, help="validation json path")
    parser.add_argument("--test", required=True, help="test/eval json path")
    args = parser.parse_args()

    val_data = load_json_list(args.val)
    test_data = load_json_list(args.test)

    # Extract image_ids
    val_image_ids = set([v["image_id"] for v in val_data])
    test_image_ids = set([t["image_id"] for t in test_data])

    # Compute overlap
    overlap = val_image_ids & test_image_ids
    
    print("Validation image_id count:", len(val_image_ids))
    print("Test/Eval image_id count:", len(test_image_ids))
    print("Overlap count:", len(overlap))
    print("Overlap fraction relative to test:",
          len(overlap) / max(1, len(test_image_ids)))

    if overlap:
        print("\nExample overlapping image_ids:", list(overlap)[:20])

        print("\nValidation examples of overlapping images:")
        shown = 0
        for v in val_data:
            if v["image_id"] in overlap:
                print(v)
                shown += 1
                if shown >= 5:
                    break

        print("\nTest/Eval examples of overlapping images:")
        shown = 0
        for t in test_data:
            if t["image_id"] in overlap:
                print({
                    "image_id": t["image_id"],
                    "image_path": t.get("image_path")
                })
                shown += 1
                if shown >= 5:
                    break

if __name__ == "__main__":
    main()
