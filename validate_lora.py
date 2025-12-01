import safetensors
model = safetensors.safe_open("/public/home/wangby2025/plusLab/outputs/sgg_qwen2vl_4_card_improved/best_model_eval/adapter_model.safetensors", framework="pt")
print(model.keys())  # Check what's actually in the file