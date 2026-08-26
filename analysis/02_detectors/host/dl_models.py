from huggingface_hub import snapshot_download
for m in ["Qwen/Qwen2.5-7B","tiiuae/falcon-7b","tiiuae/falcon-7b-instruct","codellama/CodeLlama-7b-hf"]:
    print(f"--- {m}", flush=True)
    p = snapshot_download(m, allow_patterns=["*.json","*.safetensors","*.model","*.txt"])
    print(f"    -> {p}", flush=True)
print("ALLE FERTIG", flush=True)
