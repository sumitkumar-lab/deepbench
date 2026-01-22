import torch
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import pandas as pd
import re

class ModelResearcher:
    def __init__(self):
        self.api = HfApi()

    def search_models(self, architecture_type="transformer", sort_by="downloads", limit=20):
        """
        Searches HF Hub for models based on loose architecture tags.
        """
        # Map user friendly terms to HF tags
        search_query = None
        filter_tags = []
        
        if architecture_type == "Recurrent (RNN/RWKV/Mamba)":
            filter_tags = ["rwkv"] # Mamba often needs specific custom code, keeping it simple
        elif architecture_type == "Attention (Transformer)":
            filter_tags = ["transformers"]
        
        # Fetch models
        models = self.api.list_models(
            sort=sort_by,
            direction=-1,
            limit=limit * 2, # Fetch more to filter locally
            filter=filter_tags,
            task="text-generation"
        )

        model_list = []
        for m in models:
            # Heuristic to guess parameter size from name (e.g., 7b, 1.5B)
            size_match = re.search(r'([0-9\.]+)b', m.modelId.lower())
            size_label = f"{size_match.group(1)}B" if size_match else "Unknown"
            
            model_list.append({
                "model_id": m.modelId,
                "likes": m.likes,
                "downloads": m.downloads,
                "created_at": str(m.created_at)[:10],
                "estimated_params": size_label
            })
        
        return pd.DataFrame(model_list).head(limit)

class ModelEvaluator:
    def __init__(self, model_id, device="cpu"):
        self.device = device
        self.model_id = model_id
        self.tokenizer = None
        self.model = None

    def load_model(self):
        """Loads the model and tokenizer."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            # Use float16 if on GPU for memory saving
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, 
                torch_dtype=dtype,
                trust_remote_code=True # Needed for some new architectures
            ).to(self.device)
            self.model.eval()
            return True, "Model Loaded Successfully"
        except Exception as e:
            return False, str(e)

    def calculate_perplexity(self, dataset_name="wikitext", subset="wikitext-2-raw-v1", max_samples=50):
        """
        Calculates perplexity on a small subset of a dataset.
        """
        try:
            data = load_dataset(dataset_name, subset, split=f"test[:{max_samples}]")
            encodings = self.tokenizer("\n\n".join(data["text"]), return_tensors="pt")
            
            max_length = self.model.config.max_position_embeddings
            stride = 512
            seq_len = encodings.input_ids.size(1)

            nlls = []
            prev_end_loc = 0
            
            # Sliding window perplexity calculation
            for begin_loc in range(0, seq_len, stride):
                end_loc = min(begin_loc + max_length, seq_len)
                trg_len = end_loc - prev_end_loc 
                input_ids = encodings.input_ids[:, begin_loc:end_loc].to(self.device)
                
                target_ids = input_ids.clone()
                target_ids[:, :-trg_len] = -100

                with torch.no_grad():
                    outputs = self.model(input_ids, labels=target_ids)
                    neg_log_likelihood = outputs.loss

                nlls.append(neg_log_likelihood)
                prev_end_loc = end_loc
                if end_loc == seq_len:
                    break

            ppl = torch.exp(torch.stack(nlls).mean())
            return float(ppl)
        except Exception as e:
            print(f"Error: {e}")
            return None