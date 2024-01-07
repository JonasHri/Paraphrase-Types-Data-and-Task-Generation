"""
Wrappers for hugging face functions that save downloaded assets and dont require internet access for subsequent calls.
"""
from datasets import load_dataset, load_from_disk
from os.path import exists
from functools import wraps
from transformers import AutoTokenizer
import transformers
import torch

@wraps(load_dataset)
def get_dataset(path: str, *args, **kwargs):

    location = f"./data/{path}"

    if exists(location):
        print(f"loading dataset from disk at {location}")
        dataset = load_from_disk(location, *args, **kwargs)
    else:
        print(f"downloading dataset huggingface databank at {path}")
        dataset = load_dataset(path, *args, **kwargs)
    # dataset = load_dataset("jpwahle/etpc")
        print(f"saving dataset at {location}")
        dataset.save_to_disk(location)

    return dataset

@wraps(AutoTokenizer.from_pretrained)
def get_tokenizer(path: str, *args, **kwargs):
    location = f"./tokenizers/{path}"

    if exists(location):
        print(f"loading tokenizer from disk at {location}")
        tokenizer = AutoTokenizer.from_pretrained(location, *args, **kwargs)
    else:
        print(f"downloading tokenizer from huggingface databank at {path}")
        tokenizer = AutoTokenizer.from_pretrained(path, *args, **kwargs)
    # dataset = load_dataset("jpwahle/etpc")
        print(f"saving dataset at {location}")
        tokenizer.save_pretrained(location)

    return tokenizer 

@wraps(transformers.pipeline)
def get_pipline(task: str, *args, model="", **kwargs):
    location = f"./pipelines/{model}"

    if exists(location):
        print(f"loading pipeline from disk at {location}")
        pipeline = transformers.pipeline(task, location, *args, **kwargs) 
    else:
        print(f"downloading pipeline from huggingface databank at {model}")
        pipeline = transformers.pipeline(task, *args, model=model, **kwargs)
        print(f"saving pipline at {location}")
        pipeline.save_pretrained(location)

    return pipeline




if __name__ == "__main__":
    get_dataset("jpwahle/etpc")

    get_tokenizer("meta-llama/Llama-2-7b-chat-hf")

    get_pipline(
    "text-generation",
    model="meta-llama/Llama-2-7b-chat-hf",
    torch_dtype=torch.float16,
    device_map="auto",
)



