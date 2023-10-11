from datasets import load_dataset, load_from_disk
from os.path import exists
from functools import wraps


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

if __name__ == "__main__":
    get_dataset("jpwahle/etpc")


