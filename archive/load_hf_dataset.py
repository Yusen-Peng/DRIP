from datasets import load_dataset
import sys


def load_hf_dataset(dataset_name, split):
    """
    Load a dataset from Hugging Face Datasets.

    Args:
        dataset_name (str): The name of the dataset to load.
        split (str): The split of the dataset to load (e.g., 'train', 'test', 'validation').

    Returns:
        Dataset: The loaded dataset.
    """
    try:
        dataset = load_dataset(dataset_name, split=split, streaming=True) # Load-on-the-fly
        return dataset
    except Exception as e:
        print(f"Error loading dataset {dataset_name}: {e}")
        return None


if __name__ == "__main__":
    dataset_name = "laion/laion400m"
    split = "train"
    dataset = load_hf_dataset(dataset_name, split)

    if dataset:
        print(f"Loaded streaming dataset from {dataset_name} ({split} split).")
        try:
            iterator = iter(dataset)
            sample = next(iterator)
            print("First sample:")
            print(sample)
        except Exception as e:
            print(f"Error during iteration: {e}")
    else:
        print("Failed to load dataset.")

    import gc, time
    gc.collect()
    time.sleep(1)