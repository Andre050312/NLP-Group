"""Target-preserving tokenization and a standard BERT sequence classifier."""
import torch
from torch.utils.data import Dataset


def encode_example(tokenizer, example, max_length):
    # Tokenize with offsets so repeated aspect words select the annotated occurrence.
    encoded = tokenizer(example.text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    positions = [i for i, (start, end) in enumerate(offsets)
                 if end > example.start and start < example.end and end > start]
    if not positions:
        raise ValueError("Tokenizer could not align the aspect span")
    aspect_ids = tokenizer(example.aspect, add_special_tokens=False)["input_ids"]
    budget = max_length - len(aspect_ids) - tokenizer.num_special_tokens_to_add(pair=True)
    first, last = positions[0], positions[-1] + 1
    if budget < last-first:
        raise ValueError("max_length is too small to retain both the aspect and sentence pair")
    # Keep a centered window containing the complete target, rather than right truncation.
    start = max(0, first - (budget-(last-first))//2)
    start = min(start, max(0, len(ids)-budget))
    sentence_ids = ids[start:start+budget]
    # Explicit BERT pair layout is stable across Transformers 4/5 APIs.
    if tokenizer.num_special_tokens_to_add(pair=True) != 3:
        raise ValueError("This baseline requires a BERT-style tokenizer with three pair special tokens")
    input_ids = [tokenizer.cls_token_id] + sentence_ids + [tokenizer.sep_token_id] + aspect_ids + [tokenizer.sep_token_id]
    result = {"input_ids": input_ids, "attention_mask": [1]*len(input_ids)}
    if "token_type_ids" in tokenizer.model_input_names:
        result["token_type_ids"] = [0]*(len(sentence_ids)+2) + [1]*(len(aspect_ids)+1)
    return result


class AspectDataset(Dataset):
    def __init__(self, examples, tokenizer, labels, max_length=128):
        label_to_id = {label: i for i, label in enumerate(labels)}
        self.rows = []
        for example in examples:
            row = encode_example(tokenizer, example, max_length)
            row["labels"] = label_to_id[example.label]
            self.rows.append(row)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def resolve_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
