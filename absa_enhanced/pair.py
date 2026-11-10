"""Aspect-conditioned sentence pairs with target-preserving context cropping."""
import re

import torch
from torch import nn
from torch.utils.data import Dataset

from .data import SENTIMENTS, align_span, trim_span
from .model import JointABSA
from . import heads

# Clause boundaries used by --context-scope clause. Sentiment scope is usually
# local: "the food was great, but the service was slow" carries two opposing
# opinions, and the sentence-level aggregate is a confound for the neutral one.
CLAUSE_BREAK = re.compile(
    r"[;.!?\n]|,|\b(?:but|however|although|though|whereas|yet|nevertheless|nonetheless|"
    r"while|except)\b", re.IGNORECASE)
MIN_CLAUSE_WORDS = 2


def clause_bounds(text, start, end):
    """Character range of the clause containing [start, end), widened if very short."""
    breaks = [(match.start(), match.end()) for match in CLAUSE_BREAK.finditer(text)]
    segments, cursor = [], 0
    for left, right in breaks:
        if left > cursor:
            segments.append((cursor, left))
        cursor = right
    if cursor < len(text):
        segments.append((cursor, len(text)))
    if not segments:
        return 0, len(text)
    chosen = None
    for index, (left, right) in enumerate(segments):
        if left <= start and end <= right:
            chosen = index
            break
    if chosen is None:
        return 0, len(text)
    left, right = segments[chosen]
    # Too short to carry any evidence: borrow the neighbouring clause.
    while len(re.findall(r"[A-Za-z']+", text[left:right])) < MIN_CLAUSE_WORDS and (
            chosen > 0 or chosen < len(segments) - 1):
        if chosen > 0:
            chosen -= 1
            left = segments[chosen][0]
        elif chosen < len(segments) - 1:
            chosen += 1
            right = segments[chosen][1]
    return left, right


def token_range(offsets, char_start, char_end):
    """First and last+1 token indices that lie inside [char_start, char_end)."""
    indices = [i for i, (left, right) in enumerate(offsets) if right > char_start and left < char_end]
    if not indices:
        return None
    return indices[0], indices[-1] + 1


class PairDataset(Dataset):
    """One training row per gold aspect; no implicit extraction supervision."""
    def __init__(self, documents, tokenizer, max_length=192, stride=None, scope="sentence"):
        if not tokenizer.is_fast:
            raise ValueError("PairDataset requires a fast tokenizer")
        if scope not in ("sentence", "clause"):
            raise ValueError(f"Unknown context scope: {scope}")
        self.scope = scope
        self.documents, self.rows = documents, []
        non_exact, cropped, targets, narrowed = [], 0, 0, 0
        for doc_index, document in enumerate(documents):
            encoded = tokenizer(document.text, add_special_tokens=False, return_offsets_mapping=True)
            full_ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
            for aspect in document.aspects:
                if aspect.label not in SENTIMENTS:
                    continue
                targets += 1
                text = document.text[aspect.start:aspect.end]
                aspect_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
                positions = align_span(offsets, aspect.start, aspect.end)
                if not positions or not aspect_ids:
                    raise ValueError(f"Cannot align aspect in {document.uid}: {text!r}")
                first, end = positions[0], positions[-1] + 1
                budget = max_length - len(aspect_ids) - tokenizer.num_special_tokens_to_add(pair=True)
                if budget < end - first:
                    raise ValueError("max-length too small to retain target and aspect query")
                low, high = 0, len(full_ids)
                if scope == "clause":
                    clause_start, clause_end = clause_bounds(document.text, aspect.start, aspect.end)
                    narrowed_span = token_range(offsets, clause_start, clause_end)
                    if narrowed_span is not None and narrowed_span[0] <= first and end <= narrowed_span[1]:
                        low, high = narrowed_span
                # Keep the target centred inside whichever scope is allowed.
                left = max(low, first - (budget - (end - first)) // 2)
                left = min(left, max(low, high - budget))
                context_ids = full_ids[left:min(high, left + budget)]
                cropped += int(len(context_ids) < len(full_ids))
                narrowed += int((low, high) != (0, len(full_ids)))
                input_ids = tokenizer.build_inputs_with_special_tokens(context_ids, aspect_ids)
                # The encoder's own pair template inserts CLS/SEP tokens. Negative
                # sentinels locate the two sequences without hardcoding a BERT layout.
                markers = tokenizer.build_inputs_with_special_tokens([-1] * len(context_ids), [-2] * len(aspect_ids))
                context_positions = [i for i, marker in enumerate(markers) if marker == -1]
                query_mask = [marker == -2 for marker in markers]
                start_token = context_positions[first - left]
                end_token = context_positions[end - left - 1] + 1
                inputs = {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}
                if "token_type_ids" in tokenizer.model_input_names:
                    inputs["token_type_ids"] = tokenizer.create_token_type_ids_from_sequences(context_ids, aspect_ids)
                covering = trim_span(document.text, offsets[first][0], offsets[end - 1][1])
                if covering != (aspect.start, aspect.end):
                    non_exact.append({"sentence_id": document.uid, "aspect": text,
                                      "start": aspect.start, "end": aspect.end,
                                      "covering_token_text": document.text[covering[0]:covering[1]]})
                label = SENTIMENTS.index(aspect.label)
                self.rows.append({"inputs": inputs, "valid": query_mask, "bio": [-100] * len(input_ids),
                                  "targets": [(start_token, end_token, label, aspect.start, aspect.end)],
                                  "doc_index": doc_index})
        self.alignment_stats = {"gold_aspects": targets, "non_exact_token_spans": len(non_exact),
                                "non_exact_examples": non_exact[:10], "cropped_contexts": cropped,
                                "clause_narrowed_contexts": narrowed, "context_scope": scope}
        if not self.rows:
            raise ValueError("Pair dataset has no three-class aspect targets")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class PairABSA(JointABSA):
    """The target participates in every encoder layer through the second sequence."""
    model_kind = "pair"

    def __init__(self, encoder, dropout=0.15, use_cls=True, sentiment_head="softmax"):
        nn.Module.__init__(self)
        self.encoder = encoder
        self.local_radius, self.fusion, self.dropout_rate = 0, False, dropout
        self.use_cls = use_cls
        self.sentiment_head = sentiment_head
        hidden = encoder.config.hidden_size
        features = hidden * (2 + int(use_cls))
        self.sentiment = nn.Sequential(nn.LayerNorm(features), nn.Dropout(dropout),
                                       nn.Linear(features, 256), nn.GELU(),
                                       nn.Dropout(dropout), nn.Linear(256, heads.output_dim(sentiment_head)))

    def encode(self, inputs):
        hidden = self.encoder(**inputs, return_dict=True).last_hidden_state
        # Reuse the training interface. All pair BIO labels are ignored; no BIO head.
        return hidden, hidden[..., :3] * 0

    def classify(self, hidden, query_mask, spans):
        if not len(spans):
            return hidden.new_empty((0, 3))
        batch, starts, ends = spans.unbind(-1)
        positions = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0)
        target_mask = (positions >= starts[:, None]) & (positions < ends[:, None])
        selected = hidden[batch]
        target = (selected * target_mask.unsqueeze(-1)).sum(1) / target_mask.sum(1, keepdim=True).clamp_min(1)
        query = query_mask[batch]
        query_vector = (selected * query.unsqueeze(-1)).sum(1) / query.sum(1, keepdim=True).clamp_min(1)
        parts = (hidden[batch, 0], target, query_vector) if self.use_cls else (target, query_vector)
        return self.sentiment(torch.cat(parts, -1))
