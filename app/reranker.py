import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional

def format_instruction(instruction: Optional[str], query: str, doc: str) -> str:
    if instruction is None:
        instruction = 'Given a web search query, retrieve relevant passages that answer the query'

    output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
        instruction=instruction,
        query=query, 
        doc=doc
    )

    return output

model_id = "Qwen/Qwen3-Reranker-0.6B"

_tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')
_model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(model_id).eval()

max_length = 8192

prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

prefix_tokens = _tokenizer.encode(prefix, add_special_tokens=False)
suffix_tokens = _tokenizer.encode(suffix, add_special_tokens=False)

token_false_id = _tokenizer.convert_tokens_to_ids("no")
token_true_id = _tokenizer.convert_tokens_to_ids("yes")

def process_inputs(pairs: list[str]) -> dict:
    global prefix_tokens, suffix_tokens, token_false_id, token_true_id, max_length, _tokenizer, _model
    
    inputs = _tokenizer(
        pairs, padding=False, truncation='longest_first',
        return_attention_mask=False, max_length=max_length - len(prefix_tokens) - len(suffix_tokens)
    )

    for i, ele in enumerate(inputs['input_ids']):
        inputs['input_ids'][i] = prefix_tokens + ele + suffix_tokens

    inputs = _tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)

    for key in inputs:
        inputs[key] = inputs[key].to(_model.device)
    
    return inputs

@torch.no_grad()
def compute_logits(inputs: dict) -> list[float]:
    global token_false_id, token_true_id, _model

    batch_scores = _model(**inputs).logits[:, -1, :]
    true_vector = batch_scores[:, token_true_id]
    false_vector = batch_scores[:, token_false_id]
    batch_scores = torch.stack([false_vector, true_vector], dim=1)
    batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
    scores = batch_scores[:, 1].exp().tolist()

    return scores

def calc_score(query: str, results: list[str], custom_instruction: Optional[str] = None) -> list[tuple[str, float]]:
    pairs = [format_instruction(custom_instruction, query, result) for result in results]
    inputs = process_inputs(pairs)
    scores = compute_logits(inputs)
    return list(zip(results, scores))

