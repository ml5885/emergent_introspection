from typing import List, Dict, Tuple
import math
import torch
from transformer_lens import HookedTransformer

def tokenize(model, text):
    toks = model.to_tokens(text, prepend_bos=False)
    if toks.ndim == 1:
        toks = toks.unsqueeze(0)
    return toks.to(model.cfg.device)

def decode(model, toks):
    if toks.ndim == 2:
        toks = toks[0]
    return model.to_string(toks)

def char_to_token_index(model, text, char_index):
    toks = model.to_tokens(text, prepend_bos=False)[0]
    for i in range(1, toks.shape[0] + 1):
        s = model.to_string(toks[:i])
        if len(s) >= char_index:
            return i - 1
    return toks.shape[0] - 1

def find_injection_start_token(model, full_prompt):
    marker = "\n\nTrial 1"
    idx = full_prompt.find(marker)
    if idx == -1:
        idx = full_prompt.rfind("\n\n")
        if idx == -1:
            idx = 0
    return char_to_token_index(model, full_prompt, idx)

def find_last_assistant_colon_token(model, text):
    idx = text.rfind("Assistant:")
    if idx == -1:
        toks = model.to_tokens(text, prepend_bos=False)[0]
        return toks.shape[0] - 1
    colon_idx = idx + len("Assistant:") - 1
    return char_to_token_index(model, text, colon_idx)

def make_resid_add_hook(vector, start_pos, scale):
    v = vector.detach()

    def hook(resid, hook):
        seq = resid.shape[1]
        mask = torch.arange(seq, device=resid.device) >= start_pos
        resid += scale * v[None, None, :] * mask[None, :, None]
        return resid

    return hook

class ConceptSpec:
    def __init__(self, name, prompt_pos, prompt_neg):
        self.name = name
        self.prompt_pos = prompt_pos
        self.prompt_neg = prompt_neg

def activation_at_colon(model, text, layer_idx):
    toks = tokenize(model, text)
    hook_name = f"blocks.{layer_idx}.hook_resid_pre"
    _, cache = model.run_with_cache(toks, names_filter=[hook_name])
    pos = find_last_assistant_colon_token(model, text)
    act = cache[hook_name][0, pos, :].detach()
    return act

def get_contrast_vector(model, layer_idx, spec):
    act_pos = activation_at_colon(model, spec.prompt_pos, layer_idx)
    act_neg = activation_at_colon(model, spec.prompt_neg, layer_idx)
    vec = act_pos - act_neg
    return vec / (vec.norm() + 1e-8)

class LogitDiff:
    def __init__(self, first_token_diff, full_string_diff):
        self.first_token_diff = first_token_diff
        self.full_string_diff = full_string_diff

def yes_no_ids(model):
    yes = model.tokenizer.encode(" Yes", add_special_tokens=False) or model.tokenizer.encode("Yes", add_special_tokens=False)
    no  = model.tokenizer.encode(" No",  add_special_tokens=False) or model.tokenizer.encode("No",  add_special_tokens=False)
    return yes, no

def first_token_logit_diff(model, base_tokens, fwd_hooks):
    yes_ids, no_ids = yes_no_ids(model)
    with model.hooks(fwd_hooks=fwd_hooks):
        logits = model(base_tokens)
    last = logits[0, -1, :]
    return (last[yes_ids[0]] - last[no_ids[0]]).item()

def string_logprob(model, prefix_tokens, text, fwd_hooks):
    ids = model.tokenizer.encode(text, add_special_tokens=False)
    total = 0.0
    cur = prefix_tokens.clone()

    for t in ids:
        with model.hooks(fwd_hooks=fwd_hooks):
            logits = model(cur)
        logp = torch.log_softmax(logits[0, -1, :], dim=-1)[t].item()
        cur = torch.cat([cur, torch.tensor([[t]], device=cur.device)], dim=1)
        total += logp

    return total

def yes_no_logitdiff(model, base_tokens, fwd_hooks):
    ft = first_token_logit_diff(model, base_tokens, fwd_hooks)
    full = string_logprob(model, base_tokens, " Yes", fwd_hooks) - string_logprob(model, base_tokens, " No", fwd_hooks)
    return LogitDiff(first_token_diff=ft, full_string_diff=full)

def generate_with_injection(model, prompt, layer_idx, vector, scale, max_new_tokens=128, temperature=1.0):
    toks = tokenize(model, prompt)
    start_pos = find_injection_start_token(model, prompt)
    hook_name = f"blocks.{layer_idx}.hook_resid_pre"
    hook = make_resid_add_hook(vector.to(model.cfg.device), start_pos, scale)

    with model.hooks(fwd_hooks=[(hook_name, hook)]):
        out = model.generate(toks, max_new_tokens=max_new_tokens, temperature=temperature)

    text = decode(model, out)
    metric = yes_no_logitdiff(model, toks, fwd_hooks=[(hook_name, hook)])

    return {
        "generated_text": text,
        "injection_start_token": int(start_pos),
        "first_token_logit_diff": metric.first_token_diff,
        "full_string_logit_diff": metric.full_string_diff,
    }

def control_question_prompt(q):
    return f"Human: {q}\n\nAssistant:"

def run_controls(model, layer_idx, vector, scale, questions, max_new_tokens=48, temperature=1.0):
    hook_name = f"blocks.{layer_idx}.hook_resid_pre"
    results = []

    for q in questions:
        prompt = control_question_prompt(q)
        toks = tokenize(model, prompt)
        start_pos = find_injection_start_token(model, prompt)
        hook = make_resid_add_hook(vector.to(model.cfg.device), start_pos, scale)

        metric_no = yes_no_logitdiff(model, toks, fwd_hooks=[])
        metric_inj = yes_no_logitdiff(model, toks, fwd_hooks=[(hook_name, hook)])

        with model.hooks(fwd_hooks=[(hook_name, hook)]):
            out = model.generate(toks, max_new_tokens=max_new_tokens, temperature=temperature)

        gen = decode(model, out)

        results.append({
            "question": q,
            "prompt": prompt,
            "injection_start_token": int(start_pos),
            "metric_no_injection": {"first_token_diff": metric_no.first_token_diff, "full_string_diff": metric_no.full_string_diff},
            "metric_injection": {"first_token_diff": metric_inj.first_token_diff, "full_string_diff": metric_inj.full_string_diff},
            "sampled_response_injected": gen,
        })

    return results
