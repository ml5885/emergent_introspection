import json
import os
from collections import defaultdict
from statistics import mean

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformer_lens import HookedTransformer

matplotlib.rcParams["font.family"] = "serif"

PALETTE = {
    "fox1": "#DD8D29",
    "fox2": "#E2D200",
    "fox3": "#46ACC8",
    "fox4": "#E58601",
    "fox5": "#B40F20",
}

BG_FACE = "#fff6ec"
GRID_COLOR = "#777777"
ALPHA_FAINT = 0.25

def _read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def _ensure_outdir(outdir):
    os.makedirs(outdir, exist_ok=True)

def savefig(path):
    plt.savefig(path, bbox_inches="tight", dpi=180)
    plt.close()

def plot_scale_sweep(
    main_jsonl,
    control_jsonl,
    outdir,
    concept,
    layer_idx,
    title_model,
):
    _ensure_outdir(outdir)

    main_rows = [
        r for r in _read_jsonl(main_jsonl)
        if r["concept"] == concept and r["layer"] == layer_idx
    ]

    ctrl_rows = [
        r for r in _read_jsonl(control_jsonl)
        if r["concept"] == concept and r["layer"] == layer_idx
    ]

    main_by_scale = defaultdict(list)
    ctrl_inj_by_scale = defaultdict(list)
    ctrl_no_all = []

    for r in main_rows:
        main_by_scale[float(r["scale"])].append(float(r["full_string_logit_diff"]))

    for r in ctrl_rows:
        s = float(r["scale"])
        ctrl_inj_by_scale[s].append(float(r["metric_injection"]["full_string_diff"]))
        ctrl_no_all.append(float(r["metric_no_injection"]["full_string_diff"]))

    scales = sorted(main_by_scale.keys() | ctrl_inj_by_scale.keys())

    y_main = [
        mean(main_by_scale[s]) if s in main_by_scale else float("nan")
        for s in scales
    ]

    y_ctrl = [
        mean(ctrl_inj_by_scale[s]) if s in ctrl_inj_by_scale else float("nan")
        for s in scales
    ]

    if 0.0 in main_by_scale:
        baseline_main = mean(main_by_scale[0.0])
    else:
        baseline_main = mean(
            y
            for (s, ys) in main_by_scale.items()
            for y in ys
            if abs(s) == min(abs(k) for k in main_by_scale.keys())
        )

    baseline_ctrl = mean(ctrl_no_all) if ctrl_no_all else 0.0

    fig, ax = plt.subplots(figsize=(12, 7.2))
    ax.set_facecolor(BG_FACE)

    ax.plot(
        scales,
        y_main,
        marker="o",
        linewidth=3,
        markersize=6,
        color=PALETTE["fox3"],
        label="Introspection question",
    )

    ax.plot(
        scales,
        y_ctrl,
        marker="s",
        linewidth=3,
        markersize=6,
        color=PALETTE["fox1"],
        label="Control question",
    )

    ax.axhline(
        baseline_main,
        linestyle="--",
        linewidth=1.5,
        color=PALETTE["fox3"],
        alpha=0.6,
        label=f"Baseline introspection: {baseline_main:.2f}",
    )

    ax.axhline(
        baseline_ctrl,
        linestyle="--",
        linewidth=1.5,
        color=PALETTE["fox1"],
        alpha=0.6,
        label=f"Baseline control: {baseline_ctrl:.2f}",
    )

    ax.axhline(0.0, color="black", linewidth=1, alpha=0.5)

    ax.grid(True, linestyle=":", color=GRID_COLOR, alpha=0.35)
    ax.set_xlabel("Steering Vector Scale")
    ax.set_ylabel("Logit(Yes) - Logit(No)")
    ax.set_title(
        f"Steering vector scale sweep\n{title_model}\nContrastive Steering: "
        f"\"Hi! How are you?\" - \"HI! HOW ARE YOU?\" at layer {layer_idx}"
    )
    ax.legend(frameon=True)

    savefig(os.path.join(outdir, f"scale_sweep_L{layer_idx}_{concept}.png"))

def _tokenize(model, text):
    toks = model.to_tokens(text, prepend_bos=False)
    return toks if toks.ndim == 2 else toks.unsqueeze(0)

def _char_to_token_index(model, text, char_index):
    toks = model.to_tokens(text, prepend_bos=False)[0]
    for i in range(1, toks.shape[0] + 1):
        s = model.to_string(toks[:i])
        if len(s) >= char_index:
            return i - 1
    return toks.shape[0] - 1

def _find_injection_start_token(model, prompt):
    marker = "\n\nTrial 1"
    idx = prompt.find(marker)
    if idx == -1:
        idx = prompt.rfind("\n\n")
        if idx == -1:
            idx = 0
    return _char_to_token_index(model, prompt, idx)

def _find_last_assistant_colon_token(model, text):
    idx = text.rfind("Assistant:")
    if idx == -1:
        toks = model.to_tokens(text, prepend_bos=False)[0]
        return toks.shape[0] - 1
    colon_idx = idx + len("Assistant:") - 1
    return _char_to_token_index(model, text, colon_idx)

def _activation_at_colon(model, text, layer_idx):
    toks = _tokenize(model, text)
    hook_name = f"blocks.{layer_idx}.hook_resid_pre"
    _, cache = model.run_with_cache(toks, names_filter=[hook_name])
    pos = _find_last_assistant_colon_token(model, text)
    return cache[hook_name][0, pos, :].detach()

def _contrast_vector(model, layer_idx, pos_prompt, neg_prompt):
    pos = _activation_at_colon(model, pos_prompt, layer_idx)
    neg = _activation_at_colon(model, neg_prompt, layer_idx)
    v = pos - neg
    return v / (v.norm() + 1e-8)

def _make_resid_add_hook(vector, start_pos, scale):
    v = vector.detach()

    def hook(resid, _hook):
        seq = resid.shape[1]
        mask = torch.arange(seq, device=resid.device) >= start_pos
        resid += scale * v[None, None, :] * mask[None, :, None]
        return resid

    return hook

def _yes_no_ids(model):
    yes = model.tokenizer.encode(" Yes", add_special_tokens=False) or model.tokenizer.encode("Yes", add_special_tokens=False)
    no = model.tokenizer.encode(" No", add_special_tokens=False) or model.tokenizer.encode("No", add_special_tokens=False)
    return yes[0], no[0]

def _full_string_logitdiff(model, prefix_tokens, fwd_hooks=()):
    def _logprob(s):
        ids = model.tokenizer.encode(s, add_special_tokens=False)
        total = 0.0
        cur = prefix_tokens.clone()
        for t in ids:
            with model.hooks(fwd_hooks=fwd_hooks):
                logits = model(cur)
            logp = torch.log_softmax(logits[0, -1, :], dim=-1)[t].item()
            cur = torch.cat([cur, torch.tensor([[t]], device=cur.device)], dim=1)
            total += logp
        return total
    return _logprob(" Yes") - _logprob(" No")

def plot_layer_sweep(
    model,
    *,
    concept_pos_prompt,
    concept_neg_prompt,
    main_prompt,
    control_questions,
    scale,
    outdir,
    title_model,
):
    _ensure_outdir(outdir)

    n_layers = model.cfg.n_layers
    main_tokens = _tokenize(model, main_prompt)
    main_inj_start = _find_injection_start_token(model, main_prompt)

    baseline_main = _full_string_logitdiff(model, main_tokens, fwd_hooks=())
    ctrl_tokens = [_tokenize(model, f"Human: {q}\n\nAssistant:") for q in control_questions]
    baseline_ctrl = mean(_full_string_logitdiff(model, t, fwd_hooks=()) for t in ctrl_tokens)

    main_vals = []
    ctrl_vals = []

    for L in range(n_layers):
        vec = _contrast_vector(model, L, concept_pos_prompt, concept_neg_prompt).to(model.cfg.device)
        hook_name = f"blocks.{L}.hook_resid_pre"
        hook = _make_resid_add_hook(vec, start_pos=main_inj_start, scale=scale)

        y_main = _full_string_logitdiff(model, main_tokens, fwd_hooks=[(hook_name, hook)])
        main_vals.append(y_main)

        vals = []
        for t in ctrl_tokens:
            start_pos = _find_injection_start_token(model, "Human: ?\n\nAssistant:")
            h = _make_resid_add_hook(vec, start_pos=start_pos, scale=scale)
            vals.append(_full_string_logitdiff(model, t, fwd_hooks=[(hook_name, h)]))
        ctrl_vals.append(mean(vals))

    layers = list(range(n_layers))
    fig, ax = plt.subplots(figsize=(12, 7.2))
    ax.set_facecolor(BG_FACE)

    ax.plot(layers, main_vals, marker="o", linewidth=2.5, color=PALETTE["fox3"], label="Introspection question")
    ax.plot(layers, ctrl_vals, marker="s", linewidth=2.5, color=PALETTE["fox1"], label="Control question")

    ax.axhline(baseline_main, linestyle="--", linewidth=1.5, color=PALETTE["fox3"], alpha=0.6,
               label=f"Baseline introspection: {baseline_main:.2f}")
    ax.axhline(baseline_ctrl, linestyle="--", linewidth=1.5, color=PALETTE["fox1"], alpha=0.6,
               label=f"Baseline control: {baseline_ctrl:.2f}")
    ax.axhline(0.0, color="black", linewidth=1, alpha=0.5)

    ax.grid(True, linestyle=":", color=GRID_COLOR, alpha=0.35)
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Logit(Yes) - Logit(No)")
    ax.set_title(
        f"LLM Introspection Experiment: {title_model}\n"
        f"Contrastive Steering: \"Hi! How are you?\" vs \"HI! HOW ARE YOU?\" (strength={scale})"
    )
    ax.legend(frameon=True)

    savefig(os.path.join(outdir, f"layer_sweep_strength_{scale:g}.png"))
