"""
harm_directions/extraction.py
---------------------------
Residual-stream activation extraction from transformer models.

Uses forward hooks to capture the output of each transformer block,
then applies element-wise max pooling over the token dimension.
"""

from __future__ import annotations

from typing import Literal, Any

import numpy as np
import torch

# from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import PreTrainedModel, PreTrainedTokenizerBase


def extract_activations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    layer: int,
    pooling: Literal["max", "mean", "last"] = "max",
    batch_size: int = 1,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Extract residual-stream activations at a given layer for a list of prompts.

    Parameters
    ----------
    model : AutoModelForCausalLM
        The language model.
    tokenizer : AutoTokenizer
        Corresponding tokenizer.
    prompts : list[str]
        Input prompts.
    layer : int
        Transformer block index (0-indexed).
    pooling : {"max", "mean", "last"}
        How to aggregate over the token dimension.
        - "max": element-wise maximum (default; preserves peak activations).
        - "mean": element-wise mean.
        - "last": last token position.
    batch_size : int
        Number of prompts per forward pass.
    show_progress : bool
        Print progress every 50 prompts.

    Returns
    -------
    np.ndarray of shape (n_prompts, D)
        Pooled activation vectors.
    """
    device = next(model.parameters()).device
    results = []

    # Identify the correct layer module
    layer_module = _get_layer_module(model, layer)

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        if show_progress and (i + 1) % 50 == 0:
            print(f"  Extracting: {i + 1}/{len(prompts)}")

        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)

        captured: dict = {}

        def hook_fn(_module, _inputs, output, captured=captured):
            # output may be a tuple; take the first element (hidden states)
            if isinstance(output, tuple):
                captured["hidden"] = output[0].detach()
            else:
                captured["hidden"] = output.detach()

        handle = layer_module.register_forward_hook(hook_fn)
        with torch.no_grad():
            model(**inputs)
        handle.remove()

        hidden = captured["hidden"]  # (batch, seq_len, D)
        attention_mask = inputs.get("attention_mask", None)

        for j in range(hidden.shape[0]):
            h = hidden[j]  # (seq_len, D)
            if attention_mask is not None:
                mask = attention_mask[j].bool()
                h = h[mask]  # only non-padding tokens

            if pooling == "max":
                pooled = h.max(dim=0).values
            elif pooling == "mean":
                pooled = h.mean(dim=0)
            elif pooling == "last":
                pooled = h[-1]
            else:
                raise ValueError(f"Unknown pooling: {pooling}")

            results.append(pooled.cpu().float().numpy())

    return np.stack(results)


def extract_all_layers(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    pooling: Literal["max", "mean", "last"] = "max",
    show_progress: bool = True,
) -> np.ndarray:
    """
    Extract activations from all layers at once.

    Returns
    -------
    np.ndarray of shape (n_prompts, n_layers, D)
    """
    device = next(model.parameters()).device
    n_layers = _count_layers(model)
    layer_modules = [_get_layer_module(model, layer_idx) for layer_idx in range(n_layers)]

    all_results = []

    for i, prompt in enumerate(prompts):
        if show_progress and (i + 1) % 50 == 0:
            print(f"  Extracting: {i + 1}/{len(prompts)}")

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        captured: dict[int, Any] = {}

        handles = []

        def make_hook(layer_idx, captured=captured):
            def hook_fn(_module, _inputs, output):
                out = output[0] if isinstance(output, tuple) else output
                captured[layer_idx] = out.detach()

            return hook_fn

        for layer_idx, mod in enumerate(layer_modules):
            handles.append(mod.register_forward_hook(make_hook(layer_idx)))

        with torch.no_grad():
            model(**inputs)

        for h in handles:
            h.remove()

        layers_pooled = []
        for layer_idx in range(n_layers):
            h = captured[layer_idx].squeeze(0)  # (seq_len, D)
            if pooling == "max":
                pooled = h.max(dim=0).values
            elif pooling == "mean":
                pooled = h.mean(dim=0)
            elif pooling == "last":
                pooled = h[-1]
            else:
                raise ValueError(f"Unknown pooling: {pooling}")
            layers_pooled.append(pooled.cpu().float().numpy())

        all_results.append(np.stack(layers_pooled))

    return np.stack(all_results)


# ---------------------------------------------------------------------------
# Architecture-agnostic layer access
# ---------------------------------------------------------------------------


def _get_layer_module(model, layer_idx: int) -> torch.nn.Module:
    """Return the transformer block at the given index, architecture-agnostic."""
    if hasattr(model, "model"):
        inner = model.model
        if hasattr(inner, "layers"):
            return inner.layers[layer_idx]
        elif hasattr(inner, "decoder") and hasattr(inner.decoder, "layers"):
            return inner.decoder.layers[layer_idx]
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h[layer_idx]
    raise ValueError(
        f"Cannot find transformer layers in model of type {type(model).__name__}. "
        "Please check the model architecture."
    )


def _count_layers(model) -> int:
    """Count the number of transformer blocks."""
    if hasattr(model, "model"):
        inner = model.model
        if hasattr(inner, "layers"):
            return len(inner.layers)
        elif hasattr(inner, "decoder") and hasattr(inner.decoder, "layers"):
            return len(inner.decoder.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return len(model.transformer.h)
    raise ValueError(f"Cannot count layers in model of type {type(model).__name__}.")
