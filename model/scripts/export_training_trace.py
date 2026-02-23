"""
Export training trace JSON for frontend visualization.
Re-trains the model while recording loss, learning rate, and parameter snapshots at each step.
"""

import math
import json
import random
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from in_main import (
    Value, load_dataset, build_tokenizer, build_config, init_model,
    gpt, softmax, matrix, linear, rmsnorm,
    RANDOM_SEED, NUM_STEPS, LEARNING_RATE, BETA1, BETA2, EPS_ADAM, DATA_PATH
)

def main():
    random.seed(RANDOM_SEED)
    docs, dataset_names = load_dataset(DATA_PATH)
    tokenizer = build_tokenizer(docs)
    config = build_config()
    state_dict, params = init_model(tokenizer["vocab_size"], config)

    bos = tokenizer["BOS"]
    stoi = tokenizer["stoi"]
    uchars = tokenizer["uchars"]
    block_size = config["block_size"]
    n_layer = config["n_layer"]
    n_embd = config["n_embd"]

    # Pick parameters to track
    param_options = [
        {"id": "wte_0", "label": f"wte[0] ({uchars[0]})", "matrix": "wte", "row_index": 0,
         "token_char_nfd": uchars[0], "token_char_display": uchars[0]},
        {"id": "wte_1", "label": f"wte[1] ({uchars[1]})", "matrix": "wte", "row_index": 1,
         "token_char_nfd": uchars[1], "token_char_display": uchars[1]},
        {"id": "wpe_0", "label": "wpe[0]", "matrix": "wpe", "row_index": 0},
        {"id": "attn_wq_0", "label": "attn_wq[0]", "matrix": "attn_wq", "row_index": 0},
    ]

    def get_param_row(pid):
        m = pid["matrix"]
        r = pid["row_index"]
        if m == "wte": return state_dict["wte"][r]
        if m == "wpe": return state_dict["wpe"][r]
        if m == "attn_wq": return state_dict["layer0.attn_wq"][r]
        if m == "lm_head": return state_dict["lm_head"][r]
        return None

    step_options = [0, 250, 500, 999]

    # Record initial state
    steps = [{
        "step": 0,
        "loss": None,
        "learning_rate": LEARNING_RATE,
        "word": "",
        "params": {
            pid["id"]: {
                "grad": [0.0] * n_embd,
                "after": [v.data for v in get_param_row(pid)]
            } for pid in param_options
        }
    }]

    # Training with tracing
    m_buf = [0.0] * len(params)
    v_buf = [0.0] * len(params)

    for step in range(NUM_STEPS):
        doc = docs[step % len(docs)]
        tokens = [bos] + [stoi[ch] for ch in doc] + [bos]
        n = min(block_size, len(tokens) - 1)

        keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
        losses = []
        for pos_id in range(n):
            token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
            logits = gpt(token_id, pos_id, keys, values, state_dict, config)
            probs = softmax(logits)
            losses.append(-probs[target_id].log())
        loss = (1 / n) * sum(losses)
        loss.backward()

        lr_t = LEARNING_RATE * (1 - step / NUM_STEPS)

        # Capture gradients before update
        param_grads = {}
        for pid in param_options:
            row = get_param_row(pid)
            param_grads[pid["id"]] = [v.grad for v in row]

        # Adam update
        for i, p in enumerate(params):
            m_buf[i] = BETA1 * m_buf[i] + (1 - BETA1) * p.grad
            v_buf[i] = BETA2 * v_buf[i] + (1 - BETA2) * p.grad ** 2
            m_hat = m_buf[i] / (1 - BETA1 ** (step + 1))
            v_hat = v_buf[i] / (1 - BETA2 ** (step + 1))
            p.data -= lr_t * m_hat / (v_hat ** 0.5 + EPS_ADAM)
            p.grad = 0

        # NFC display name
        import unicodedata
        display_word = unicodedata.normalize("NFC", doc)

        step_record = {
            "step": step + 1,
            "loss": round(loss.data, 6),
            "learning_rate": round(lr_t, 8),
            "word": display_word,
            "params": {
                pid["id"]: {
                    "grad": [round(g, 8) for g in param_grads[pid["id"]]],
                    "after": [round(v.data, 8) for v in get_param_row(pid)]
                } for pid in param_options
            }
        }
        steps.append(step_record)

        if (step + 1) % 100 == 0:
            print(f"step {step+1:4d}/{NUM_STEPS} | loss {loss.data:.4f}")

    trace = {
        "num_steps": NUM_STEPS,
        "step_options": step_options,
        "optimizer": {
            "name": "Adam",
            "learning_rate": LEARNING_RATE,
            "beta1": BETA1,
            "beta2": BETA2,
            "eps": EPS_ADAM,
        },
        "parameter_options": param_options,
        "steps": steps,
    }

    output_path = Path(__file__).resolve().parents[1] / ".." / "app" / "public" / "data" / "in_training_trace.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(trace, f, ensure_ascii=False)

    file_size = output_path.stat().st_size
    print(f"\nSaved training trace: {output_path.resolve()}")
    print(f"File size: {file_size / 1024:.1f} KB")
    print(f"Steps: {len(steps)}")


if __name__ == "__main__":
    main()
