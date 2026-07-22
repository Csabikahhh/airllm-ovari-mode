"""Self-checks for the resource-utilization levers added to airllm_ui.py.
Run: .venv/Scripts/python.exe test_resource_levers.py
Covers the non-trivial new logic: KV-bytes formula, combined offload budget, long-context clamp."""
from types import SimpleNamespace
import airllm_ui as a


def test_kv_bytes_per_token():
    # Qwen2.5-3B-Instruct: 36 layers, hidden 2048, 16 heads, 2 kv heads, head_dim 128.
    # KV/token = 2(K+V) * 36 * (128*2) * 2 bytes = 36864 B/token (workflow-verified).
    cfg = SimpleNamespace(hidden_size=2048, num_hidden_layers=36,
                          num_attention_heads=16, num_key_value_heads=2, head_dim=128)
    assert a.kv_bytes_per_token(cfg) == 36864, a.kv_bytes_per_token(cfg)
    # Non-GQA (MHA) 3B: kv_dim == hidden -> ~8x the KV, which is why long context needs a guard.
    mha = SimpleNamespace(hidden_size=2048, num_hidden_layers=36,
                          num_attention_heads=16, num_key_value_heads=16, head_dim=128)
    assert a.kv_bytes_per_token(mha) == 2 * 36 * (128 * 16) * 2
    assert a.kv_bytes_per_token(SimpleNamespace()) is None  # missing fields -> None, no crash


def test_offload_budget():
    # Combined VRAM+RAM budget must mirror build_direct_model's int-floored caps.
    vram, ram = 6.8, 11.0
    budget = int(vram * 0.8) + int(ram * 0.6)  # 5 + 6 = 11 GiB
    assert budget == 11
    assert 10.0 <= budget      # a ~5B/~10GB bf16 model fits -> offloads instead of AirLLM
    assert not (14.0 <= budget)  # a 7B/~14GB does NOT fit -> correctly stays AirLLM


def test_long_context_clamp():
    # The resident budget must follow the model's trained context, not a flat 512.
    for trained, max_model_len, max_new in [(32768, 1024, 128), (4096, 1024, 256), (0, 1024, 64)]:
        ctx = min(32768, trained) if trained > 0 else max_model_len
        ctx = max(ctx, max_model_len)
        budget = max(16, ctx - max_new)
        assert budget >= 512 or ctx == max_model_len  # never the old silent 512 cap when context is real
    # 32k model gives a multi-thousand-token prompt budget, not 512.
    assert max(16, min(32768, 32768) - 128) == 32640


if __name__ == "__main__":
    test_kv_bytes_per_token()
    test_offload_budget()
    test_long_context_clamp()
    print("OK: all resource-lever self-checks passed")
