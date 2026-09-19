"""Unit test OFFLINE cho cơ chế fallback model dạng vòng tròn (không gọi API thật)."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from conftest import REPO

# auto_model_relay nằm trong thư mục cac_hien_tuong... (import env_paths/knowledge_graph cùng thư mục).
_HT_DIR = REPO / "src/test_set/datasets_qa/benchmark_qa/speech/cac_hien_tuong_dac_biet_trong_tieng_viet"
sys.path.insert(0, str(_HT_DIR))
import auto_model_relay as amr  # noqa: E402


class _FakeChat:
    def __init__(self, fail_models):
        self.fail_models = set(fail_models)
        self.calls: list[str] = []

    def create(self, *, model, messages, temperature, max_tokens):
        self.calls.append(model)
        if model in self.fail_models:
            raise RuntimeError("quota exhausted")
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=f"OK:{model}"))])


class _FakeClient:
    def __init__(self, fail_models=()):
        self.chat = SimpleNamespace(completions=_FakeChat(fail_models))


def _call(client, candidates=None, key="test:openai", base_model=None):
    return amr.call_model(
        "openai", "prompt", openai_client=client,
        openai_model=base_model or amr.DEFAULT_OPENAI_MODEL,
        model_candidates=candidates, rotation_key=key,
    )


def setup_function(_):
    amr._MODEL_ROTATION.clear()


def test_fallback_nhay_sang_model_ke_khi_model_dau_loi():
    first, second = amr.DEBATE_OPENAI_MODELS[0], amr.DEBATE_OPENAI_MODELS[1]
    client = _FakeClient(fail_models={first})
    out = _call(client)
    assert out == f"OK:{second}"
    assert client.chat.completions.calls == [first, second]
    # thành công ở model thứ 2 -> lần sau bắt đầu luôn ở model đó (đổi-khi-lỗi, giữ model)
    out2 = _call(client)
    assert out2 == f"OK:{second}"
    assert client.chat.completions.calls == [first, second, second]


def test_ca_pool_loi_thi_raise_va_con_tro_tien_1_buoc():
    pool = amr.DEBATE_OPENAI_MODELS
    client = _FakeClient(fail_models=set(pool))
    with pytest.raises(RuntimeError):
        _call(client, candidates=pool)
    # đã thử đủ cả pool (vòng tròn)
    assert set(client.chat.completions.calls) == set(pool)
    # con trỏ đã tiến 1 bước cho lần retry sau
    assert amr._MODEL_ROTATION["test:openai"] == 1


def test_pool_cung_ho_va_tach_theo_giai_doan():
    # OpenAI: chỉ model họ OpenAI/GPT (cx hoặc cl/openai), không lẫn Gemini/Claude/khác.
    for m in amr.DEBATE_OPENAI_MODELS + amr.FILTER_OPENAI_MODELS:
        assert m.startswith(("cx/", "cl/openai/")), m
    # Gemini: chỉ họ gemini.
    for m in amr.DEBATE_GEMINI_MODELS + amr.FILTER_GEMINI_MODELS:
        assert "gemini" in m, m
    # Pool filter phải chứa model khoẻ khác pool debate (tách giai đoạn).
    assert amr.FILTER_GEMINI_MODELS[0] != amr.DEBATE_GEMINI_MODELS[0]
    assert amr.FILTER_OPENAI_MODELS[0] != amr.DEBATE_OPENAI_MODELS[0]


def test_filter_pool_duoc_dung_khi_truyen_candidates():
    client = _FakeClient()
    out = _call(client, candidates=amr.FILTER_OPENAI_MODELS, key="test:filter",
                base_model=amr.FILTER_OPENAI_MODELS[0])
    assert out == f"OK:{amr.FILTER_OPENAI_MODELS[0]}"
