"""Tests for hidra.rebind. Changing the review rate must not drop the fine-tuned checkpoint.
Runs under pytest, or directly: python tests/test_hidra_rebind.py
"""
from laras_labeler import hidra

TUNED = {"lab": "LyricalHare", "action": "approach", "collapse": "directed", "rate": 0.15,
         "weights": "/m/{config}__x.pkl", "finetune_mode": "head"}


def test_new_rate_keeps_checkpoint():
    out = hidra.rebind(TUNED, "LyricalHare", "approach", "directed", 0.05)
    assert out["rate"] == 0.05
    assert out["weights"] == TUNED["weights"] and out["finetune_mode"] == "head"


def test_new_collapse_keeps_checkpoint():
    out = hidra.rebind(TUNED, "LyricalHare", "approach", "scene", 0.15)
    assert out["collapse"] == "scene" and out["weights"] == TUNED["weights"]


def test_different_head_drops_checkpoint():
    assert "weights" not in hidra.rebind(TUNED, "GroovyShrew", "approach", "directed", 0.15)
    assert "weights" not in hidra.rebind(TUNED, "LyricalHare", "attack", "directed", 0.15)


def test_first_bind_has_no_checkpoint():
    out = hidra.rebind(None, "LyricalHare", "approach", "directed", 0.15)
    assert out == {"lab": "LyricalHare", "action": "approach", "collapse": "directed", "rate": 0.15}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
