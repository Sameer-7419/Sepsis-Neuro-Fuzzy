import torch
from src.anfis_model import NeuroFuzzySepsis

def test_forward():
    m=NeuroFuzzySepsis()
    y=m(torch.zeros(4,7))
    assert y.shape==(4,)
    assert torch.isfinite(y).all()

def test_rule_count():
    assert len(NeuroFuzzySepsis().rules)==128
