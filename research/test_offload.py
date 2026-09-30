from accelerate import disk_offload
import pytest
import torch
from moep.profiler import profile

@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_disk_offload_matches_resident(tiny_ckpt, tiny_adapter, tmp_path, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from conftest import build_tiny
    model = build_tiny()
    ids = torch.arange(32).reshape(1, 32)
    expected, layers = profile(model, tiny_adapter, ids, batch_size=1, log_every=0)
    disk_offload(model, tmp_path / 'disk', execution_device=torch.device(device))
    forwards = [(block.experts, block.experts._old_forward) for _, block in tiny_adapter.moe_blocks(model)]
    observed, observed_layers = profile(model, tiny_adapter, ids, batch_size=1, log_every=0)
    assert layers == observed_layers
    assert expected.total_tokens == observed.total_tokens
    for key in expected.tensors:
        assert torch.isfinite(observed.tensors[key]).all()
        assert torch.allclose(expected.tensors[key], observed.tensors[key].cpu(), atol=1e-4, rtol=1e-4), key
    assert all(module._old_forward is original for module, original in forwards)
