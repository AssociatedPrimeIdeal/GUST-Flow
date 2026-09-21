from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch

def _require_cuda_float32(*tensors: torch.Tensor) -> None:
    for tensor in tensors:
        if not tensor.is_cuda:
            raise ValueError('GUST-Flow CUDA inputs must be CUDA tensors')
        if tensor.dtype != torch.float32:
            raise ValueError('GUST-Flow currently supports float32 tensors only')

def _cupy_view(cp, tensor: torch.Tensor):
    return cp.from_dlpack(tensor.detach())

@lru_cache(maxsize=1)
def _kernels():
    try:
        import cupy as cp
    except ImportError as exc:
        raise RuntimeError('The CUDA backend requires CuPy. Install cupy-cuda12x/13x for the installed CUDA version.') from exc
    source = Path(__file__).with_name('kernels.cu').read_text(encoding='ascii')
    names = ('phase_l1_forward', 'phase_l1_backward')
    module = cp.RawModule(code=source, options=('--std=c++14', '--use_fast_math'), name_expressions=names)
    return (cp, *(module.get_function(name) for name in names))

class _FusedPhaseL1(torch.autograd.Function):

    @staticmethod
    def forward(ctx, phase, wrapped, confidence, wt, wfe, wpe, wspe):
        _require_cuda_float32(phase, wrapped, confidence)
        if phase.shape != wrapped.shape or phase.shape != confidence.shape:
            raise ValueError('phase, wrapped, and confidence must have the same shape')
        if phase.ndim != 5:
            raise ValueError('phase must use [channel, time, FE, PE, SPE] layout')
        phase = phase.contiguous()
        wrapped = wrapped.contiguous()
        confidence = confidence.contiguous()
        loss = phase.new_zeros(())
        _, time_size, fe_size, pe_size, spe_size = phase.shape
        cp, forward_kernel, _ = _kernels()
        threads = 256
        blocks = min((phase.numel() + threads - 1) // threads, 4096)
        with cp.cuda.Device(phase.device.index), cp.cuda.ExternalStream(torch.cuda.current_stream(phase.device).cuda_stream):
            forward_kernel((blocks,), (threads,), (_cupy_view(cp, phase), _cupy_view(cp, wrapped), _cupy_view(cp, confidence), _cupy_view(cp, loss), np.int64(phase.numel()), np.int32(time_size), np.int32(fe_size), np.int32(pe_size), np.int32(spe_size), np.float32(wt), np.float32(wfe), np.float32(wpe), np.float32(wspe)), shared_mem=threads * 4)
        ctx.save_for_backward(phase, wrapped, confidence)
        ctx.weights = (wt, wfe, wpe, wspe)
        return loss

    @staticmethod
    def backward(ctx, grad_loss):
        phase, wrapped, confidence = ctx.saved_tensors
        _, time_size, fe_size, pe_size, spe_size = phase.shape
        grad_phase = torch.empty_like(phase)
        wt, wfe, wpe, wspe = ctx.weights
        cp, _, backward_kernel = _kernels()
        threads = 256
        blocks = min((phase.numel() + threads - 1) // threads, 65535)
        with cp.cuda.Device(phase.device.index), cp.cuda.ExternalStream(torch.cuda.current_stream(phase.device).cuda_stream):
            backward_kernel((blocks,), (threads,), (_cupy_view(cp, phase), _cupy_view(cp, wrapped), _cupy_view(cp, confidence), _cupy_view(cp, grad_loss.contiguous()), _cupy_view(cp, grad_phase), np.int64(phase.numel()), np.int32(time_size), np.int32(fe_size), np.int32(pe_size), np.int32(spe_size), np.float32(wt), np.float32(wfe), np.float32(wpe), np.float32(wspe)))
        return (grad_phase, None, None, None, None, None, None)

def fused_phase_l1(phase: torch.Tensor, wrapped: torch.Tensor, confidence: torch.Tensor, weights: tuple[float, float, float, float]=(1.0, 1.0, 1.0, 1.0)) -> torch.Tensor:
    if len(weights) != 4:
        raise ValueError('weights must contain (T, FE, PE, SPE)')
    return _FusedPhaseL1.apply(phase, wrapped, confidence, *weights)
