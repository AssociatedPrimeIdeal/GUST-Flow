from contextlib import contextmanager
from dataclasses import dataclass
import math
import random
import time

import numpy as np
import torch
from torch import nn

from .cuda_backend import fused_phase_l1
from .local_renderer import LocalGaussianRenderer


@contextmanager
def full_precision_matmul():
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def wrap_phase(value):
    period = 2.0 * torch.pi
    return value - torch.div(value + torch.pi, period, rounding_mode="floor") * period


def _axis_angle_to_matrix(rotation):
    x, y, z = rotation.unbind(dim=-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1)
    skew = skew.reshape(*rotation.shape[:-1], 3, 3)
    theta_sq = rotation.square().sum(dim=-1, keepdim=True)
    theta = theta_sq.clamp_min(1e-8).sqrt()
    sinc = torch.sinc(theta / torch.pi)
    factor = torch.where(theta_sq > 1e-8,
                         (1.0 - torch.cos(theta)) / theta_sq.clamp_min(1e-8),
                         torch.full_like(theta_sq, 0.5))
    identity = torch.eye(3, device=rotation.device, dtype=rotation.dtype)
    identity = identity.expand(*rotation.shape[:-1], 3, 3)
    return identity + sinc[..., None] * skew + factor[..., None] * (skew @ skew)


def _sample_centers(confidence, count):
    spatial = confidence.detach().mean(dim=(0, 1)).clamp_min(0.0)
    probabilities = spatial.reshape(-1)
    nonzero = int(torch.count_nonzero(probabilities).item())
    if float(probabilities.sum().item()) <= 0.0:
        probabilities = torch.ones_like(probabilities)
        nonzero = probabilities.numel()
    indices = torch.multinomial(probabilities, count, replacement=nonzero < count)
    fe, pe, spe = spatial.shape
    fe_index = torch.div(indices, pe * spe, rounding_mode="floor")
    remainder = indices - fe_index * pe * spe
    pe_index = torch.div(remainder, spe, rounding_mode="floor")
    spe_index = remainder - pe_index * spe
    indices = torch.stack((fe_index, pe_index, spe_index), dim=-1).to(confidence.dtype)
    denominators = (confidence.new_tensor((fe, pe, spe)) - 1.0).clamp_min(1.0)
    centers = 2.0 * indices / denominators - 1.0
    centers = centers + (torch.rand_like(centers) - 0.5) * (2.0 / denominators)
    return centers.clamp(-0.98, 0.98)


def _nrmse(prediction, reference, mask):
    selected = mask.expand_as(prediction) != 0
    error = torch.sqrt(torch.mean((prediction[selected] - reference[selected]).square()))
    scale = reference[selected].max() - reference[selected].min()
    return float((error / (scale + 1e-8)).item())


class GaussianSplat(nn.Module):
    def __init__(self, spatial_shape, frames, initial_centers, init_sigma, venc):
        super().__init__()
        self.spatial_shape = tuple(spatial_shape)
        self.frames = int(frames)
        self.num_primitives = self.active_primitives = len(initial_centers)
        self.channels = 3
        axes = [torch.linspace(-1.0, 1.0, n) for n in self.spatial_shape]
        coords = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)
        self.register_buffer("coords", coords)
        self.center_raw = nn.Parameter(torch.atanh(initial_centers.clamp(-0.98, 0.98)))
        sigma = torch.as_tensor(init_sigma, dtype=torch.float32).view(1, 3)
        sigma = sigma.repeat(self.num_primitives, 1)
        self.log_scales = nn.Parameter(torch.log(torch.expm1((sigma - .015).clamp_min(1e-4))))
        self.rotation_raw = nn.Parameter(torch.zeros(self.num_primitives, 3))
        coefficient = .01 * torch.randn(3, self.frames, self.num_primitives)
        coefficient = coefficient[:, :1].repeat(1, self.frames, 1)
        phase_scale = venc.mean() / venc
        initial_velocity = coefficient[:, 0, :] / phase_scale[:, None]
        amplitude = initial_velocity.norm(dim=0).clamp_min(1e-12)
        direction = initial_velocity / amplitude[None, :]
        q = coefficient.new_zeros(1, self.frames, self.num_primitives)
        q[0] = amplitude[None, :]
        self.coeff = nn.Parameter(q)
        self.direction_raw = nn.Parameter(direction[:, None, :])
        self.register_buffer("velocity_phase_scale", phase_scale)
        for name in ("spatial_scale_offset", "temporal_transform", "coefficient_volume_reference"):
            self.register_buffer(name, None)

    def spatial_scales(self, capped=True):
        offset = .015 if self.spatial_scale_offset is None else self.spatial_scale_offset
        scales = torch.nn.functional.softplus(self.log_scales) + offset
        return scales.clamp_max(1.0) if capped else scales

    @torch.no_grad()
    def configure(self):
        initial_scales = self.spatial_scales(capped=False)
        self.spatial_scale_offset = initial_scales.new_tensor(.005)
        self.log_scales.copy_(torch.log(torch.expm1(initial_scales - self.spatial_scale_offset)))
        with full_precision_matmul():
            identity = torch.eye(self.frames, device=self.coeff.device, dtype=self.coeff.dtype)
            laplacian = 2 * identity - torch.roll(identity, 1, 0) - torch.roll(identity, -1, 0)
            transform = torch.linalg.solve(identity + 10.0 * laplacian, identity)
            latent = torch.linalg.solve(transform, self.coeff.permute(1, 0, 2).reshape(self.frames, -1))
            self.coeff.copy_(latent.reshape(self.frames, 1, -1).permute(1, 0, 2))
            self.temporal_transform = transform
            self.coefficient_volume_reference = self.spatial_scales().prod(-1).clone()
        self.local_renderer = LocalGaussianRenderer(self.spatial_shape, self.coords)

    def velocity_directions(self):
        return self.direction_raw / self.direction_raw.norm(dim=0, keepdim=True).clamp_min(1e-8)

    def temporal_amplitudes(self):
        with full_precision_matmul():
            amplitude = torch.einsum("ts,ksn->ktn", self.temporal_transform, self.coeff)
        ratio = self.coefficient_volume_reference / self.spatial_scales().prod(-1)
        return amplitude * ratio.pow(1.0)

    def temporal_weights(self):
        with full_precision_matmul():
            velocity = torch.einsum("ckn,ktn->ctn", self.velocity_directions(), self.temporal_amplitudes())
            return velocity * self.velocity_phase_scale[:, None, None]

    def forward(self):
        centers = torch.tanh(self.center_raw)
        scales = self.spatial_scales()
        rotation = _axis_angle_to_matrix(self.rotation_raw)
        weights = self.temporal_weights()
        packed = torch.cat((weights.reshape(1, -1, self.num_primitives),
                            weights.new_ones(1, 1, self.num_primitives)), dim=1)
        rendered = self.local_renderer(centers, scales, rotation, packed)[0]
        coverage = rendered[-1:].clamp_min(1e-8)
        return (rendered[:-1] / coverage).reshape(3, self.frames, *self.spatial_shape)


@dataclass
class GUSTResult:
    recovered: torch.Tensor
    phase: torch.Tensor
    model: GaussianSplat
    history: dict


class GUSTFlow:
    def __init__(self, venc=50.0, voxel_spacing=(1., 1., 1.), num_primitives=8192,
                 num_iter=1000, lr=.03, device="cuda", seed=314159):
        self.venc = torch.as_tensor(venc, dtype=torch.float32, device="cpu").flatten()
        if self.venc.numel() == 1:
            self.venc = self.venc.repeat(3)
        if self.venc.shape != (3,) or not torch.isfinite(self.venc).all() or (self.venc <= 0).any():
            raise ValueError("venc must contain one or three finite positive values")
        self.voxel_spacing = tuple(float(s) for s in voxel_spacing)
        if len(self.voxel_spacing) != 3 or any(not math.isfinite(s) or s <= 0 for s in self.voxel_spacing):
            raise ValueError("voxel_spacing must contain three finite positive values")
        self.num_primitives, self.num_iter = int(num_primitives), int(num_iter)
        self.lr, self.seed, self.device = float(lr), int(seed), torch.device(device)
        if self.num_primitives <= 0 or self.num_iter <= 0 or not math.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("num_primitives, num_iter and lr must be positive")

    def fit(self, wrapped_phase, weightmask, *, center_confidence, gt_velocity=None,
            segmask=None, eval_every=0, callback=None):
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("GUSTFlow requires a CUDA device")
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.backends.cuda.matmul.allow_tf32 = True

        def tensor(value):
            if isinstance(value, np.ndarray) and any(s < 0 for s in value.strides):
                value = np.ascontiguousarray(value)
            return torch.as_tensor(value, dtype=torch.float32, device=self.device)

        wrapped = tensor(wrapped_phase)
        if wrapped.ndim != 5 or wrapped.shape[0] != 3 or min(wrapped.shape) <= 0:
            raise ValueError("wrapped_phase must have shape [3, time, FE, PE, SPE]")
        weight = torch.broadcast_to(tensor(weightmask), wrapped.shape).contiguous().clamp_min(0.)
        confidence = tensor(center_confidence)
        if tuple(confidence.shape[-3:]) != tuple(wrapped.shape[-3:]) or confidence.numel() != math.prod(wrapped.shape[-3:]):
            raise ValueError("center_confidence must be a spatial std(PCMRA) map matching the input FOV")
        confidence = confidence.reshape(1, 1, *wrapped.shape[-3:])
        if not all(bool(torch.isfinite(t).all()) for t in (wrapped, weight, confidence)):
            raise ValueError("Phase, weight and center confidence must be finite")
        weight = weight / (weight.max() + 1e-12)
        normalizer = weight.sum().detach().clamp_min(1.0)
        venc = self.venc.to(self.device)[:, None, None, None, None]
        reference = mask = None
        if (gt_velocity is None) != (segmask is None):
            raise ValueError("gt_velocity and segmask must be provided together")
        if gt_velocity is not None:
            reference = torch.broadcast_to(tensor(gt_velocity), wrapped.shape)
            mask = torch.broadcast_to(tensor(segmask), wrapped.shape)
            if not bool((mask != 0).any()):
                raise ValueError("Evaluation mask must not be empty")

        ratio = tuple(max(self.voxel_spacing) / s for s in self.voxel_spacing)
        tv_weights = (1.0, *ratio)
        init_sigma = tuple(.06 * r for r in ratio)
        initial_centers = _sample_centers(confidence, self.num_primitives)
        model = GaussianSplat(wrapped.shape[-3:], wrapped.shape[1], initial_centers,
                              init_sigma, self.venc).to(self.device)
        model.configure()

        def objective(phase):
            return fused_phase_l1(phase, wrapped, weight, tv_weights) / normalizer

        model.zero_grad(set_to_none=True)
        objective(model()).backward()
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
            lr_lambda=lambda step: .05 + .95 * .5 * (1.0 + math.cos(math.pi * step / self.num_iter)))
        history = dict(eval_iterations=[], eval_loss=[], nrmse=[], nonfinite_steps=[],
                       kernel_snapshots=[], initial_centers=initial_centers.detach().cpu())
        started = time.perf_counter()
        for iteration in range(1, self.num_iter + 1):
            optimizer.zero_grad(set_to_none=True)
            loss = objective(model())
            loss.backward()
            gradient_norm = torch.nn.utils.get_total_norm([p.grad for p in model.parameters() if p.grad is not None])
            if bool((torch.isfinite(loss.detach()) & torch.isfinite(gradient_norm.detach())).item()):
                optimizer.step()
            else:
                history["nonfinite_steps"].append(iteration)
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.zero_()
            scheduler.step()
            if (eval_every > 0 and iteration % eval_every == 0) or iteration == self.num_iter:
                with torch.no_grad():
                    phase = model()
                    recovered = (phase + wrap_phase(wrapped - phase)) / math.pi * venc
                    history["eval_iterations"].append(iteration)
                    history["eval_loss"].append(float(objective(phase).item()))
                    if reference is not None:
                        history["nrmse"].append((iteration, _nrmse(recovered, reference, mask)))
                    if callback is not None:
                        history["kernel_snapshots"].append(dict(iteration=iteration,
                            centers=torch.tanh(model.center_raw).detach().cpu(),
                            scales=model.spatial_scales().detach().cpu(),
                            rotations=_axis_angle_to_matrix(model.rotation_raw).detach().cpu()))
                        callback(iteration, recovered, history)
        torch.cuda.synchronize(self.device)
        history["time_ms"] = [(time.perf_counter() - started) * 1e3 / self.num_iter] * self.num_iter
        history["iterations_completed"] = self.num_iter
        history["optimizer_steps_completed"] = self.num_iter - len(history["nonfinite_steps"])
        if reference is not None:
            history["returned_nrmse"] = history["nrmse"][-1][1]
        return GUSTResult(recovered, phase, model, history)
