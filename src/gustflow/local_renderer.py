from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
from .cuda_backend import _cupy_view, _require_cuda_float32

@lru_cache(None)
def kernels():
    import cupy as cp
    names = ('tile_candidates', 'tile_basis_forward', 'tile_basis_backward', 'tile_weight_backward', 'gather_tile_weights', 'tile_layout')
    module = cp.RawModule(code=Path(__file__).with_name('local_kernels.cu').read_text(), options=('--std=c++14', '--use_fast_math'), name_expressions=names)
    return (cp, *(module.get_function(n) for n in names))

def launch(kernel, blocks, threads, tensors, integers):
    cp = kernels()[0]
    device = tensors[0].device
    with cp.cuda.Device(device.index), cp.cuda.ExternalStream(torch.cuda.current_stream(device).cuda_stream):
        kernel((blocks,), (threads,), tuple((_cupy_view(cp, t) for t in tensors)) + tuple((np.int32(v) for v in integers)))

class LocalBasis(torch.autograd.Function):

    @staticmethod
    def forward(ctx, xyz, valid, ids, counts, inverse, centers, scales, rotation, width):
        _require_cuda_float32(xyz, centers, scales, rotation)
        tiles, voxels = valid.shape
        n = centers.shape[0]
        basis = centers.new_empty((tiles, width, voxels))
        launch(kernels()[2], min((basis.numel() + 255) // 256, 65535), 256, (xyz, valid, ids, counts, centers, scales, rotation, basis), (tiles, width, ids.shape[1], voxels))
        ctx.save_for_backward(xyz, valid, inverse, counts, centers, scales, rotation, basis)
        return basis

    @staticmethod
    def backward(ctx, upstream):
        xyz, valid, inverse, counts, centers, scales, rotation, basis = ctx.saved_tensors
        gc, gs, gr = (torch.zeros_like(t) for t in (centers, scales, rotation))
        tiles, width, voxels = basis.shape
        launch(kernels()[3], centers.shape[0], 256, (xyz, valid, inverse, centers, scales, rotation, basis, upstream.contiguous(), gc, gs, gr), (width, centers.shape[0], voxels, tiles))
        return (None, None, None, None, None, gc, gs, gr, None)

class LocalWeights(torch.autograd.Function):

    @staticmethod
    def forward(ctx, weights, gather_ids, inverse, tiles, width):
        n = weights.shape[-1]
        ctx.save_for_backward(inverse)
        ctx.shape, ctx.tiles, ctx.width = (weights.shape, tiles, width)
        output = weights.new_empty(tiles, weights.shape[0] * weights.shape[1], width)
        launch(kernels()[5], min((output.numel() + 255) // 256, 65535), 256, (weights.contiguous(), gather_ids, output), (tiles, output.shape[1], width, n, gather_ids.shape[1]))
        return output

    @staticmethod
    def backward(ctx, upstream):
        inverse, = ctx.saved_tensors
        channels, frames, n = ctx.shape
        gradient = upstream.new_empty(channels, frames, n)
        launch(kernels()[4], n, 256, (upstream.transpose(1, 2).contiguous(), inverse, gradient), (ctx.tiles, ctx.width, channels * frames, n))
        return (gradient, None, None, None, None)

class TileLayout(torch.autograd.Function):

    @staticmethod
    def forward(ctx, voxel_ids, order, voxels, *groups):
        dimensions, size = groups[0].shape[1:]
        output = groups[0].new_empty(dimensions, voxels)
        ctx.save_for_backward(voxel_ids, order)
        ctx.shapes = [g.shape for g in groups]
        ctx.voxels = voxels
        offset = 0
        for group in groups:
            launch(kernels()[6], min((group.numel() + 255) // 256, 65535), 256, (group.contiguous(), voxel_ids, order, output), (group.shape[0], dimensions, size, voxels, offset, 0))
            offset += group.shape[0]
        return output

    @staticmethod
    def backward(ctx, upstream):
        voxel_ids, order = ctx.saved_tensors
        upstream = upstream.contiguous()
        gradients = []
        offset = 0
        for shape in ctx.shapes:
            gradient = upstream.new_empty(shape)
            launch(kernels()[6], min((gradient.numel() + 255) // 256, 65535), 256, (upstream, voxel_ids, order, gradient), (*shape, ctx.voxels, offset, 1))
            gradients.append(gradient)
            offset += shape[0]
        return (None, None, None, *gradients)

class LocalGaussianRenderer(torch.nn.Module):

    def __init__(self, shape, coords):
        super().__init__()
        self.shape = tuple(shape)
        self.tile_shape = (8, 8, 4)
        self.grid = tuple(((n + b - 1) // b for n, b in zip(shape, self.tile_shape)))
        axes = [torch.arange(n, device=coords.device) for n in self.grid]
        tiles = torch.stack(torch.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)
        axes = [torch.arange(n, device=coords.device) for n in self.tile_shape]
        local = torch.stack(torch.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)
        indices = tiles[:, None, :] * torch.tensor(self.tile_shape, device=coords.device) + local[None]
        valid = (indices < torch.tensor(shape, device=coords.device)).all(-1)
        clamped = torch.minimum(indices, torch.tensor(shape, device=coords.device) - 1)
        flat = (clamped[..., 0] * shape[1] + clamped[..., 1]) * shape[2] + clamped[..., 2]
        xyz = coords[flat].contiguous()
        self.register_buffer('xyz', xyz, persistent=False)
        self.register_buffer('valid', valid.int().contiguous(), persistent=False)
        self.register_buffer('lo', xyz.amin(1).contiguous(), persistent=False)
        self.register_buffer('hi', xyz.amax(1).contiguous(), persistent=False)
        self.register_buffer('voxel_ids', torch.where(valid, flat, -1).int().contiguous(), persistent=False)
        self.last_width = 0

    def forward(self, centers, scales, rotation, weights):
        centers, scales, rotation = (x.contiguous() for x in (centers, scales, rotation))
        n, tiles = (centers.shape[0], self.valid.shape[0])
        with torch.no_grad():
            extents = 4.0 * (rotation.square() * scales.square()[:, None, :]).sum(-1).sqrt()
            ids = torch.zeros((tiles, n), device=centers.device, dtype=torch.int32)
            counts = torch.empty(tiles, device=centers.device, dtype=torch.int32)
            inverse = torch.full((n, tiles), -1, device=centers.device, dtype=torch.int32)
            launch(kernels()[1], tiles, 256, (self.lo, self.hi, centers, extents, ids, counts, inverse), (n, tiles))
            order = torch.argsort(counts, stable=True)
            boundaries = [i * tiles // min(4, tiles) for i in range(min(4, tiles) + 1)]
            endpoints = torch.tensor([b - 1 for b in boundaries[1:]], device=centers.device)
            widths = [max(1, int(x)) for x in counts[order[endpoints]].cpu().tolist()]
            widths = [min(n, (w + 63) // 64 * 64) for w in widths]
            self.last_width = max(widths)
        outputs = []
        for group, width in enumerate(widths):
            selected = order[boundaries[group]:boundaries[group + 1]]
            xyz, valid = (self.xyz[selected], self.valid[selected])
            group_ids, group_counts, reverse = (ids[selected], counts[selected], inverse[:, selected].contiguous())
            group_tiles = valid.shape[0]
            basis = LocalBasis.apply(xyz, valid, group_ids, group_counts, reverse, centers, scales, rotation, width)
            gathered = LocalWeights.apply(weights, group_ids, reverse, group_tiles, width)
            outputs.append(torch.bmm(gathered, basis))
        return TileLayout.apply(self.voxel_ids, order, self.shape[0] * self.shape[1] * self.shape[2], *outputs).reshape(*weights.shape[:2], *self.shape)
