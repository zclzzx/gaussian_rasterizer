#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from typing import NamedTuple

import torch
import torch.nn as nn

from . import _C


def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(
        item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)


def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    clip_features,  # 新增clip特征参数
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        clip_features,  # 传递clip特征
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    )

def check_tensor_validity(tensor, name):
    if tensor is None:
        print(f"{name} is None")
        return
    print(f"{name}: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}")
    if not tensor.is_cuda:
        print(f"WARNING: {name} is not on CUDA device")
    if tensor.nelement() > 0 and not tensor.isfinite().all():
        print(f"ERROR: {name} contains NaN or Inf values")
    # 检查索引范围
    if tensor.dim() > 0 and (tensor < 0).any():
        print(f"WARNING: {name} contains negative values")

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        clip_features,  # 新增clip特征参数
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg,
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            clip_features,  # 传递给C++函数
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug,
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            # Copy them before they can be corrupted
            cpu_args = cpu_deep_copy_tuple(args)
            try:
                num_rendered, color, depth, alpha, clips, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(
                    *args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, color, depth, alpha, clips, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(
                *args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations,
                              cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer, alpha, clips)
        return color, depth, alpha, radii, clips

    @staticmethod
    def backward(ctx, grad_out_color, grad_out_depth, grad_out_alpha, grad_out_radii, grad_out_clips):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer, alpha, clip_features = ctx.saved_tensors
        grad_out_clips = grad_out_clips.float()
        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D,
                radii,
                colors_precomp,
                scales,
                rotations,
                raster_settings.scale_modifier,
                cov3Ds_precomp,
                raster_settings.viewmatrix,
                raster_settings.projmatrix,
                raster_settings.tanfovx,
                raster_settings.tanfovy,
                grad_out_color,
                grad_out_depth,
                grad_out_alpha,
                sh,
                grad_out_clips,
                clip_features,
                raster_settings.sh_degree,
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                alpha,
                raster_settings.debug)
        # print type of grad_out_clips and clip_features
        # print("=========in _RasterizeGaussians backward grad_out_clips  is: " + str(grad_out_clips))
        # print("=========in _RasterizeGaussians backward clip_features  is: " + str(clip_features))
        # print("=========in _RasterizeGaussians backward grad_out_color is: " + str(grad_out_color))
        # print("=========in _RasterizeGaussians backward colors_precomp is: " + str(colors_precomp))
        # print shape of above tensors

        # check_tensor_validity(rotations, "rotations")
        # check_tensor_validity(grad_out_depth, "grad_out_depth")
        # check_tensor_validity(grad_out_alpha, "grad_out_alpha")

        # check_tensor_validity(grad_out_clips, "grad_out_clips")
        # check_tensor_validity(clip_features, "clip_features")
        # check_tensor_validity(grad_out_color, "grad_out_color")
        # check_tensor_validity(colors_precomp, "colors_precomp")
        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            # Copy them before they can be corrupted
            cpu_args = cpu_deep_copy_tuple(args)
            try:
                grad_means2D, grad_colors_precomp, grad_clip_features, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(
                    *args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print(
                    "\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
            grad_means2D, grad_colors_precomp, grad_clip_features, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(
                *args)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_clip_features,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
        )

        return grads


class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int
    tanfovx: float
    tanfovy: float
    bg: torch.Tensor
    scale_modifier: float
    viewmatrix: torch.Tensor
    projmatrix: torch.Tensor
    sh_degree: int
    campos: torch.Tensor
    prefiltered: bool
    debug: bool


class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)

        return visible

    def forward(self, means3D, means2D, opacities, shs=None, clip_features=None, colors_precomp=None, scales=None, rotations=None, cov3D_precomp=None):
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception(
                'Please provide excatly one of either SHs or precomputed colors!')

        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception(
                'Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')

        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])
        if clip_features is None:
            print("=========in GaussianRasterizer forward clip_features is None, set to empty tensor")
            clip_features = torch.Tensor([])
        # print content of clip_features to screen
        # print("=========in GaussianRasterizer forward clip_features is: " + str(clip_features))
        # Invoke C++/CUDA rasterization routine
        # print("=========in GaussianRasterizer forward raster_settings type is: " + str(type(raster_settings)))
        
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            clip_features,
            colors_precomp,
            opacities,
            scales,
            rotations,
            cov3D_precomp,
            raster_settings,
        )
