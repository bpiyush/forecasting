"""DINOv2 backbone."""
import os

import torch
import numpy as np
import einops

# import shared.utils as su

def num_params(model, round=3):
    n_params = sum([p.numel() for p in model.parameters()])
    if round is not None:
        value = np.round(n_params / 1e6, round)
        unit = "M"
    else:
        value = n_params
        unit = ""
    print(f"::: Number of total parameters in {model.__class__.__name__}: {value}{unit}")



def get_terminal_width():
    import shutil
    return shutil.get_terminal_size().columns


def print_update(update, fillchar=".", color="yellow", pos="left", **kwargs):
    from termcolor import colored
    # add ::: to the beginning and end of the update s.t. the total length of the
    # update spans the whole terminal
    try:
        terminal_width = get_terminal_width()
    except:
        terminal_width = 98
    if pos == "center":
        update = update.center(len(update) + 2, " ")
        update = update.center(terminal_width, fillchar)
    elif pos == "left":
        update = update.ljust(terminal_width, fillchar)
        update = update.ljust(len(update) + 2, " ")
    elif pos == "right":
        update = update.rjust(terminal_width, fillchar)
        update = update.rjust(len(update) + 2, " ")
    else:
        raise ValueError("pos must be one of 'center', 'left', 'right'")
    print(colored(update, color, **kwargs))


def dinov2_with_registers(model_id='vit_base_patch14_reg4_dinov2.lvd142m'):
    import timm
    image_backbone = timm.create_model(
        model_id,
        pretrained=True,
        num_classes=0,  # remove classifier nn.Linear
        dynamic_img_size=True,
        dynamic_img_pad=True,
    )
    return image_backbone


def simdino(model_id="vit_base_patch16_224.dino"):
    assert model_id == "vit_base_patch16_224.dino", \
        "Only this model is supported."
    import timm
    ckpt_path = "/work/piyush/pretrained_checkpoints/SimDINO/vitb16_reg4_SimDNIOv2_ep100.pth"
    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
    sd = torch.load(ckpt_path, map_location="cpu")['teacher']
    key_map = {
        "backbone.": "",
        "blocks.0.": "blocks.",
        "blocks.1.": "blocks.",
        "blocks.2.": "blocks.",
        "blocks.3.": "blocks.",
    }
    for k_old, k_new in key_map.items():
        for k in list(sd.keys()):
            if k.startswith(k_old):
                sd[k.replace(k_old, k_new)] = sd.pop(k)
    import ipdb; ipdb.set_trace()

    model = timm.create_model(
        model_id,
        pretrained=True,
        num_classes=0,
        dynamic_img_size=True,
        dynamic_img_pad=True,
    )
    raise NotImplementedError("Not implemented yet.")


class DINOv2Dense(torch.nn.Module):
    def __init__(self, model_id='dense_vit_base_patch14_reg4_dinov2.lvd142m'):
        """
        DINOv2 backbone with dense features.
        
        NOTE:
        - I have added modified code in `timm` in `py39-pt201`/`qwen`   
          environment in `vision_transformers_dense.py`. Thus,
          use `dense_` predix in defining the model.
        - If using the environment, `py39-pt201-cu122`, then,
          use no need to use the prefix `dense_`.
        - Don't forget to add `from .vision_transformer_dense import *` in `__init__.py`.
          
        Options for model_id:
        - `dense_vit_base_patch14_reg4_dinov2.lvd142m`
        - `vit_base_patch14_reg4_dinov2.lvd142m`
        - `vit_small_patch8_224.dino`
        See more: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
        """
        super().__init__()
        print_update(" [:::] Loading DINOv2 Backbone", pos="left")
        import timm
        self.image_backbone = timm.create_model(
            model_id,
            pretrained=True,
            num_classes=0,  # remove classifier nn.Linear
            dynamic_img_size=True,
            dynamic_img_pad=True,
        )
        self.num_patches = self.image_backbone.patch_embed.num_patches
        self.patch_size = self.image_backbone.patch_embed.patch_size
        print("::: Done")
        num_params(self.image_backbone)
    
    def forward_cls(self, x):
        """Computes only the CLS token features."""
        return self.image_backbone(x)

    def forward(self, x):
        """Takes in a batch of images and outputs dense features."""
        
        H, W = x.shape[-2:]
        # Since dynamic_img_size=True, H and W are not fixed
        # Pad H s.t. it is a multiple of patch size ph
        ph = self.patch_size[0]
        H = H + (ph - H % ph) % ph
        # Pad W s.t. it is a multiple of patch size pw
        pw = self.patch_size[1]
        W = W + (pw - W % pw) % pw
        nh = H // ph
        nw = W // pw

        feats = self.image_backbone.get_intermediate_layers(x)[0]
        feats = einops.rearrange(feats, 'b (h w) d -> b h w d', h=nh, w=nw)
        return feats
    
    def pixel_location_to_patch_index(self, pixel_ih, pixel_iw, H, W):
        """
        Given pixel indices (i, j), returns the corresponding patch index.
        
        Args:
            pixel_ih (int): Pixel index i.
            pixel_iw (int): Pixel index j.
            H (int): Height of the image.
            W (int): Width of the image.
        """

        # Since dynamic_img_size=True, H and W are not fixed
        # Pad H s.t. it is a multiple of patch size ph
        ph = self.patch_size[0]
        H = H + (ph - H % ph) % ph
        # Pad W s.t. it is a multiple of patch size pw
        pw = self.patch_size[1]
        W = W + (pw - W % pw) % pw
        nh = H // ph
        nw = W // pw

        # Now, compute the patch index
        ih = pixel_ih // ph
        iw = pixel_iw // pw
        patch_index = ih * nw + iw

        return patch_index
    
    def patch_index_to_token_index(self, patch_index, H, W):
        """
        Given a patch index i, returns the corresponding token index (i', j')
        
        Args:
            patch_index (int): Patch index.
            H (int): Height of the image.
            W (int): Width of the image.
        """

        # Since dynamic_img_size=True, H and W are not fixed
        # Pad H s.t. it is a multiple of patch size ph
        ph = self.patch_size[0]
        H = H + (ph - H % ph) % ph
        # Pad W s.t. it is a multiple of patch size pw
        pw = self.patch_size[1]
        W = W + (pw - W % pw) % pw
        nh = H // ph
        nw = W // pw

        # Now, compute the token index
        ih = patch_index // nw
        iw = patch_index % nw
        
        # Now, get the pixel indices of these
        pixel_ih = ih * ph
        pixel_iw = iw * pw

        return ih, iw, pixel_ih, pixel_iw

    def forward_self_attention(self, x):
        H, W = x.shape[-2:]
        # Since dynamic_img_size=True, H and W are not fixed
        # Pad H s.t. it is a multiple of patch size ph
        ph = self.patch_size[0]
        H = H + (ph - H % ph) % ph
        # Pad W s.t. it is a multiple of patch size pw
        pw = self.patch_size[1]
        W = W + (pw - W % pw) % pw
        nh = H // ph
        nw = W // pw

        attn = self.image_backbone.get_last_selfattention(x)
        # num_reg_tokens = self.image_backbone.num_reg_tokens
        num_prefix_tokens = self.image_backbone.num_prefix_tokens
        # attn = attn[:, :, num_reg_tokens:, num_reg_tokens:]
        attn_cls_to_all = attn[:, :, 0, num_prefix_tokens:]

        # reshapre
        attn_cls_to_all = einops.rearrange(attn_cls_to_all, 'b n (h w) -> b n h w', h=nh, w=nw)
        return attn_cls_to_all


class DINOv2ForVideo(torch.nn.Module):
    def __init__(self, model_id):
        super().__init__()
        self.model_id = model_id
        self.model = dinov2_with_registers(model_id=model_id)
        self.embed_dim = self.model.embed_dim
        self.num_heads = 12
    
    def forward(self, videos):
        """
        Args:
            videos (torch.Tensor): Shape (B, C, T, H, W).
        """
        b = videos.shape[0]
        images = einops.rearrange(videos, 'b c t h w -> (b t) c h w')
        z = self.model.forward(images)
        z = einops.rearrange(z, '(b t) d -> b t d', b=b)
        return z
    
    def forward_dense(self, videos):
        """Outputs dense DINO features."""
        b = videos.shape[0]
        images = einops.rearrange(videos, 'b c t h w -> (b t) c h w')
        assert self.model_id.startswith("dense_"), \
            "Use the dense model for dense features."
        # [n c h w] -> [n d h' w']
        z = self.model.get_intermediate_layers(images, reshape=True)[0]
        z = einops.rearrange(z, '(b t) ... -> b t ...', b=b)
        return z
    
    def forward_cls_and_registers(self, videos, pool_registers="concat"):
        """Outputs CLS token and register features."""
        b = videos.shape[0]
        images = einops.rearrange(videos, 'b c t h w -> (b t) c h w')
        assert "reg" in self.model_id, "Model should have registers."
        z = self.model.get_intermediate_layers(images, return_prefix_tokens=True)[0][1]
        z = einops.rearrange(z, '(b t) ... -> b t ...', b=b)
        if pool_registers == "concat":
            # Concatenate CLS and register tokens
            z = einops.rearrange(z, 'b t p d -> b t (p d)')
        elif pool_registers == "mean":
            # Average the register tokens
            z = z.mean(dim=2)
        else:
            raise ValueError(f"Unknown pool_registers: {pool_registers}")
        return z


class DINOv2ForVideoWithTemporalDifference(torch.nn.Module):
    def __init__(self, model_id, drop_half=True):
        super().__init__()
        self.model = dinov2_with_registers(model_id=model_id)
        self.drop_half = drop_half
        if not drop_half:
            self.embed_dim = self.model.embed_dim * 2
        else:
            self.embed_dim = self.model.embed_dim
        self.num_heads = 12

    def forward(self, videos):
        """
        Args:
            videos (torch.Tensor): Shape (B, C, T, H, W).
        """
        b = videos.shape[0]
        images = einops.rearrange(videos, 'b c t h w -> (b t) c h w')
        z = self.model.forward(images)
        z = einops.rearrange(z, '(b t) d -> b t d', b=b)

        # Concatenate the temporal difference features
        z_diff = torch.cat([z[:, 1:] - z[:, :-1], z[:, [-1]]], dim=1)
        z = torch.cat([z, z_diff], dim=-1)

        # Drop half of the features
        if self.drop_half:
            z = z[:, :, ::2]
        
        return z


# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from typing import Sequence

import torch
from torchvision import transforms


class GaussianBlur(transforms.RandomApply):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, *, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0):
        # NOTE: torchvision is applying 1 - probability to return the original image
        keep_p = 1 - p
        transform = transforms.GaussianBlur(kernel_size=9, sigma=(radius_min, radius_max))
        super().__init__(transforms=[transform], p=keep_p)


class MaybeToTensor(transforms.ToTensor):
    """
    Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor, or keep as is if already a tensor.
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image, numpy.ndarray or torch.tensor): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, torch.Tensor):
            return pic
        return super().__call__(pic)


# Use timm's names
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def make_normalize_transform(
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
):
    return transforms.Normalize(mean=mean, std=std)


# This roughly matches torchvision's preset for classification training:
#   https://github.com/pytorch/vision/blob/main/references/classification/presets.py#L6-L44
def make_classification_train_transform(
    *,
    crop_size: int = 224,
    interpolation=transforms.InterpolationMode.BICUBIC,
    hflip_prob: float = 0.5,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
):
    transforms_list = [transforms.RandomResizedCrop(crop_size, interpolation=interpolation)]
    if hflip_prob > 0.0:
        transforms_list.append(transforms.RandomHorizontalFlip(hflip_prob))
    transforms_list.extend(
        [
            MaybeToTensor(),
            make_normalize_transform(mean=mean, std=std),
        ]
    )
    return transforms.Compose(transforms_list)


# This matches (roughly) torchvision's preset for classification evaluation:
#   https://github.com/pytorch/vision/blob/main/references/classification/presets.py#L47-L69
def make_classification_eval_transform(
    *,
    resize_size: int = 256,
    interpolation=transforms.InterpolationMode.BICUBIC,
    crop_size: int = 224,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
):
    transforms_list = [
        transforms.Resize(resize_size, interpolation=interpolation),
        transforms.CenterCrop(crop_size),
        MaybeToTensor(),
        make_normalize_transform(mean=mean, std=std),
    ]
    return transforms.Compose(transforms_list)
