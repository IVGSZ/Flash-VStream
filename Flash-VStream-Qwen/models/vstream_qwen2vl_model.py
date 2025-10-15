from functools import partial
import math
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from torch.nn import CrossEntropyLoss, LayerNorm
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VLForConditionalGeneration,
    Qwen2VLCausalLMOutputWithPast,
    Qwen2VLModel,
    Qwen2VLPreTrainedModel,
    Qwen2VisionTransformerPretrainedModel,
    _prepare_4d_causal_attention_mask_with_cache_position,
    PatchEmbed,
    VisionRotaryEmbedding,
    Qwen2VLVisionBlock,
    PatchMerger,
)
from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLConfig, Qwen2VLVisionConfig
from transformers.cache_utils import Cache, StaticCache
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_flash_attention_utils import _flash_attention_forward

from .compress_functions import (
    drop_feature, merge_feature, kmeans_feature, 
    weighted_kmeans_feature, weighted_kmeans_ordered_feature, 
    pca_weighted_kmeans_ordered_feature, torchpca_weighted_kmeans_ordered_feature,
    fast_weighted_kmeans_ordered_feature,
    k_drop_feature, k_merge_feature,
    dbscan_feature, gmm_feature, attention_feature
)
from .flash_memory_constants import (
    DEFAULT_FLASH_MEMORY_CONFIG,
)

# calc the grid_thw after memory compression
def get_real_grid_thw(thw, flash_memory_config):
    if flash_memory_config is None:
        return thw
    t_len = flash_memory_config['flash_memory_temporal_length'] // 2
    t_pool = flash_memory_config['flash_memory_temporal_poolsize']
    t, h, w = thw
    t = min(t, t_len)  # temporal compress
    if t_pool == 2:
        h = h // 2
        w = w // 2
        if h % 2 != 0:
            h += 1
        if w % 2 != 0:
            w += 1
    elif t_pool > 2:
        raise NotImplementedError(f"Only support t_pool=2 or t_pool=1, t_pool={t_pool}")
    real_thw = torch.tensor([t, h, w], dtype=thw.dtype, device=thw.device)
    return real_thw

def get_real_grid_thws(grid_thw, flash_memory_config):
    real_grid_thw = []
    for thw in grid_thw:
        real_thw = get_real_grid_thw(thw, flash_memory_config)
        real_grid_thw.append(real_thw)
    return torch.stack(real_grid_thw, dim=0)

def get_spatial_real_grid_thw(thw, flash_memory_config):
    t, h, w = thw
    if flash_memory_config is None:
        t = 0
    s_len = flash_memory_config['flash_memory_spatial_length'] // 2
    t = min(t, s_len)  # spatial compress
    real_thw = torch.tensor([t, h, w], dtype=thw.dtype, device=thw.device)
    return real_thw


class FlashMemory(nn.Module):
    def __init__(
        self, 
        flash_memory_temporal_length=120, 
        flash_memory_temporal_method='kmeans_ordered',
        flash_memory_temporal_poolsize=2,
        flash_memory_temporal_pca_dim=32,
        flash_memory_spatial_length=60,
        flash_memory_spatial_method='klarge_retrieve',
    ):
        super().__init__()
        self.config = dict(
            flash_memory_temporal_length=flash_memory_temporal_length,
            flash_memory_temporal_method=flash_memory_temporal_method,
            flash_memory_temporal_poolsize=flash_memory_temporal_poolsize,
            flash_memory_temporal_pca_dim=flash_memory_temporal_pca_dim,
            flash_memory_spatial_length=flash_memory_spatial_length,
            flash_memory_spatial_method=flash_memory_spatial_method,
        )
        assert flash_memory_temporal_length % 2 == 0, f"In FlashMemory, temporal_length should be even, temporal_length={flash_memory_temporal_length}"
        self.temporal_length = flash_memory_temporal_length // 2
        self.temporal_method = flash_memory_temporal_method
        self.temporal_poolsize = flash_memory_temporal_poolsize
        self.temporal_pca_dim = flash_memory_temporal_pca_dim
        assert flash_memory_spatial_length % 2 == 0, f"In FlashMemory, spatial_length should be even, spatial_length={flash_memory_temporal_length}"
        self.spatial_length = flash_memory_spatial_length // 2
        self.spatial_method = flash_memory_spatial_method

        # if self.temporal_method == 'attention':
        #     self.attn = CrossAttention()

    """ This is a slow implementation. 
    TODO: when we have time, upgrade it to a faster version by integrating to ImageProcessor
    """
    def temporal_pool(self, x, thw):
        # grid_thw is [T/2, H/14, W/14]
        # x.shape is [grid_t x grid_h/2 x grid_w/2 x 2 x 2, 1280]
        t, h, w = thw
        xdim = x.shape[-1]
        assert self.temporal_poolsize == 2
        assert xdim == 3 * 2 * 14 * 14
        # print(f'Note, Perform temporal pool, x={x.shape}, thw={thw}')
        # preknowledge: input [T, 3, H, W] shape, h = H / 14, w = W / 14
        x = x.reshape(t, h // 2, w // 2, 2, 2, 3, 2, 14, 14)  # [T // 2, h // 2, w // 2, 2(for h), 2(for w), 3, 2(for t), 14, 14]
        x = x.permute(0, 1, 2, 5, 6, 3, 7, 4, 8)  # [T // 2, h // 2, w // 2, 3, 2, 2, 14, 2, 14]
        x = x.reshape(-1, 6, 28, 28)  # 
        x = F.avg_pool2d(x, kernel_size=2, stride=2)  # [-1, 6, 14, 14]
        x = x.reshape(t, h // 2, w // 2, 3, 2, 14, 14) 
        pad_h = (h // 2) % 2
        pad_w = (w // 2) % 2
        # No need to repeat padding, we ensure it in the FlashVStreamQwen2VLImageProcessor
        if pad_h > 0:
            raise NotImplementedError(f"Performing temporal pool, pad_h > 0, pad_h={pad_h}")
        if pad_w > 0:
            raise NotImplementedError(f"Performing temporal pool, pad_w > 0, pad_w={pad_w}")
        new_h = x.shape[1] // 2
        new_w = x.shape[2] // 2
        x = x.reshape(t, new_h, 2, new_w, 2, 3, 2, 14, 14)
        x = x.permute(0, 1, 3, 2, 4, 5, 6, 7, 8)
        x = x.reshape(t, new_h, new_w, 2 * 2 * xdim).reshape(-1, xdim)
        new_thw = thw.clone()
        new_thw[1] = new_h * 2
        new_thw[2] = new_w * 2
        return x, new_thw
    
    """ Calc CSM memory, from temporal clustering """
    def temporal_compress(self, x, thw, temporal_length):
        # grid_thw is [T/2, H/14, W/14]
        # x.shape is [grid_t x grid_h/2 x grid_w/2 x 2 x 2, 1280]
        t, h, w = thw
        if t <= temporal_length:
            return x, thw, torch.ones(t, device=x.device), torch.arange(t, device=x.device, dtype=torch.int32), [[i] for i in range(t)]
        assert h % 2 == 0
        assert w % 2 == 0
        x = x.reshape(t, h // 2 * w // 2 * 2 * 2, x.shape[-1])
        if temporal_length == 0:
            x = x[:0, ...]
            tem_thw = thw.clone()
            tem_thw[0] = 0
            return x.reshape(-1, x.shape[-1]), tem_thw, torch.ones(0, device=x.device), torch.arange(0, device=x.device, dtype=torch.int32), []
        # compress [t, ...] to [temporal_length, ...]
        method_dic = {
            'sample': lambda x, t_len: (x[torch.linspace(0, t - 1, t_len).long()], None, torch.linspace(0, t - 1, t_len, device=x.device).long(), None),
            'merge': merge_feature,
            'drop': drop_feature,
            'kmeans': weighted_kmeans_feature,
            'kmeans_ordered': weighted_kmeans_ordered_feature,
            'pca_kmeans_ordered': pca_weighted_kmeans_ordered_feature,
            'torchpca_kmeans_ordered': torchpca_weighted_kmeans_ordered_feature,
            'fast_kmeans_ordered': fast_weighted_kmeans_ordered_feature,
            'dbscan': dbscan_feature,
            'gmm': gmm_feature,
            'attention': partial(attention_feature, attention_fn=None),
        }
        if self.temporal_method in method_dic:
            x, weights, timestamps, indices = method_dic[self.temporal_method](x, temporal_length)
        else:
            raise ValueError(f"temporal_method should be one of {method_dic.keys()}")
        tem_thw = thw.clone()
        tem_thw[0] = x.shape[0]
        return x.reshape(-1, x.shape[-1]), tem_thw, weights, timestamps, indices

    """ Given tem_x (CSM memory), retrieve spa_x (DAM memory) from x (Feature Bank )"""
    def spatial_enhance(self, x, small_x, thw, tem_x, tem_thw, tem_weights, tem_positions, tem_indices):
        # Euclidean distance
        def efficient_euclidean_distance(A, B):
            assert A.ndim == 2
            assert B.ndim == 2
            assert A.shape[1] == B.shape[1]
            A_2 = torch.sum(A ** 2, dim=1, keepdim=True)
            B_2 = torch.sum(B ** 2, dim=1, keepdim=True)
            AB = A @ B.T
            dists_2 = A_2 + B_2.T - 2 * AB
            dists = torch.sqrt(dists_2)
            return dists
        def cosine_similarity(A, B):
            assert A.ndim == 2
            assert B.ndim == 2
            assert A.shape[1] == B.shape[1]
            A_norm = A / A.norm(dim=-1, keepdim=True)  # [k, d]
            B_norm = B / B.norm(dim=-1, keepdim=True)  # [t, d]
            cosine_similarity = torch.matmul(A_norm, B_norm.T)  # [k, t]
            return cosine_similarity
        t, h, w = thw
        xdim = x.shape[-1]
        x = x.reshape(t, h // 2 * w // 2 * 2 * 2, xdim)
        small_x = small_x.reshape(t, h // 4 * w // 4 * 2 * 2, xdim)
        st, sh, sw = tem_thw
        tem_x = tem_x.reshape(st, sh // 2 * sw // 2 * 2 * 2, xdim)
        centroids = tem_x
        method_list = ['sample', 'nearest', 'klarge_retrieve', 'klarge_retrieve_cos']
        metric_dic = {
            'klarge_retrieve': efficient_euclidean_distance,
            'klarge_retrieve_cos': cosine_similarity,
        }
        if t <= self.spatial_length:
            spa_x = x
            spa_positions = torch.arange(t, device=x.device).long()
        else:
            if self.spatial_method == 'sample':
                idx = torch.linspace(0, t - 1, self.spatial_length, device=x.device).round().long()
                spa_x = x[idx]
                spa_positions = idx
            elif self.spatial_method == 'nearest':  # retrieve by temporal distance (round to integer)
                sorted_indices = torch.argsort(tem_weights, descending=True)
                klarge_indices = sorted_indices[:self.spatial_length]
                idx = tem_positions[klarge_indices]
                spa_x = x[idx]
                spa_positions = idx
            elif self.spatial_method.startswith('klarge_retrieve'):  # retrieve by feature map Euclidean distance
                centroids = centroids.reshape(st, sh // 2 * sw // 2 * 2 * 2, xdim)
                sorted_indices = torch.argsort(tem_weights, descending=True)
                klarge_indices = sorted_indices[:self.spatial_length]
                centroids = centroids[klarge_indices].reshape(self.spatial_length, -1) # [T1, P*D]
                small_x = small_x.reshape(t, -1)
                dist_func = metric_dic[self.spatial_method]
                dist = dist_func(centroids, small_x)  # [k, t]
                idx = torch.argmin(dist, dim=1)
                spa_x = x[idx]
                spa_positions = idx
            else:
                raise ValueError(f"spatial_method should be one of {method_list}")
        spa_thw = thw.clone()
        spa_thw[0] = spa_x.shape[0]
        # print(f'Perform spatial_enhance, method = {self.spatial_method}, x.shape={x.shape}, spa_x.shape={spa_x.shape}, tem_x.shape={tem_x.shape}')
        return spa_x, spa_thw, spa_positions

    def cat_spa_tem(self, spa_x, tem_x):
        xdim = spa_x.shape[-1]
        spa_x = spa_x.reshape(-1, 2 * 2, xdim)
        tem_x = tem_x.reshape(-1, 2 * 2, xdim)
        cat_x = torch.cat([spa_x, tem_x], dim=0).reshape(-1, xdim).contiguous()
        return cat_x

    """ AM-RoPE: recalculate 3D visual position embedding based on CSM and DAM temporal positions """
    def calc_am_rope(self, position_id, visual_position_id, tem_thw, tem_positions, spa_thw, spa_positions):
        """ Position_ids update only suppor batchsize=1 now!"""
        mask = visual_position_id >= 0
        visual_token_indices = torch.nonzero(mask, as_tuple=False)
        visual_start_pos = visual_token_indices[0].item()
        visual_start_id = position_id[0, visual_start_pos]  # only suppor batchsize=1
        assert position_id[0, visual_start_pos] == position_id[1, visual_start_pos]
        assert position_id[1, visual_start_pos] == position_id[2, visual_start_pos]
        visual_end_pos = visual_token_indices[-1].item()
        def get_mm_index_with_positions(thw_grid, t_positions):
            llm_grid_t, llm_grid_h, llm_grid_w = (thw_grid[0].item(), thw_grid[1].item() // 2, thw_grid[2].item() // 2)
            assert t_positions.shape[0] == llm_grid_t, f"t_positions.shape={t_positions.shape} should be equal to llm_grid_t={llm_grid_t}"
            t_index = t_positions.view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
            h_index = torch.arange(llm_grid_h, device=t_index.device).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
            w_index = torch.arange(llm_grid_w, device=t_index.device).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
            return torch.stack([t_index, h_index, w_index]), thw_grid.prod() // 4
        spa_pos_ids, spa_size = get_mm_index_with_positions(spa_thw, spa_positions)
        tem_pos_ids, tem_size = get_mm_index_with_positions(tem_thw, tem_positions)
        tem_pos_ids = tem_pos_ids + spa_size
        cat_pos_ids = torch.cat([spa_pos_ids, tem_pos_ids], dim=1)
        flash_memory_position_ids = visual_start_id + cat_pos_ids  # [3, L]
        assert spa_size + tem_size == visual_end_pos - visual_start_pos + 1, f"sth went wrong! check: spa_size={spa_size}, tem_size={tem_size}, visual_end_pos={visual_end_pos}, visual_start_pos={visual_start_pos}"
        position_id[:, mask] = flash_memory_position_ids  # [3, videoL]
        return position_id

    def forward(self, x, grid_thw, small_grid_thw, position_ids, visual_position_ids):
        # grid_thw is [[T/2, H/14, W/14]], shape is [B, 3]
        # x.shape is [B * grid_t x grid_h/2 x grid_w/2 x 2 x 2, 1280]
        # print(f'In FlashMemory forward, x.shape={x.shape}, grid_thw={grid_thw.shape} {grid_thw}, small_grid_thw={small_grid_thw} self.spatial_length={self.spatial_length} {self.spatial_method}')
        if small_grid_thw is not None:
            seqlens = torch.cat([grid_thw, small_grid_thw], dim=0).prod(dim=1)
            all_list = torch.split(x, seqlens.tolist())
            assert len(all_list) % 2 == 0
            bsz = len(all_list) // 2
            x_list, small_x_list = all_list[:bsz], all_list[bsz:]
        else:
            seqlens = grid_thw.prod(dim=1)
            x_list = torch.split(x, seqlens.tolist())
            small_x_list = x_list
            small_grid_thw = grid_thw

        new_x_list = []
        new_position_id_list = []
        for x, thw, small_x, small_thw, position_id, visual_position_id in \
        zip(x_list, grid_thw, small_x_list, small_grid_thw, torch.unbind(position_ids, dim=1), visual_position_ids):
            tem_x, tem_thw, tem_weights, tem_timestamp, tem_indices = self.temporal_compress(small_x, small_thw, self.temporal_length)
            tem_positions = tem_timestamp.round().long()
            if self.spatial_length > 0:
                spa_x, spa_thw, spa_positions = self.spatial_enhance(
                    x=x, 
                    small_x=small_x,
                    thw=thw, 
                    tem_x=tem_x, 
                    tem_thw=tem_thw, 
                    tem_weights=tem_weights, 
                    tem_positions=tem_positions,
                    tem_indices=tem_indices,
                )
            else:
                spa_x = x[0:0]
                spa_thw = thw.clone()
                spa_thw[0] = 0
                spa_positions = torch.tensor([], device=x.device).long()
            new_x = self.cat_spa_tem(spa_x=spa_x, tem_x=tem_x)
            new_x_list.append(new_x)
            new_position_id = self.calc_am_rope(position_id, visual_position_id, tem_thw, tem_positions, spa_thw, spa_positions)
            new_position_id_list.append(new_position_id)
        x = torch.stack(new_x_list, dim=0)
        position_ids = torch.stack(new_position_id_list, dim=1)
        return x, position_ids


class FlashVStreamQwen2VisionTransformerPretrainedModel(Qwen2VLPreTrainedModel):
    config_class = Qwen2VLVisionConfig
    _no_split_modules = ["Qwen2VLVisionBlock"]

    def __init__(self, config) -> None:
        super().__init__(config)
        self.spatial_merge_size = config.spatial_merge_size

        self.patch_embed = PatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.embed_dim,
        )

        head_dim = config.embed_dim // config.num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList(
            [Qwen2VLVisionBlock(config, config._attn_implementation) for _ in range(config.depth)]
        )
        if getattr(config, 'flash_memory_config', None) is None:
            warnings.warn(f'Qwen2VLVisionConfig.flash_memory_config is not set. Set it to default, sample 10000')
            config.flash_memory_config = DEFAULT_FLASH_MEMORY_CONFIG
        self.flash_memory = FlashMemory(**config.flash_memory_config)
        self.merger = PatchMerger(dim=config.hidden_size, context_dim=config.embed_dim)

    def get_dtype(self) -> torch.dtype:
        return self.blocks[0].mlp.fc2.weight.dtype

    def get_device(self) -> torch.device:
        return self.blocks[0].mlp.fc2.weight.device

    def rot_pos_emb(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def forward(
        self, 
        hidden_states: torch.Tensor, 
        grid_thw: torch.Tensor, 
        position_ids: torch.Tensor,
        visual_position_ids: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = hidden_states.view(-1, 3 * 2 * 14 * 14)
        # small resolution pathway
        if self.flash_memory.temporal_poolsize > 1:
            small_hidden_states, small_grid_thw = [], []
            bsz = grid_thw.shape[0]
            st = 0
            for i in range(bsz):
                ed = st + grid_thw[i].prod()
                new_x, new_thw = self.flash_memory.temporal_pool(hidden_states[st:ed], grid_thw[i])
                small_hidden_states.append(new_x)
                small_grid_thw.append(new_thw)
                st = ed
            small_hidden_states = torch.cat(small_hidden_states, dim=0)
            small_grid_thw = torch.stack(small_grid_thw, dim=0)
            hidden_states = torch.cat([hidden_states, small_hidden_states], dim=0)
            total_grid_thw = torch.cat([grid_thw, small_grid_thw], dim=0)
            # print(f'In Vision forward, hidden_states={hidden_states.shape}, total_grid_thw={total_grid_thw.shape} {total_grid_thw}')
        else:
            small_hidden_states, small_grid_thw = None, None
            total_grid_thw = grid_thw
            
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(total_grid_thw)

        cu_seqlens = torch.repeat_interleave(total_grid_thw[:, 1] * total_grid_thw[:, 2], total_grid_thw[:, 0]).cumsum(
            dim=0, dtype=torch.int32
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for blk in self.blocks:
            hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb)
        hidden_states, position_ids = self.flash_memory(hidden_states, grid_thw, small_grid_thw, position_ids, visual_position_ids)
        hidden_states = self.merger(hidden_states)
        return hidden_states, position_ids
    

class FlashVStreamQwen2VLConfig(Qwen2VLConfig):
    model_type = "flash_vstream_qwen2_vl"
    def __init__(
        self,
        vision_config=None,
        **kwargs,
    ):
        super().__init__(vision_config=vision_config, **kwargs)
        if isinstance(vision_config, dict) and isinstance(vision_config.get('flash_memory_config', None), dict):
            self.vision_config.flash_memory_config = vision_config['flash_memory_config']
        else:
            warnings.warn(f'note that vision_config.flash_memory_config is not set. Please set it using set_flash_memory_config')

    def set_flash_memory_config(
        self, 
        flash_memory_temporal_length, 
        flash_memory_temporal_method,
        flash_memory_temporal_poolsize,
        flash_memory_temporal_pca_dim,
        flash_memory_spatial_length,
        flash_memory_spatial_method
    ):
        self.vision_config.flash_memory_config = dict(
            flash_memory_temporal_length=flash_memory_temporal_length,
            flash_memory_temporal_method=flash_memory_temporal_method,
            flash_memory_temporal_poolsize=flash_memory_temporal_poolsize,
            flash_memory_temporal_pca_dim=flash_memory_temporal_pca_dim,
            flash_memory_spatial_length=flash_memory_spatial_length,
            flash_memory_spatial_method=flash_memory_spatial_method,
        )
        print(f'Set flash_memory_config to {self.vision_config.flash_memory_config}')


class FlashVStreamQwen2VLModel(Qwen2VLForConditionalGeneration):
    config_class = FlashVStreamQwen2VLConfig

    # override the __init__ method of Qwen2VLForConditionalGeneration
    def __init__(self, config): 
        Qwen2VLPreTrainedModel.__init__(self, config)

        self.visual = FlashVStreamQwen2VisionTransformerPretrainedModel._from_config(
            config.vision_config, attn_implementation=config._attn_implementation
        )
        self.model = Qwen2VLModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.padding_side = "left"  # set it to left by default, user can use setter to change padding_sides

        # Initialize weights and apply final processing
        self.post_init()

        print(f'FlashVStreamQwen2VLModel is initialized')
        print(f'FlashVStreamQwen2VLModel config:\n{config}')
        if getattr(config.vision_config, 'flash_memory_config', None):
            print(f'FlashMemory config:\n{config.vision_config.flash_memory_config}')
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        video_embeds: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        visual_position_ids=None,
    ) -> Union[Tuple, Qwen2VLCausalLMOutputWithPast]:
        
        if self.training:
            assert not use_cache, "`use_cache` should not be set during training, not support cache + dist_attn yet. Very dangerous!"

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            # load from image
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.get_dtype())
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw).to(inputs_embeds.device)
                image_mask = input_ids == self.config.image_token_id
                if self.training:
                    inputs_embeds = inputs_embeds.clone()
                inputs_embeds[image_mask] = image_embeds
            # load from video
            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
                video_embeds, position_ids = self.visual(pixel_values_videos, grid_thw=video_grid_thw, position_ids=position_ids, visual_position_ids=visual_position_ids)
                video_embeds = video_embeds.to(inputs_embeds.device)  # [N, dim]
                video_mask = input_ids == self.config.video_token_id
                if self.training:
                    inputs_embeds = inputs_embeds.clone()
                    visual_position_local = visual_position_ids.to(inputs_embeds.device)[video_mask]
                    assert torch.all(visual_position_local >= 0), f"visual_position_ids should be >= 0, visual_position_ids={visual_position_ids}"
                    video_embeds = video_embeds[visual_position_local]
                inputs_embeds[video_mask] = video_embeds
                # print(f'In forward, video_embeds={video_embeds.shape}, inputs_embeds={inputs_embeds.shape}')
            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)
                
        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]

        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Qwen2VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=rope_deltas,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        video_embeds=None,
        image_grid_thw=None,
        video_grid_thw=None,
        visual_position_ids=None,
        **kwargs,
    ):
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        rope_deltas = kwargs.get("rope_deltas", None)
        if attention_mask is not None and position_ids is None:
            if cache_position is None or (cache_position is not None and cache_position[0] == 0):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids, image_grid_thw, video_grid_thw, attention_mask
                )
            else:
                batch_size, seq_length = input_ids.shape
                delta = (
                    cache_position[0] + rope_deltas if cache_position is not None and rope_deltas is not None else 0
                )
                position_ids = torch.arange(seq_length, device=input_ids.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        if cache_position is not None and cache_position[0] != 0:
            pixel_values = None
            pixel_values_videos = None
            video_embeds = None

        if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
            if inputs_embeds is not None:
                batch_size, sequence_length = inputs_embeds.shape
                device = inputs_embeds.device
            else:
                batch_size, sequence_length = input_ids.shape
                device = input_ids.device

            dtype = self.lm_head.weight.dtype
            min_dtype = torch.finfo(dtype).min

            attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=sequence_length,
                target_length=past_key_values.get_max_length(),
                dtype=dtype,
                device=device,
                min_dtype=min_dtype,
                cache_position=cache_position,
                batch_size=batch_size,
            )
        elif attention_mask.ndim == 2 and self.config._attn_implementation == 'eager':
            batch_size, sequence_length = input_ids.shape
            device = input_ids.device
            dtype = self.lm_head.weight.dtype
            min_dtype = torch.finfo(dtype).min
            if cache_position is None:
                cache_position = torch.arange(
                    0, input_ids.shape[1], device=device
                )
            attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=sequence_length,
                target_length=attention_mask.shape[-1],
                dtype=dtype,
                device=device,
                min_dtype=min_dtype,
                cache_position=cache_position,
                batch_size=batch_size,
            )

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "pixel_values_videos": pixel_values_videos,
                "video_embeds": video_embeds,
                "image_grid_thw": image_grid_thw,
                "video_grid_thw": video_grid_thw,
                "rope_deltas": rope_deltas,
                "visual_position_ids": visual_position_ids,
            }
        )
        return model_inputs

    def prepare_inputs_for_training(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        video_embeds=None,
        image_grid_thw=None,
        video_grid_thw=None,
        labels=None,
        visual_position_ids=None,
        **kwargs,
    ):
        assert labels is not None, "Labels are required for training!"
        use_cache = False

        rope_deltas = kwargs.get("rope_deltas", None)
        if attention_mask is not None and position_ids is None:
            if cache_position is None or (cache_position is not None and cache_position[0] == 0):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids, image_grid_thw, video_grid_thw, attention_mask
                )
            else:
                batch_size, seq_length = input_ids.shape
                delta = (
                    cache_position[0] + rope_deltas if cache_position is not None and rope_deltas is not None else 0
                )
                position_ids = torch.arange(seq_length, device=input_ids.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        if cache_position and cache_position[0] != 0:
            pixel_values = None
            pixel_values_videos = None
            video_embeds = None

        if attention_mask.ndim == 2 and self.config._attn_implementation == 'eager':
            batch_size, sequence_length = input_ids.shape
            device = input_ids.device
            dtype = self.lm_head.weight.dtype
            min_dtype = torch.finfo(dtype).min
            if cache_position is None:
                cache_position = torch.arange(
                    0, input_ids.shape[1], device=device
                )
            attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=sequence_length,
                target_length=attention_mask.shape[-1],
                dtype=dtype,
                device=device,
                min_dtype=min_dtype,
                cache_position=cache_position,
                batch_size=batch_size,
            )

        model_inputs = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "pixel_values_videos": pixel_values_videos,
            "video_embeds": video_embeds,
            "image_grid_thw": image_grid_thw,
            "video_grid_thw": video_grid_thw,
            "rope_deltas": rope_deltas,
            "labels": labels,
            "visual_position_ids": visual_position_ids,
        }
        model_inputs = {k: v for k, v in model_inputs.items() if v is not None}

        return model_inputs


    def get_rope_index(
        self,
        input_ids: torch.LongTensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

        Explanation:
            Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

            For pure text embedding sequence, the rotary position embedding has no difference with mordern LLMs.
            Examples:
                input_ids: [T T T T T], here T is for text.
                temporal position_ids: [0, 1, 2, 3, 4]
                height position_ids: [0, 1, 2, 3, 4]
                width position_ids: [0, 1, 2, 3, 4]

            For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
            and 1D rotary position embeddin for text part.
            Examples:
                Assume we have a video input with 3 temporal patches, 2 height patches and 2 width patches.
                input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
                vision temporal position_ids: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
                vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                text temporal position_ids: [3, 4, 5, 6, 7]
                text height position_ids: [3, 4, 5, 6, 7]
                text width position_ids: [3, 4, 5, 6, 7]
                Here we calculate the text start position_ids as the max vision position_ids plus 1.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

        Returns:
            position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
        """
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []
        if image_grid_thw is not None or video_grid_thw is not None:
            total_input_ids = input_ids
            position_ids = torch.ones(
                3, input_ids.shape[0], input_ids.shape[1], dtype=input_ids.dtype, device=input_ids.device
            )
            image_index, video_index = 0, 0
            # print(f'In get_rope_index, video_grid_thw={video_grid_thw}, input_ids={input_ids.shape}')
            for i, input_ids in enumerate(total_input_ids):
                if attention_mask is not None:
                    input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums = 0, 0
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                # print(f'In get_rope_index, i={i}, input_ids={input_ids.shape}, image_nums={image_nums}, video_nums={video_nums}')
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:  # TODO: support image
                        raise NotImplementedError
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        image_index += 1
                        remain_images -= 1  # haoji: this code is not safe, it changes tensor image_nums
                        ed = ed_image
                    else:
                        video_grid = video_grid_thw[video_index].cpu()
                        video_index += 1
                        remain_videos -= 1  # haoji: this too
                        ed = ed_video
                    
                    text_len = ed - st
                    # print(f'[image_nums={image_nums}][video_nums={video_nums}], find one video, st={st}, ed={ed_video}')
                    # print(f'input_ids[{ed_video-5}:]={input_tokens[ed_video-5:ed_video+5]}')

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    temporal_real_grid = get_real_grid_thw(video_grid, self.config.vision_config.flash_memory_config)
                    spatial_real_grid = get_spatial_real_grid_thw(video_grid, self.config.vision_config.flash_memory_config)

                    def get_mm_index(thw_grid):
                        llm_grid_t, llm_grid_h, llm_grid_w = (thw_grid[0].item(), thw_grid[1].item() // 2, thw_grid[2].item() // 2)
                        t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                        h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                        w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                        return torch.stack([t_index, h_index, w_index]), thw_grid.prod() // 4
                    flash_memory_spa_ids, spa_size = get_mm_index(spatial_real_grid)
                    flash_memory_tem_ids, tem_size = get_mm_index(temporal_real_grid)
                    llm_pos_ids_list.append(flash_memory_spa_ids + text_len + st_idx)
                    llm_pos_ids_list.append(flash_memory_tem_ids + text_len + st_idx + spa_size)
                    st = ed + spa_size + tem_size
                    # print(f'spatial_real_grid={spatial_real_grid}, temporal_real_grid={temporal_real_grid}')
                    # print(f'st={st}, input_ids[{st-5}:]={input_tokens[st-5:st+5]}')

                if st < len(input_tokens):
                    if len(llm_pos_ids_list) > 0:
                        if llm_pos_ids_list[-1].numel() > 0:
                            st_idx = llm_pos_ids_list[-1].max() + 1
                        else:
                            st_idx = llm_pos_ids_list[-2].max() + 1
                    else:
                        st_idx = 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                # print(f'position_ids={position_ids.shape}, llm_positions={llm_positions.shape}')
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(input_ids.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas

AutoConfig.register("flash_vstream_qwen2_vl", FlashVStreamQwen2VLConfig)
AutoModelForCausalLM.register(FlashVStreamQwen2VLConfig, FlashVStreamQwen2VLModel)
transformers.FlashVStreamQwen2VLModel = FlashVStreamQwen2VLModel
