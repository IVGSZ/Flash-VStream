#    This file may have been modified by Flash-VStream Authors (Flash-VStream Modifications”). All Flash-VStream Modifications are Copyright 2024 Flash-VStream Authors. 
# ------------------------------------------------------------------------
# Based on https://github.com/haotian-liu/LLaVA. Below is the original copyright:
#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import time
import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.multiprocessing import Lock, Manager

from abc import ABC, abstractmethod
from flash_vstream.model.multimodal_encoder.builder import build_vision_tower
from flash_vstream.model.multimodal_projector.builder import build_vision_projector
from flash_vstream.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from flash_vstream.model.compress_functions import drop_feature, merge_feature, kmeans_feature, weighted_kmeans_feature, k_drop_feature, k_merge_feature, attention_feature


class NeuralTuringMachine(nn.Module):
    def __init__(self, input_dim=1024, output_dim=1024, attention_dropout=0.1):
        super(NeuralTuringMachine, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.q_proj = nn.Linear(input_dim, output_dim)
        self.k_proj = nn.Linear(input_dim, output_dim)
        self.v_proj = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(attention_dropout)
        self.out_proj = nn.Linear(output_dim, input_dim)
        self.out_dropout = nn.Dropout(attention_dropout)
        self.out_ln = nn.LayerNorm(input_dim, eps=1e-12)

    def get_weight(self, x, y):
        query = self.q_proj(x)
        key = self.k_proj(y)
        scores = torch.matmul(query, key.transpose(0, 1)) / math.sqrt(self.output_dim)
        weight = F.softmax(scores, dim=-1)
        return weight
    
    def forward(self, x, y):
        query = self.q_proj(x)
        key = self.k_proj(y)
        scores = torch.matmul(query, key.transpose(0, 1)) / math.sqrt(self.output_dim)
        weight = F.softmax(scores, dim=-1)
        attn = self.dropout(weight)
        value = self.v_proj(y)
        output = torch.matmul(attn, value)
        output = self.out_proj(output)
        output = self.out_dropout(output)
        output = self.out_ln(output.unsqueeze(0)).squeeze(0)
        return output


class VStreamMetaModel:

    def __init__(self, config):
        super(VStreamMetaModel, self).__init__(config)

        self.mm_input_dim = config.mm_hidden_size
        if getattr(config, 'mm_use_4_vision_tokens', False):
            self.mm_input_dim = self.mm_input_dim * 4

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config, self.mm_input_dim)

        compress_Turing_hidden_dim = getattr(self.config, "compress_Turing_hidden_dim", 32)
        self.attention_model = NeuralTuringMachine(self.mm_input_dim, compress_Turing_hidden_dim)

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        self.config.compress_type = getattr(model_args, "compress_type", None)
        self.config.compress_size = getattr(model_args, "compress_size", 1)
        self.config.compress_long_memory_size = getattr(model_args, "compress_long_memory_size", 1)
        self.config.compress_Turing_memory_size = getattr(model_args, "compress_Turing_memory_size", 1)
        self.config.compress_Turing_update_ratio = getattr(model_args, "compress_Turing_update_ratio", 0.2)
        self.config.video_max_frames = getattr(model_args, "video_max_frames", 50)
        self.config.video_long_memory_length = getattr(model_args, "video_long_memory_length", 10)
        self.config.video_Turing_memory_length = getattr(model_args, "video_Turing_memory_length", 10)
        self.config.video_short_memory_length = getattr(model_args, "video_short_memory_length", 10)
        self.config.video_current_memory_length = getattr(model_args, "video_current_memory_length", 1)
        self.config.video_sample_type = getattr(model_args, "video_sample_type", "center")

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))

class VStreamMetaForCausalLM(ABC):

    def __init__(self, config):
        super(VStreamMetaForCausalLM, self).__init__(config)
        # support video streaming mode
        self.use_video_streaming_mode = False
        self.video_embedding_memory = None  # set to torch.multiprocessing.Manager.list() when launching
        self.video_embedding_mem_lock = Lock() 

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        return image_features

    def reshape_2x2_image_features(self, image_features):
        B, P, D = image_features.shape
        patch_size = round(math.sqrt(P))
        assert patch_size % 2 == 0, "Patch size must be divisible by 2."
        image_features = image_features.reshape(B, patch_size, patch_size, D)
        image_features_2x2 = image_features.reshape(B, patch_size // 2, 2, patch_size // 2, 2, D)
        image_features_2x2 = image_features_2x2.permute(0, 1, 3, 2, 4, 5)  
        image_features_2x2 = image_features_2x2.reshape(B, patch_size // 2, patch_size // 2, 4 * D)  # concat 2x2 neighbor patches
        image_features = image_features_2x2.reshape(B, (patch_size // 2) ** 2, 4 * D)
        return image_features
    
    def attention(self, turing_memory, new_feature, update_ratio=0.2):
        T1, D1 = turing_memory.shape
        T2, D2 = new_feature.shape
        assert D1 == D2, f"dimmension not match, {D1} != {D2}"
        model = self.get_model().attention_model
        weight = model.get_weight(turing_memory, new_feature)
        weight = weight * update_ratio  # [T1, T2]
        decay = weight.sum(dim=1, keepdim=True)  # [T0*P, 1], 表示当前NTM memory和新来的feat的相似度
        turing_memory = turing_memory * (1 - decay) + torch.mm(weight, new_feature)
        return turing_memory
    
    def attention2(self, turing_memory, new_feature, update_ratio=0.2):  # deprecated
        T1, D1 = turing_memory.shape
        T2, D2 = new_feature.shape
        assert D1 == D2, f"dimmension not match, {D1} != {D2}"
        model = self.get_model().attention_model
        turing_memory = model.forward(turing_memory, new_feature)
        return turing_memory

    def compress_spatial_features(self, image_features, compress_size=1):
        compress_type = getattr(self.config, "compress_type", None)
        patch_size = round(math.sqrt(image_features.shape[1]))
        assert patch_size * patch_size == image_features.shape[1], f"For ViT feature map, {patch_size}*{patch_size}={patch_size**2} != {image_features.shape[1]}"
        if patch_size == compress_size:
            return image_features
        elif compress_type is not None:
            if 'mean' in self.config.compress_type:
                # TODO: currently use 1 token per frame (or image), direct poolt
                if compress_size == 1:
                    image_features = image_features.mean(dim=1, keepdim=True)
                else:
                    image_features = image_features.view(-1, patch_size, patch_size, image_features.shape[-1])
                    image_features = image_features.permute(0, 3, 1, 2)  # [B*T, D, P, P]
                    pooled_features = F.avg_pool2d(image_features, (patch_size // compress_size, patch_size // compress_size))
                    pooled_features = pooled_features.permute(0, 2, 3, 1)  # [B*T, P, P, D]
                    image_features = pooled_features.view(-1, compress_size * compress_size, pooled_features.shape[-1])
            else:
                raise NotImplementedError(f"`compress_type` {self.config.compress_type} is not supported yet.")
        return image_features
    
    def compress_temporal_features(self, image_features):
        video_long_memory_length = getattr(self.config, "video_long_memory_length", 10)
        video_Turing_memory_length = getattr(self.config, "video_Turing_memory_length", 10)
        video_short_memory_length = getattr(self.config, "video_short_memory_length", 10)  # not used
        video_current_memory_length = getattr(self.config, "video_current_memory_length", 1)
        compress_long_memory_size = getattr(self.config, "compress_long_memory_size", 1)
        compress_Turing_memory_size = getattr(self.config, "compress_Turing_memory_size", 1)
        compress_Turing_update_ratio = getattr(self.config, "compress_Turing_update_ratio", 0.2)
        compress_fn_dic = {
            'drop': drop_feature,
            'merge': merge_feature,
            'kmeans': kmeans_feature,
            'weighted_kmeans': weighted_kmeans_feature,
            'kdrop': k_drop_feature,
            'kmerge': k_merge_feature,
            'attention': attention_feature,
        }
        compress_type = self.config.video_sample_type
        if compress_type in compress_fn_dic:
            compress_fn = compress_fn_dic[compress_type]
        else:
            raise NotImplementedError(f'max_length = {self.config.video_max_frames},'
                                        f'while video_sample_type = {compress_type} is not supported yet.')
        new_image_features = []
        step_indices = []
        step_features = []
        for img_feature in image_features:  # [T, P*P, D]
            cur_start = min(video_current_memory_length, img_feature.shape[0])
            ### Calc Spatial Memory
            if cur_start == 0:
                cur_memory = img_feature[:0]
                long_memory = img_feature
                Turing_memory = img_feature
            else:
                cur_memory = img_feature[-cur_start:]  # [C, P*P, D]
                long_memory = img_feature[:-cur_start]  # [L, P*P, D]
                Turing_memory = img_feature[:-cur_start]  # [L, P*P, D]
            if compress_long_memory_size * compress_long_memory_size != long_memory.shape[1]:
                long_memory = self.compress_spatial_features(long_memory, compress_long_memory_size) # [L, P'*P', D]
            if compress_Turing_memory_size * compress_Turing_memory_size != Turing_memory.shape[1]:
                Turing_memory = self.compress_spatial_features(Turing_memory, compress_Turing_memory_size) # [L, P'*P', D]
            ### Calc Temporal Memory
            if video_long_memory_length == 0 or long_memory.shape[0] == 0:
                long_memory_compreesed = long_memory[:0]
            else:
                long_memory_compreesed, weight, step_long_indices = compress_fn(long_memory, video_long_memory_length) # [L_long, P'*P', D], [L_long]
                ### Calc Retrieved Memory
                sorted_indices = torch.argsort(weight, descending=True)  # [L_long]
                key_centroids = long_memory[sorted_indices]  # [L_long, P'*P', D]
                key_length = 3
                if key_centroids.shape[0] > key_length:
                    key_centroids = key_centroids[:key_length]
                dists = ((long_memory.unsqueeze(1) - key_centroids.unsqueeze(0)) ** 2).sum(dim=3).sum(dim=2).sqrt()  # [L_long, k_L]
                min_indices = torch.argmin(dists, dim=0)  # [k_L]
                key_memory = img_feature[min_indices]
                cur_memory = torch.cat([key_memory, cur_memory], dim=0)
            ### Calc Abstract Memory
            if video_Turing_memory_length == 0 or Turing_memory.shape[0] == 0:
                Turing_memory_compreesed = Turing_memory[:0]
            else:
                Turing_memory_compreesed, _ = attention_feature(Turing_memory, video_Turing_memory_length, self.attention, update_ratio=compress_Turing_update_ratio)
            memory_feature = torch.cat([Turing_memory_compreesed.flatten(0, 1), long_memory_compreesed.flatten(0, 1), cur_memory.flatten(0, 1)], dim=0)
            new_image_features.append(memory_feature)
        return new_image_features

    def cat_proj(self, all_features):  # concatenate features and project them together
        feature_split_size = [x.shape[0] for x in all_features]
        feature_embed = torch.cat(all_features, dim=0)
        feature_proj = self.get_model().mm_projector(feature_embed)
        feature_proj = torch.split(feature_proj, feature_split_size, dim=0)
        return feature_proj
        
    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        labels,
        images,
        features
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or (images is None and features is None) or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and ((images is not None) or (features is not None)) and input_ids.shape[1] == 1:
                target_shape = past_key_values[-1][-1].shape[-2] + 1
                if target_shape - attention_mask.shape[1] >= 0:
                    attention_mask = torch.cat((attention_mask, torch.ones(
                        (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device
                    )), dim=1)
                elif target_shape - attention_mask.shape[1] < 0:
                    attention_mask = attention_mask[:, :target_shape]
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if (features is not None) or (type(images) is list) or (images.ndim == 5):
            compress_size = getattr(self.config, "compress_size", 1)
            if images is not None:
                images = [image if len(image.shape) == 4 else image.unsqueeze(0) for image in images]  # [B, T, C, H, W]
                concat_images = torch.cat([image for image in images], dim=0)  # [B*T, C, H, W]
                image_features = self.encode_images(concat_images)  # [B*T, P, D]
                if getattr(self.config, 'mm_use_4_vision_tokens', False):
                    image_features = self.reshape_2x2_image_features(image_features)  # [B*T, P/4, 4*D]
                image_features = self.compress_spatial_features(image_features, compress_size)  # [B*T, P', D]
                split_sizes = [image.shape[0] for image in images]
                image_features = torch.split(image_features, split_sizes, dim=0)  # [B, T, P, D]
            else:
                image_features = [feat if len(feat.shape) == 3 else feat.unsqueeze(0) for feat in features]
                origin_img_features = image_features
                if getattr(self.config, 'mm_use_4_vision_tokens', False):
                    image_features = [self.reshape_2x2_image_features(img_feature) for img_feature in image_features]  # [B*T, P/4, 4*D]
                image_features = [self.compress_spatial_features(image_feature, compress_size) for image_feature in image_features]  # [B*T, P', D]
            # perform memory consolidation
            image_features = self.compress_temporal_features(image_features)  # [B, TP, D]
            image_features = [x.to(self.device) for x in image_features]  # [B, TP, D]
            image_features = self.cat_proj(image_features)
        else:
            image_features = self.encode_images(images).to(self.device)  # [B, 576, 2048]
            if getattr(self.config, 'mm_use_4_vision_tokens', False):
                image_features = self.reshape_2x2_image_features(image_features)  # [B*T, P/4, 4*D]
            image_features = self.get_model().mm_projector(image_features)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- TODO: double check
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]
        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]  # only input first image_token
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)
            assert cur_image_idx == batch_idx + 1

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def prepare_inputs_labels_for_multimodal_streaming(  # Asynchronous encoding with a SemLock, only for videos, batch_size=1
        self,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        labels
    ):
        assert self.use_video_streaming_mode
        logger = logging.getLogger(__name__)
        vision_tower = self.get_vision_tower()
        if vision_tower is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and input_ids.shape[1] == 1:
                target_shape = past_key_values[-1][-1].shape[-2] + 1
                if target_shape - attention_mask.shape[1] >= 0:
                    attention_mask = torch.cat((attention_mask, torch.ones(
                        (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device
                    )), dim=1)
                elif target_shape - attention_mask.shape[1] < 0:
                    attention_mask = attention_mask[:, :target_shape]
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            return input_ids, position_ids, attention_mask, past_key_values, None, labels
        # Have some tries to avoid deadlock
        attempt_times = 0
        while attempt_times < 300:
            try:
                with self.video_embedding_mem_lock:
                    cur_memory, long_memory_compreesed, Turing_memory_compreesed, _ = self.video_embedding_memory
                    logger.info(f'Read cur_memory={cur_memory.shape} {cur_memory.dtype}, long_memory_compreesed={long_memory_compreesed.shape} {long_memory_compreesed.dtype}, Turing_memory_compreesed={Turing_memory_compreesed.shape} {Turing_memory_compreesed.dtype}')
                    image_feature = torch.cat([Turing_memory_compreesed.flatten(0, 1), long_memory_compreesed.flatten(0, 1), cur_memory.flatten(0, 1)], dim=0)
                    image_features = [image_feature.to(self.device)]
                    break
                    
            except Exception as e:
                logger.error(f'Attempt:{attempt_times} Failed to get video features, Error: {e}')
                image_features = []
                time.sleep(0.1)
                attempt_times += 1
        
        image_features = [x.to(self.device) for x in image_features]  # [B, TP, D]
        image_features = self.cat_proj(image_features)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- TODO: double check
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]  # only input first image_token
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)
            assert cur_image_idx == batch_idx + 1

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels
    
    def embed_video_streaming(  # Asynchronous encoding with a SemLock, only for videos, batch_size=1
        self, 
        images
    ):
        assert self.use_video_streaming_mode
        logger = logging.getLogger(__name__)

        compress_size = getattr(self.config, "compress_size", 1)
        video_long_memory_length = getattr(self.config, "video_long_memory_length", 10)
        video_Turing_memory_length = getattr(self.config, "video_Turing_memory_length", 10)
        video_short_memory_length = getattr(self.config, "video_short_memory_length", 10)  # not used
        video_current_memory_length = getattr(self.config, "video_current_memory_length", 1)
        compress_long_memory_size = getattr(self.config, "compress_long_memory_size", 1)
        compress_Turing_memory_size = getattr(self.config, "compress_Turing_memory_size", 1)
        compress_Turing_update_ratio = getattr(self.config, "compress_Turing_update_ratio", 0.2)
        compress_fn_dic = {
            'drop': drop_feature,
            'merge': merge_feature,
            'kmeans': kmeans_feature,
            'weighted_kmeans': weighted_kmeans_feature,
            'kdrop': k_drop_feature,
            'kmerge': k_merge_feature,
            'uni_kmerge': k_merge_feature,
            'both_kmerge': k_merge_feature,
            'split_kmerge': k_merge_feature,
            'attention': attention_feature,
        }
        
        if type(images) is list or images.ndim == 5:
            assert len(images) == 1
            images = [image if len(image.shape) == 4 else image.unsqueeze(0) for image in images]  # [B, T, C, H, W]
            concat_images = torch.cat([image for image in images], dim=0)  # [B*T, C, H, W]
            image_features = self.encode_images(concat_images)  # [B*T, P, D]
            image_features = self.compress_spatial_features(image_features, compress_size)  # [B*T, P', D]
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)  # [B, T, P, D]
        else:
            raise NotImplementedError('Should input video frames, not a single image')
        image_feature = image_features[0].detach().to(torch.float16).to(self.device)  # [T, P, D]
        img_feature_buffer = image_feature.cpu()

        cur_start = min(video_current_memory_length, image_feature.shape[0])
        if cur_start == 0:
            cur_memory = image_feature[:0]
        else:
            cur_memory = image_feature[-cur_start:]  # [L_c, P*P, D]
        long_memory = image_feature
        Turing_memory = image_feature
        if compress_long_memory_size * compress_long_memory_size != long_memory.shape[1]:
            long_memory = self.compress_spatial_features(long_memory, compress_long_memory_size) # [L_l, P'*P', D]
        if compress_Turing_memory_size * compress_Turing_memory_size != Turing_memory.shape[1]:
            Turing_memory = self.compress_spatial_features(Turing_memory, compress_Turing_memory_size) # [L_t, P'*P', D]
        compress_type = self.config.video_sample_type
        if compress_type in compress_fn_dic:
            compress_fn = compress_fn_dic[compress_type]
        else:
            raise NotImplementedError(f'max_length = {self.config.video_max_frames},'
                                        f'while video_sample_type = {compress_type} is not supported yet.')
        long_memory_compreesed = long_memory
        Turing_memory_compreesed = Turing_memory
        # Read old memory from shared memory, do not need an I/O lock
        if self.video_embedding_memory is not None and len(self.video_embedding_memory) > 0:
            old_cur_memory, old_long_memory_compreesed, old_Turing_memory_compreesed, old_img_feature_buffer = self.video_embedding_memory
            old_long_memory_compreesed = old_long_memory_compreesed.to(self.device)
            old_Turing_memory_compreesed = old_Turing_memory_compreesed.to(self.device)
            img_feature_buffer = torch.cat([old_img_feature_buffer, image_feature.cpu()], dim=0)
            assert isinstance(old_long_memory_compreesed, torch.Tensor) and old_long_memory_compreesed.shape[1:] == long_memory_compreesed.shape[1:]
            long_memory = torch.cat((old_long_memory_compreesed, long_memory_compreesed), dim=0)
            long_memory_compreesed, weight, step_long_indices = compress_fn(long_memory, video_long_memory_length)
            # Retrive key frames
            sorted_indices = torch.argsort(weight, descending=True)  # [L_long]
            key_centroids = long_memory[sorted_indices]  # [L_long, P'*P', D]
            key_length = 3
            if key_centroids.shape[0] > key_length:
                key_centroids = key_centroids[:key_length]
            dists = ((long_memory.unsqueeze(1) - key_centroids.unsqueeze(0)) ** 2).sum(dim=3).sum(dim=2).sqrt()  # [L_long, k_L]
            min_indices = torch.argmin(dists, dim=0)  # [k_L]
            key_memory = img_feature_buffer[min_indices.cpu()].to(self.device)
            cur_memory = torch.cat([key_memory, cur_memory], dim=0)
            Turing_memory = torch.cat((old_Turing_memory_compreesed, Turing_memory_compreesed), dim=0)
            Turing_memory_compreesed, _ = attention_feature(Turing_memory, video_Turing_memory_length, self.attention, update_ratio=compress_Turing_update_ratio)
        # Write to shared memory, need an I/O lock
        with self.video_embedding_mem_lock:
            self.video_embedding_memory[:] = [cur_memory.cpu(), long_memory_compreesed.cpu(), Turing_memory_compreesed.cpu(), img_feature_buffer]  # Only change content
            logger.info(f'Write cur_memory={cur_memory.shape} {cur_memory.dtype}, long_memory_compreesed={long_memory_compreesed.shape} {long_memory_compreesed.dtype}, Turing_memory_compreesed={Turing_memory_compreesed.shape} {Turing_memory_compreesed.dtype}')

        return []


    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
