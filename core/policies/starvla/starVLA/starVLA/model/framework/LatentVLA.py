# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Junqiu YU / Fudan University] in [2025]. 
# Design and Merged by [Jinhui YE / HKUST University] in [2025].
"""
Qwen-GR00T Framework
A lightweight implementation that Qwen-VL + Flow-matching head to directly predict continuous actions
Flow-matching header is copyright from GR00T N1.5,
"""
import sys
from pathlib import Path

# Add workspace root to Python path if not already there
_workspace_root = Path(__file__).parent.parent.parent.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

from typing import List
from tqdm import tqdm
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image



from starVLA.training.trainer_utils import initialize_overwatch
from deployment.model_server.tools.image_tools import to_pil_preserve

logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.modules.action_model.GR00T_ActionHeader import get_action_model, FlowmatchingActionHead
from starVLA.training.trainer_utils.trainer_tools import resize_images
from starVLA.model.tools import FRAMEWORK_REGISTRY

def get_image_token_counts(batch_inputs):
    IMAGE_TOKEN_ID = 151655 
    
    # input_ids shape: [Batch_Size, Seq_Len]
    # result shape: [Batch_Size]
    num_tokens_per_sample = torch.sum(batch_inputs['input_ids'] == IMAGE_TOKEN_ID, dim=1)
    
    return num_tokens_per_sample

class ActionDependentReconstructor(nn.Module):
    """
    强制依赖 Action 的重建模块。
    机制：
    1. 结构上：使用 Action 作为 Query 去查询 Unmasked Tokens。
    2. Loss上：使用对比损失，惩罚"不使用Action也能重建好"的情况。
    """
    def __init__(self, hidden_size=2048, action_dim=7, bottleneck_dim=256, num_latents=16, mask_ratio=0.5):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.num_latents = num_latents
        self.bottleneck_dim = bottleneck_dim

        # 1. 降维 (Compression) - 必不可少，防止信息泄露
        self.down_proj = nn.Sequential(
            nn.Linear(hidden_size, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.GELU()
        )

        # 2. Action Mapping
        self.action_proj = nn.Linear(action_dim, bottleneck_dim)

        # 3. Positional Embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, num_latents, bottleneck_dim) * 0.02)

        # 4. 核心组件：Cross-Attention Decoder
        # Query = Action + Mask Query
        # Key/Value = Unmasked Tokens
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=bottleneck_dim,
            num_heads=4,
            batch_first=True,
            dropout=0.1
        )
        
        # 简单的 FFN
        self.ffn = nn.Sequential(
            nn.Linear(bottleneck_dim, bottleneck_dim * 4),
            nn.GELU(),
            nn.Linear(bottleneck_dim * 4, bottleneck_dim)
        )
        self.norm1 = nn.LayerNorm(bottleneck_dim)
        self.norm2 = nn.LayerNorm(bottleneck_dim)

        # 5. Prediction Head
        self.pred_head = nn.Linear(bottleneck_dim, bottleneck_dim)

    def _predict(self, unmasked_tokens, unmasked_pos, mask_pos, action_embeds):
        """
        内部预测函数
        unmasked_tokens: [B, N_keep, D]
        unmasked_pos: [B, N_keep, D]
        mask_pos: [B, N_mask, D]
        action_embeds: [B, 1, D] (or similar)
        """
        B = unmasked_tokens.shape[0]
        
        # --- 构建 Query ---
        # Query 必须包含：
        # 1. 我在哪？(mask_pos: 位置信息)
        # 2. 我要做什么？(action_embeds: 语义引导)
        # 我们把 action 加到每一个 mask query 上，作为条件
        # action_embeds: [B, 1, D] -> [B, N_mask, D]
        action_cond = action_embeds.unsqueeze(1).expand(-1, mask_pos.shape[1], -1)
        
        # Query = Mask的位置 + Action的信息
        queries = mask_pos + action_cond 

        # --- 构建 Key/Value ---
        # Key = Unmasked的位置 + Unmasked的内容
        keys = unmasked_tokens + unmasked_pos
        values = unmasked_tokens

        # --- Cross Attention ---
        # Q 找 K，利用 Action 也就是 Q 中的信息，去 K 中检索需要补全的内容
        attn_out, _ = self.cross_attn(query=queries, key=keys, value=values)
        
        # Residual & Norm & FFN
        x = self.norm1(attn_out + queries) # Res connect
        x = self.norm2(x + self.ffn(x))
        
        return self.pred_head(x)

    def forward(self, latent_tokens, actions):
        """
        latent_tokens: [B, 16, 2048]
        actions: [B, 8, 7] -> 我们通常取最后一帧或平均池化作为条件
        """
        B, L, H = latent_tokens.shape
        device = latent_tokens.device

        # 1. 预处理
        # 压缩 Latent
        latents = self.down_proj(latent_tokens) # [B, 16, 256]
        # 处理 Action: 取 Action 序列的 Mean 作为全局条件，或者只取最后一个
        # actions: [B, 8, 7] -> [B, 256]
        action_feat = self.action_proj(actions.mean(dim=1)) 

        # 2. 准备 Target (Detach!)
        targets = latents.detach()

        # 3. Masking Logic (生成随机 Mask)
        num_masked = int(self.mask_ratio * L)
        noise = torch.rand(B, L, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        
        ids_keep = ids_shuffle[:, num_masked:] # 保留的部分
        ids_mask = ids_shuffle[:, :num_masked] # 被遮挡的部分

        # Gather Unmasked Tokens (作为 Key/Value)
        # [B, N_keep, D]
        unmasked_tokens = torch.gather(latents, 1, ids_keep.unsqueeze(-1).expand(-1, -1, self.bottleneck_dim))
        
        # Gather Positional Embeddings
        pos_embed = self.pos_embed.expand(B, -1, -1)
        unmasked_pos = torch.gather(pos_embed, 1, ids_keep.unsqueeze(-1).expand(-1, -1, self.bottleneck_dim))
        mask_pos = torch.gather(pos_embed, 1, ids_mask.unsqueeze(-1).expand(-1, -1, self.bottleneck_dim))
        
        # Gather Targets (只计算被 Mask 部分的 Loss)
        target_tokens = torch.gather(targets, 1, ids_mask.unsqueeze(-1).expand(-1, -1, self.bottleneck_dim))

        # ==========================================
        # 核心逻辑：对比推断 (Contrastive Inference)
        # ==========================================

        # A. 正向推断 (Positive Pass): 给正确的 Action
        pred_pos = self._predict(unmasked_tokens, unmasked_pos, mask_pos, action_feat)

        # B. 负向推断 (Negative Pass): 给错误的 Action (比如 Shuffle 或 全零)
        # 这里我们构造一个"与当前图无关"的 Action，比如 Batch 内随机 Shuffle
        with torch.no_grad(): # 负样本不需要更新 Action Projection 的梯度，只作为对比基准
            action_feat_neg = action_feat[torch.randperm(B)]
        
        pred_neg = self._predict(unmasked_tokens, unmasked_pos, mask_pos, action_feat_neg)

        # ==========================================
        # Loss 计算：Triplet / Contrastive
        # ==========================================
        
        # 1. 计算两个 Pass 的 MSE
        mse_pos = (pred_pos - target_tokens) ** 2
        mse_pos = mse_pos.mean(dim=-1).mean() # Scalar

        mse_neg = (pred_neg - target_tokens) ** 2
        mse_neg = mse_neg.mean(dim=-1).mean() # Scalar

        # 2. Triplet Margin Loss
        # 目标：mse_pos 要很小，且 (mse_neg - mse_pos) 要大于 margin
        # 即：如果没有正确的 Action，误差应该显著增大
        margin = 0.5 # 这是一个超参数，表示你希望"有Action"比"没Action"强多少
        
        # loss_contrast = max(0, margin + pos - neg)
        # 如果 neg 很大 (重建很差)，loss 就为 0，只优化 pos
        # 如果 neg 很小 (没 Action 也能重建)，loss 就会惩罚模型
        loss_contrast = torch.clamp(margin + mse_pos - mse_neg, min=0.0)

        # 总 Loss
        # alpha 用于平衡：主要还是为了重建好(mse_pos)，其次才是拉开差距
        alpha = 0.5 
        total_loss = mse_pos + alpha * loss_contrast

        return total_loss, mse_pos, mse_neg

@FRAMEWORK_REGISTRY.register("LatentVLA")
class LatentVLA(baseframework):
    """
    Multimodal vision-language-action model.

    Components:
      - Qwen2.5 VL interface for fused language/vision token embeddings
      - Layer-wise QFormer for multi-layer feature aggregation
      - DINO encoder for dense multi-view spatial tokens
      - DiT diffusion head for future action sequence modeling

    Focus: Predict future continuous actions conditioned on images + instruction.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Construct all submodules and cache key configuration values.

        Args:
            config: Hierarchical configuration (OmegaConf/dict) containing framework + trainer sections.
            **kwargs: Reserved for future overrides (unused).
        """
        super().__init__()
        self.config = config
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        # align dims --> we should put them to config or no?
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = self.qwen_vl_interface.model.config.hidden_size

        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)  # 修复后续引用

        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size

        self.num_latent_action_query = 16
        self.latent_action_query = "".join([f"<|action_{i}|>" for i in range(self.num_latent_action_query)])
        self.action_token_ids = None  # cached {'first','last'}
        
        # self.lm_head = nn.Linear(self.qwen_vl_interface.model.config.hidden_size, self.qwen_vl_interface.model.config.text_config.vocab_size, bias=False)

        # self.cross_attn = torch.nn.MultiheadAttention(
        #     embed_dim=self.qwen_vl_interface.model.config.hidden_size,
        #     num_heads=config.framework.get("cross_attn_heads", 8),
        #     batch_first=True,
        #     dropout=config.framework.get("cross_attn_dropout", 0.1)
        # )
        self.reconstructor = ActionDependentReconstructor(
            hidden_size=self.qwen_vl_interface.model.config.hidden_size,
            bottleneck_dim=256,
            action_dim=7,
            num_latents=self.num_latent_action_query,
            mask_ratio=0.75
        )
        

    def _ensure_action_token_ids(self, tokenizer):
        if self.action_token_ids is None:
            self.action_token_ids = {
                "first": tokenizer.convert_tokens_to_ids("<|action_0|>"),
                "last": tokenizer.convert_tokens_to_ids(f"<|action_{self.num_latent_action_query-1}|>"),
            }

    # ---------------------------------------------------------------------
    # Action block helpers
    # ---------------------------------------------------------------------
    def _get_action_block_start(self, input_ids_1d: torch.Tensor, tokenizer) -> int:
        self._ensure_action_token_ids(tokenizer)
        first_id = self.action_token_ids["first"]
        last_id = self.action_token_ids["last"]

        pos = (input_ids_1d == int(first_id)).nonzero(as_tuple=True)[0]
        if pos.numel() == 0:
            return -1

        start = int(pos[0].item())
        end = start + self.num_latent_action_query
        if end > input_ids_1d.shape[0]:
            return -1
        if int(input_ids_1d[end - 1].item()) != int(last_id):
            return -1
        return start

    def _extract_action_query_hidden_states(
        self,
        hidden_states: torch.Tensor,   # [B, S, H]
        input_ids: torch.Tensor,       # [B, S]
        tokenizer,
        return_starts: bool = False,
    ):
        self._ensure_action_token_ids(tokenizer)

        B = hidden_states.shape[0]
        out = []
        starts = []
        for b in range(B):
            start = self._get_action_block_start(input_ids[b], tokenizer)
            assert start != -1, "No valid contiguous action token block found in the sequence."
            end = start + self.num_latent_action_query
            out.append(hidden_states[b, start:end, :])
            starts.append(start)

        out = torch.stack(out, dim=0)  # [B, K, H]
        if return_starts:
            return out, torch.tensor(starts, device=input_ids.device, dtype=torch.long)
        return out

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        """

        """
        batch_images = [example["image"] for example in examples]  # [B, [PIL...]]
        instructions = [example["lang"] + self.latent_action_query for example in examples]  # L + A
        
        actions = [example["action"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None
        
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions
        )

        # Step 1: QWenVL input format
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
            posteriori_last_hidden = qwenvl_outputs.hidden_states[-1]  # [B, S, H]
            latent_out, posteriori_action_starts = self._extract_action_query_hidden_states(
                posteriori_last_hidden,
                qwen_inputs["input_ids"],
                self.qwen_vl_interface.processor.tokenizer,
                return_starts=True
            )  # [B, K, H], [B]

            # avg_latent = latent_out.mean(dim=1)  # [B, H]
            # logits = self.qwen_vl_interface.model.lm_head(avg_latent)  # [B, vocab_size]
            # # logits = self.lm_head(avg_latent)  # [B, vocab_size]

            # tokenizer = self.qwen_vl_interface.processor.tokenizer
            # vocab_size = logits.shape[-1]
            # batch_size = logits.shape[0]
            
            # cls_targets = torch.zeros((batch_size, vocab_size), device=logits.device, dtype=torch.float32)
            
            # raw_langs = [example["lang"] for example in examples]
            # for b, text in enumerate(raw_langs):
            #     token_ids = tokenizer.encode(text, add_special_tokens=False)
                
            #     if len(token_ids) > 0:
            #         unique_ids = list(set(token_ids))

            #         valid_ids = [idx for idx in unique_ids if idx < vocab_size]
                    
            #         if valid_ids:
            #             indices = torch.tensor(valid_ids, device=logits.device, dtype=torch.long)
            #             cls_targets[b].scatter_(0, indices, 1.0)
            #             cls_targets[b] /= cls_targets[b].sum()

            # vl_hidden_states = []
            # for b in range(posteriori_action_starts.shape[0]):
            #     start_of_latent = posteriori_action_starts[b].item()
            #     vl_hidden_states.append(posteriori_last_hidden[b, :start_of_latent, :])
            # vl_hidden_states = torch.stack(vl_hidden_states, dim=0)

            # queryed_vl_feature, _ = self.cross_attn(query=latent_out, key=vl_hidden_states, value=vl_hidden_states)
            # latent_out = latent_out + queryed_vl_feature

        
        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.tensor(
                np.array(actions), device=latent_out.device, dtype=latent_out.dtype
            )  # [B, T_full, action_dim]
            actions_target = actions[:, -(self.future_action_window_size+1):, :]  # (B, chunk_len, action_dim)

            repeated_diffusion_steps = (
                self.config.trainer.get("repeated_diffusion_steps", 4) if self.config and self.config.trainer else 4
            )
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            latent_out_repeated = latent_out.repeat(repeated_diffusion_steps, 1, 1)
            
            state_repeated = None
            if state is not None:
                state = torch.tensor(
                    np.array(state), device=latent_out.device, dtype=latent_out.dtype
                )
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

            action_loss = self.action_model(latent_out_repeated, actions_target_repeated, state_repeated)  # (B, chunk_len, action_dim)
            # cls_loss = F.cross_entropy(logits, cls_targets)
            recon_loss, mse_pos, mse_neg = self.reconstructor(latent_out, actions_target)

        # return {"action_loss": action_loss}
        return {
            "action_loss": action_loss,
            "cls_loss": 0, #cls_loss
            "recon_loss": recon_loss, # recon_loss
            "mse_pos": mse_pos,
            "mse_neg": mse_neg
        }

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict],
        **kwargs: str,
    ) -> np.ndarray:
        if type(examples) is not list:
            examples = [examples]

        # robustly preserve PIL for each view
        batch_images = []
        for ex in examples:
            imgs = ex["image"]
            if isinstance(imgs, list):
                batch_images.append([to_pil_preserve(im) for im in imgs])
            else:
                batch_images.append([to_pil_preserve(imgs)])

        instructions_posteriori = [ex["lang"] + self.latent_action_query for ex in examples]
        state = [ex["state"] for ex in examples] if "state" in examples[0] else None

        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions_posteriori
        )

        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )

            last_hidden = qwenvl_outputs.hidden_states[-1]
            latent_out, action_starts = self._extract_action_query_hidden_states(
                last_hidden,
                qwen_inputs["input_ids"],
                self.qwen_vl_interface.processor.tokenizer,
                return_starts=True
            )  # [B, K, H]

        state_tensor = None
        if state is not None:
            state_tensor = torch.from_numpy(np.array(state)).to(latent_out.device, dtype=latent_out.dtype)

        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(latent_out, state_tensor)

        return {"normalized_actions": pred_actions.detach().cpu().numpy()}



if __name__ == "__main__":
    from omegaconf import OmegaConf
    import debugpy
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="./examples/LIBERO/train_files/starvla_cotrain_libero.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    
    model: LatentVLA = LatentVLA(cfg)
    print(model)

    # fake sample 
    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    # Create a sample
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16), # action_chunk, action_dim
        "image": [image], # three views
        "lang": "Put all the toys in the child's room - the three board games (two on the bed and one on the table), the two jigsaw puzzles on the table, and the tennis ball on the table - inside the toy box on the table in the child's room.",
        # "state" : np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16), # chunk, state_dim
    }
    sample2 = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16), # action_chunk, action_dim
        "image": [image], # three views
        "lang": "Put all the toys in the child's room - the three board games (two on the bed and one on the table), the two jigsaw puzzles on the table, and the tennis ball on the table - inside the toy box on the table in the child's room.",
        # "state" : np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16), # chunk, state_dim
    }

    batch  = [sample, sample2]  # batch size 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    action_loss = forward_output['action_loss']
    print(f"Action Loss: {action_loss.item()}")

    # test predict action
    predict_output = model.predict_action(examples=[sample]) #, state=[batch[0]["state"]]
    normalized_actions = predict_output['normalized_actions']
    print(f"Unnormalized Action: {normalized_actions}")

    # # Advance: try forward model with dataloader
    # # can be fake sample， but here get from dataloader for simpler
    vla_dataset_cfg = cfg.datasets.vla_data
    from torch.utils.data import DataLoader
    from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn
    cfg.datasets.vla_data.include_state = "False"
    dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)

    train_dataloader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=1,  # For Debug
        collate_fn=collate_fn,
    )
    # forward model with dataloader
    for batch in tqdm(train_dataloader, desc="Processing Batches"):
        # try get model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model(batch)
        # break

    action = model.predict_action(examples=batch)
    print("Finished")