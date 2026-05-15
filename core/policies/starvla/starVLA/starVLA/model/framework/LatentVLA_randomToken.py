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

# @FRAMEWORK_REGISTRY.register("LatentVLA")
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

        # self.num_latent_tokens = self.config.framework.qwenvl.get("num_latent_token", 16)
        # self.hidden_size = self.qwen_vl_interface.model.config.hidden_size 
        # self.latent_token = nn.Parameter(torch.randn(self.num_latent_tokens, self.qwen_vl_interface.model.config.hidden_size))
        # nn.init.normal_(self.latent_token, mean=0.0, std=0.02)
        self.action_query_num = 16
        self.action_query = nn.Parameter(torch.randn(self.action_query_num, self.qwen_vl_interface.model.config.hidden_size))
        self.dummy_action_token = "🔍" # TODO also can add spacail token to Qwen, but too complex
        self.dummy_action_token_id = self.qwen_vl_interface.processor.tokenizer("🔍", add_special_tokens=False)["input_ids"][0]
        self.dummy_action_prompt = self.dummy_action_token * self.action_query_num
        nn.init.normal_(self.action_query, mean=0.0, std=0.02)

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        """

        """
        batch_images = [example["image"] for example in examples]  #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]

        prompt_suffix = f" Please predict the next {self.chunk_len} robot actions: <action>{self.dummy_action_prompt}<action>."
        instructions = [instruction + prompt_suffix for instruction in instructions]

        actions = [example["action"] for example in examples]  # label [B， len, 7]
        
        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]

        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        
        input_ids = qwen_inputs['input_ids']
        action_mask = (input_ids == self.dummy_action_token_id)  # [B, L]
            
        # ============================================================
        # Hook to replace action token embeddings (OPTIMIZED)
        # ============================================================
        # Pre-compute action positions outside the hook
        batch_size = qwen_inputs['input_ids'].shape[0]
        device = qwen_inputs['input_ids'].device
        action_positions_tensor = torch.full((batch_size, self.action_query_num), 0, dtype=torch.long, device=device)
        valid_counts = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for b in range(batch_size):
            act_pos = torch.where(action_mask[b])[0]
            if len(act_pos) == self.action_query_num:
                action_positions_tensor[b] = act_pos
                valid_counts[b] = True

        def inject_query_hook(module, inputs, output):
            """Replace action placeholder embeddings with learnable queries (VECTORIZED)."""
            query_embed = self.action_query.to(dtype=output.dtype, device=output.device)  # [N, H]
            
            # Vectorized replacement using advanced indexing
            batch_indices = torch.arange(batch_size, device=output.device).unsqueeze(1).expand(-1, self.action_query_num)  # [B, N]
            
            # Only update valid samples (where action token count matches)
            valid_batch_indices = batch_indices[valid_counts]
            valid_action_positions = action_positions_tensor[valid_counts]
            
            if len(valid_batch_indices) > 0:
                output[valid_batch_indices, valid_action_positions, :] = query_embed.unsqueeze(0)
            
            return output
        # Register hook on text embedding layer (this is OK!)
        embedding_layer = self.qwen_vl_interface.model.model.get_input_embeddings()
        hook_handle = embedding_layer.register_forward_hook(inject_query_hook)
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                qwenvl_outputs = self.qwen_vl_interface(
                    **qwen_inputs,
                    output_attentions=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
        finally:
            hook_handle.remove()

        last_hidden = qwenvl_outputs.hidden_states[-1]  # [B, S, H]
        batch_indices_action = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, self.action_query_num)  # [B, N]
        latent_out = last_hidden[batch_indices_action, action_positions_tensor, :]
        
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

        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict],
        **kwargs: str,
    ) -> np.ndarray:
        """
        Steps:
          1. Resize images to training resolution (if specified)
          2. Encode with QwenVL (hidden states retained)
          6. Return normalized action trajectory
        Returns:
            dict:
                normalized_actions (np.ndarray): Shape [B, T, action_dim], diffusion-sampled normalized actions.
        """
        
        batch_images = [example["image"] for example in examples]  #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]

        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        prompt_suffix = f" Please predict the next {self.chunk_len} robot actions: <action>{self.dummy_action_prompt}<action>."
        instructions = [instruction + prompt_suffix for instruction in instructions]
        
        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]

        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        
        input_ids = qwen_inputs['input_ids']
        action_mask = (input_ids == self.dummy_action_token_id)  # [B, L]

        # ============================================================
        # Hook to replace action token embeddings (OPTIMIZED)
        # ============================================================
        # Pre-compute action positions outside the hook
        batch_size = qwen_inputs['input_ids'].shape[0]
        device = qwen_inputs['input_ids'].device
        action_positions_tensor = torch.full((batch_size, self.action_query_num), 0, dtype=torch.long, device=device)
        valid_counts = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for b in range(batch_size):
            act_pos = torch.where(action_mask[b])[0]
            if len(act_pos) == self.action_query_num:
                action_positions_tensor[b] = act_pos
                valid_counts[b] = True

        def inject_query_hook(module, inputs, output):
            """Replace action placeholder embeddings with learnable queries (VECTORIZED)."""
            query_embed = self.action_query.to(dtype=output.dtype, device=output.device)  # [N, H]
            
            # Vectorized replacement using advanced indexing
            batch_indices = torch.arange(batch_size, device=output.device).unsqueeze(1).expand(-1, self.action_query_num)  # [B, N]
            
            # Only update valid samples (where action token count matches)
            valid_batch_indices = batch_indices[valid_counts]
            valid_action_positions = action_positions_tensor[valid_counts]
            
            if len(valid_batch_indices) > 0:
                output[valid_batch_indices, valid_action_positions, :] = query_embed.unsqueeze(0)
            
            return output
        # Register hook on text embedding layer (this is OK!)
        embedding_layer = self.qwen_vl_interface.model.model.get_input_embeddings()
        hook_handle = embedding_layer.register_forward_hook(inject_query_hook)
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                qwenvl_outputs = self.qwen_vl_interface(
                    **qwen_inputs,
                    output_attentions=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
        finally:
            hook_handle.remove()

        last_hidden = qwenvl_outputs.hidden_states[-1]  # [B, S, H]
        batch_indices_action = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, self.action_query_num)  # [B, N]
        latent_out = last_hidden[batch_indices_action, action_positions_tensor, :]

        state = torch.from_numpy(np.array(state)).to(latent_out.device, dtype=latent_out.dtype) if state is not None else None
        
        # Step 4: Action Expert Forward
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(latent_out, state)  # (B, chunk_len, action_dim)

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}



if __name__ == "__main__":
    from omegaconf import OmegaConf
    import debugpy
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="./examples/Robotwin/train_files/starvla_cotrain_robotwin.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Rank 0 waiting for debugger attach on port 10092...")
    debugpy.wait_for_client()
    args.config_yaml = "examples/MultiRobot/train_files/starvla_cotrain_multiRobot.yaml"
    cfg = OmegaConf.load(args.config_yaml)
    # try get model
    # cfg.framework.action_model.action_hidden_dim = 2048

    # cfg.framework.qwenvl.base_vlm = "./playground/Pretrained_models/Florence-2-large"
    

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