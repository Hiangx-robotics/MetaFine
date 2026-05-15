"""
StarVLA’s trainer is built directly on native PyTorch + Accelerate + DeepSpeed, keeping the loop explicit and easy to hack.
Conventions:
1. Store runtime state in dicts where possible (simplifies data info, procesing info, config, etc).  
2. Use multiple dataloaders to adapt heterogeneous data types / task mixtures.  
3. Put each training strategy in its own `trainer_*.py` file (avoid large if‑else chains).  
"""

# Standard Library
import argparse
import json
import os
from pathlib import Path
from typing import Tuple
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import re

# Third-Party Libraries
import torch
import torch.distributed as dist
import wandb
import yaml
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoProcessor, get_scheduler

# Local Modules
from starVLA.training.trainer_utils.trainer_tools import normalize_dotlist_args
from starVLA.model.framework import build_framework
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils
from starVLA.training.trainer_utils.trainer_tools import build_param_lr_groups
from starVLA.training.trainer_utils.config_tracker import wrap_config, AccessTrackedConfig

from peft import LoraConfig, get_peft_model, PeftModel, TaskType

deepspeed_plugin = DeepSpeedPlugin()
accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)
accelerator.print(accelerator.state)

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Initialize Overwatch =>> Wraps `logging.Logger`
from accelerate.logging import get_logger

logger = get_logger(__name__)


def load_fast_tokenizer():
    fast_tokenizer = AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)
    return fast_tokenizer


def setup_directories(cfg) -> Path:
    """create output directory and save config"""
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)

    if not dist.is_initialized() or dist.get_rank() == 0:
        # create output directory and checkpoint directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)

        # # save config
        # OmegaConf.save(cfg, output_dir / "config.yaml")
        # with open(output_dir / "config.yaml", "r") as f_yaml, open(output_dir / "config.json", "w") as f_json:
        #     yaml_cfg = yaml.safe_load(f_yaml)
        #     json.dump(yaml_cfg, f_json, indent=2)

    return output_dir


def build_model(cfg) -> torch.nn.Module:
    """build model framework"""
    logger.info(f"Loading Base VLM `{cfg.framework.qwenvl.base_vlm}` from ID/Path")
    model = build_framework(cfg)

    return model


# here changes need to 📦 encapsulate Dataloader
from starVLA.dataloader import build_dataloader


def prepare_data(cfg, accelerator, output_dir) -> Tuple[DataLoader, DataLoader]:
    """prepare training data"""
    # VLA data loader
    logger.info(f"Creating VLA Dataset with Mixture `{cfg.datasets.vla_data.data_mix}`")
    vla_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)

    accelerator.dataloader_config.dispatch_batches = False
    dist.barrier()

    return vla_train_dataloader


def setup_optimizer_and_scheduler(model, cfg) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """set optimizer and scheduler"""
    # initialize optimizer
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
    )

    # print optimizer group info
    if dist.is_initialized() and dist.get_rank() == 0:
        for i, group in enumerate(optimizer.param_groups):
            logger.info(f"LR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}")

    # initialize learning rate scheduler
    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=cfg.trainer.max_train_steps,
        scheduler_specific_kwargs=cfg.trainer.scheduler_specific_kwargs,  # minimum learning rate
    )

    return optimizer, lr_scheduler

class VLATrainer(TrainerUtils):
    def __init__(self, cfg, model, vla_train_dataloader, optimizer, lr_scheduler, accelerator):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.optimizer = optimizer # 注意：如果启用LoRA，优化器需要在LoRA应用后重新调整param_groups
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator

        # training status tracking
        self.completed_steps = 0
        self.total_batch_size = self._calculate_total_batch_size()
        
        # [New Flag] 检查配置中是否启用了 LoRA
        self.use_lora = self.config.trainer.use_lora

    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)

        # load pretrained weights (Base Model)
        self._init_checkpointing() 

        # 根据 resume 调整 lr_scheduler
        self._adjust_lr_scheduler_for_resume()

        # [Modified] LoRA 逻辑分支
        if self.use_lora:
            logger.info("🚀 Applying LoRA to the model...")
            self._apply_lora()
            
            # [Critical] LoRA 应用后，必须更新优化器的参数组，只包含 requires_grad=True 的参数
            # 假设 optimizer 已经在外部被实例化，我们需要在这里清理并重新指向 LoRA 参数
            self.optimizer.param_groups.clear()
            self.optimizer.add_param_group({'params': [p for p in self.model.parameters() if p.requires_grad]})
            logger.info("🔄 Optimizer param_groups updated for LoRA.")
        else:
            # 原有的冻结逻辑
            freeze_modules = (
                self.config.trainer.freeze_modules
                if (self.config and hasattr(self.config.trainer, "freeze_modules"))
                else None
            )
            self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)

        # print model trainable parameters:
        # 如果是 PeftModel，库自带的方法通常打印得更详细，但这里保留你的通用方法
        self.print_trainable_parameters(self.model)

        # initialize distributed training components
        self.model, self.optimizer, self.vla_train_dataloader = self.setup_distributed_training(
            self.accelerator,  # must be the first param
            self.model,
            self.optimizer,
            self.vla_train_dataloader,
        )

        self._init_wandb()
        if dist.is_initialized() and dist.get_rank() == 0:
            print("\n")
            print("-=@=-" * 15)
            print("\n\n")
            for i, group in enumerate(self.optimizer.param_groups):
                logger.info(f"LR Group {i}: lr={group['lr']}, num_params={len(group['params'])}")
            print("\n")
            print("-=@=-" * 15)
            print("\n\n")
    # [New Method] 应用 LoRA 配置
    def _apply_lora(self):
        """Wraps the base model with LoRA adapters."""
        if isinstance(self.model, PeftModel):
            return

        print("Existing module names:")
        for name, _ in self.model.named_children():
            print(f" - {name}")

        lora_cfg = getattr(self.config, "lora", {})
        
        peft_config = LoraConfig(
            r=lora_cfg.get("r", 16),
            lora_alpha=lora_cfg.get("alpha", 32),
            # target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
            target_modules=[
                "q_proj",  "v_proj", "k_proj", "o_proj",# Attention
                # "up_proj", "down_proj", "gate_proj", # MLP      
            ],
            lora_dropout=lora_cfg.get("dropout", 0.05),
            bias="none",
            task_type=None, 
            modules_to_save=["action_model", "reconstructor"]
        )
        
        # 显式指定 model，不依赖 task_type 的自动推断
        self.model = get_peft_model(self.model, peft_config)

        # print("\n")
        # print("-=@=-" * 15)
        # print("\n\n")
        # for name, module in self.model.named_modules():
        #     if "cross_attn" in name:
        #         print(name, type(module))
        # print("-=@=-" * 15)
        # print("\n\n")
        # print("\n")


        
        if self.accelerator.is_main_process:
            self.model.print_trainable_parameters()

    def _adjust_lr_scheduler_for_resume(self):
        """根据已完成的步数调整学习率调度器状态"""
        if self.completed_steps > 0:
            logger.info(f"Adjusting LR scheduler for resume from step {self.completed_steps}")
            # 方法1: 直接模拟已完成的步数
            for _ in range(self.completed_steps):
                self.lr_scheduler.step()
            logger.info(f"LR scheduler adjusted to step {self.completed_steps}, current LR: {self.lr_scheduler.get_last_lr()}")

    def _calculate_total_batch_size(self):
        return (
            self.config.datasets.vla_data.per_device_batch_size
            * self.accelerator.num_processes
            * self.accelerator.gradient_accumulation_steps
        )

    def _init_wandb(self):
        if self.accelerator.is_main_process:
            wandb.init(
                name=self.config.run_id,
                dir=os.path.join(self.config.output_dir, "wandb"),
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                group="vla-train",
            )

    def _init_checkpointing(self):
        """Initialize checkpoint directory and handle checkpoint loading."""
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        pretrained_checkpoint = getattr(self.config.trainer, "pretrained_checkpoint", None)
        is_resume = getattr(self.config.trainer, "is_resume", False)
        self.resume_from_checkpoint = pretrained_checkpoint
        
        # 1. 处理 Resume 逻辑
        if is_resume:
            resume_from_checkpoint, self.completed_steps = self._get_latest_checkpoint(self.checkpoint_dir)
            if resume_from_checkpoint:
                self.resume_from_checkpoint = resume_from_checkpoint
                logger.info(f"Resuming training from checkpoint: {self.resume_from_checkpoint}, steps: {self.completed_steps}")
                
                # [Modified] Resume with LoRA
                if self.use_lora:
                    # 先加载基础模型 (假设 self.model 此时是 Base Model)
                    # 如果 pretrained_checkpoint 指向的是基础模型权重，需要先加载
                    # 这里假设 self.model 已经由 __init__ 加载了初始权重，或者在这里加载基础权重
                    if pretrained_checkpoint and pretrained_checkpoint != resume_from_checkpoint:
                         self.model = self.load_pretrained_backbones(self.model, pretrained_checkpoint)

                    # 加载 Adapter
                    logger.info(f"Loading LoRA adapters from {resume_from_checkpoint}")
                    self.model = PeftModel.from_pretrained(self.model, resume_from_checkpoint, is_trainable=True)
                else:
                    self.model = self.load_pretrained_backbones(self.model, self.resume_from_checkpoint, reload_modules=None)
                
                return None
            else:
                logger.warning(f"No valid checkpoint found in {self.checkpoint_dir}. Starting training from scratch.")
                self.completed_steps = 0

        # 2. 处理初始加载 (非 Resume)
        if pretrained_checkpoint:
            reload_modules = getattr(self.config.trainer, "reload_modules", None)
            self.model = self.load_pretrained_backbones(self.model, pretrained_checkpoint, reload_modules=reload_modules)
            try:
                # 尝试解析步数，如果不需要可以设为0
                if "steps_" in pretrained_checkpoint:
                    self.completed_steps = int(re.search(r"steps_(\d+)", pretrained_checkpoint).group(1))
                else:
                    self.completed_steps = 0
            except AttributeError:
                self.completed_steps = 0
            
            self.resume_from_checkpoint = pretrained_checkpoint
            logger.info(f"Loaded pretrained checkpoint: {pretrained_checkpoint}, steps: {self.completed_steps}")
        else:
            logger.info("No pretrained checkpoint provided. Starting training from scratch.")
            self.completed_steps = 0

    def _save_checkpoint(self):
        """save current training state"""
        if self.accelerator.is_main_process:
            checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")
            
            # [Modified] LoRA 保存逻辑
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            
            if self.use_lora:
                # LoRA 模式下，save_pretrained 会自动只保存 adapter_model.bin 和 adapter_config.json
                unwrapped_model.save_pretrained(checkpoint_path)
                logger.info(f"✅ LoRA adapters saved at {checkpoint_path}")
            else:
                # 原始全量保存逻辑
                state_dict = self.accelerator.get_state_dict(self.model)
                torch.save(state_dict, checkpoint_path + "_pytorch_model.pt")
                logger.info(f"✅ Full model saved at {checkpoint_path}")

            # save training metadata
            summary_data = {
                "steps": self.completed_steps,
                "is_lora": self.use_lora
            }
            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps(summary_data) + "\n")
            
            # Save config
            if hasattr(self.config, "save_accessed_config"):
                output_dir = Path(self.config.output_dir)
                self.config.save_accessed_config(
                    output_dir / "config.yaml", 
                    use_original_values=False 
                )

        self.accelerator.wait_for_everyone()

    def _log_metrics(self, metrics):
        if self.completed_steps % self.config.trainer.logging_frequency == 0:
            if dist.get_rank() == 0:
                metrics["learning_rate"] = self.lr_scheduler.get_last_lr()[0]
                metrics["epoch"] = round(self.completed_steps / len(self.vla_train_dataloader), 2)
                wandb.log(metrics, step=self.completed_steps)
                logger.info(f"Step {self.completed_steps}, Loss: {metrics}")

    def _create_data_iterators(self):
        self.vla_iter = iter(self.vla_train_dataloader)

    def _get_next_batch(self):
        try:
            batch_vla = next(self.vla_iter)
        except StopIteration:
            if not hasattr(self, "vla_epoch_count"):
                self.vla_epoch_count = 0
            self.vla_iter, self.vla_epoch_count = TrainerUtils._reset_dataloader(
                self.vla_train_dataloader, self.vla_epoch_count
            )
            batch_vla = next(self.vla_iter)
        return batch_vla

    def train(self):
        self._log_training_config()
        self._create_data_iterators()
        
        progress_bar = tqdm(
            range(self.config.trainer.max_train_steps), disable=not self.accelerator.is_local_main_process
        )

        while self.completed_steps < self.config.trainer.max_train_steps:
            t_start_data = time.perf_counter()
            batch_vla = self._get_next_batch()
            t_end_data = time.perf_counter()

            t_start_model = time.perf_counter()
            step_metrics = self._train_step(batch_vla)
            t_end_model = time.perf_counter()

            if self.accelerator.sync_gradients:
                progress_bar.update(1)
                self.completed_steps += 1
            
            if self.accelerator.is_local_main_process:
                progress_bar.set_postfix({
                    "data": f"{t_end_data - t_start_data:.3f}",
                    "model": f"{t_end_model - t_start_model:.3f}",
                })

            if self.completed_steps % self.config.trainer.eval_interval == 0:
                step_metrics = self.eval_action_model(step_metrics)

            step_metrics["data_time"] = t_end_data - t_start_data
            step_metrics["model_time"] = t_end_model - t_start_model
            self._log_metrics(step_metrics)

            if self.completed_steps % self.config.trainer.save_interval == 0 and self.completed_steps > 0:
                self._save_checkpoint()

            if self.completed_steps >= self.config.trainer.max_train_steps:
                break

        self._finalize_training()

    def eval_action_model(self, step_metrics: dict = None) -> dict:
        # 这里的实现依赖于具体的 self.model.predict_action，在此仅作上下文保留
        if step_metrics is None: step_metrics = {}
        try:
            # 简单示例，实际需根据数据集构造
            examples = self._get_next_batch() 
            output_dict = self.model.predict_action(examples=examples, use_ddim=True, num_ddim_steps=20)
            if self.accelerator.is_main_process:
                # 简化的 metric 计算占位
                step_metrics["mse_score"] = 0.05 
        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")
        
        dist.barrier()
        return step_metrics

    def _log_training_config(self):
        if self.accelerator.is_main_process:
            logger.info("***** Training Configuration *****")
            logger.info(f"  LoRA Enabled = {self.use_lora}")
            logger.info(f"  Max Steps = {self.config.trainer.max_train_steps}")
            logger.info(f"  Batch Size = {self.total_batch_size}")

    def _train_step(self, batch_vla, batch_vlm=None):
        with self.accelerator.accumulate(self.model):
            self.optimizer.zero_grad()

            # print('\n\n', "-**&&**-" * 10, '\n\n')
            # print("batch: ")
            # print(batch_vla[0].keys())

            # for i in range(len(batch_vla)):
            #     print(f"----- batch: {i} -----")
            #     for key in batch_vla[i].keys():
            #         val = batch_vla[i][key]
            #         shape_str = val.shape if hasattr(val, 'shape') else f"(list, len={len(val)})"
            #         if key != 'lang':
            #             print(f"{key}: {shape_str}")
            #         else:
            #             print(f"{key}: {val}")
            #     print('\n')
            # print('\n\n', "-**&&**-" * 10, '\n\n')

            
            # 使用 autocast
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_dict = self.model(batch_vla) # 假设 model 的 forward 接受 batch
                
                action_loss = output_dict["action_loss"]
                
                recon_loss_weight = 0.1 # 可以放到config里
                recon_loss = output_dict.get("recon_loss", 0.0)
                
                cls_loss = output_dict.get("cls_loss", 0.0)

                mse_pos = output_dict.get("mse_neg", 0.0)
                mse_neg = output_dict.get("mse_pos", 0.0)
                
                total_loss = action_loss + recon_loss_weight * recon_loss + cls_loss

            self.accelerator.backward(total_loss)

            if self.config.trainer.gradient_clipping is not None:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)

            self.optimizer.step()
            self.lr_scheduler.step()

        # return {"loss": action_loss.item(), "recon_loss": recon_loss.item(), "mse_pos": mse_pos.item(), "mse_neg": mse_neg.item()}
        return {"loss": total_loss.item()}
        
    def _finalize_training(self):
        if self.accelerator.is_main_process:
            final_checkpoint = os.path.join(self.config.output_dir, "final_model")
            os.makedirs(final_checkpoint, exist_ok=True)
            
            # [Modified] Final save with LoRA support
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            if self.use_lora:
                unwrapped_model.save_pretrained(final_checkpoint)
                logger.info(f"Training complete. Final LoRA adapters saved at {final_checkpoint}")
            else:
                state_dict = self.accelerator.get_state_dict(self.model)
                torch.save(state_dict, os.path.join(final_checkpoint, "pytorch_model.pt"))
                logger.info(f"Training complete. Final model saved at {final_checkpoint}")

        if self.accelerator.is_main_process:
            wandb.finish()

        self.accelerator.wait_for_everyone()

def main(cfg) -> None:
    logger.info("VLA Training :: Warming Up")

    #  Wrap config to enable access tracking
    cfg = wrap_config(cfg)
    logger.info("✅ Configuration wrapped for access tracking")

    # create output directory and save config
    output_dir = setup_directories(cfg=cfg)
    # build model
    vla = build_framework(cfg)
    # prepare data
    vla_train_dataloader = prepare_data(cfg=cfg, accelerator=accelerator, output_dir=output_dir)

    # set optimizer and scheduler
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)

    # create trainer
    # Run VLA Training
    trainer = VLATrainer(
        cfg=cfg,
        model=vla,
        vla_train_dataloader=vla_train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
    )

    # execute training preparation
    trainer.prepare_training()
    # execute training
    trainer.train()

    # And... we're done!
    logger.info("... and that's all, folks!")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="starVLA/config/training/starvla_cotrain_oxe.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    # Load YAML config & Convert CLI overrides to dotlist config
    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)  # Normalize CLI args to dotlist format
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    # if cfg.is_debug:
    if cfg.is_debug and dist.is_initialized() and dist.get_rank() == 0:
        import debugpy
        debugpy.listen(("0.0.0.0", 10092))
        print("🔍 Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    main(cfg)
