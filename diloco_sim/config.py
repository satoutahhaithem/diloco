from dataclasses import dataclass, field
from typing import Optional, Callable, Type
import torch


@dataclass
class DilocoSimulatorConfig:
    model_cls: Type[torch.nn.Module]
    model_kwargs: dict
    loss_fn: Callable[..., torch.Tensor]
    train_dataset: torch.utils.data.Dataset
    optimizer_kwargs: dict
    optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.AdamW
    batch_size: int = 16
    eval_dataset: Optional[torch.utils.data.Dataset] = None
    ckpt_interval: Optional[int] = None  # num of outersteps to save model
    eval_iters: int = 400
    eval_interval: int = 1000
    save_dir: Optional[str] = None
    num_epochs: int = 1
    p_sparta: float = 0.0
    sparta_interval: int = 1
    cosine_anneal: bool = False
    warmup_steps: int = 0
    model_path: Optional[str] = None
    num_nodes: int = 4
    diloco_interval: int = 500
    outer_optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.SGD
    outer_optimizer_kwargs: dict = field(default_factory=lambda: {"lr": 0.7, "nesterov": True, "momentum": 0.9})
    max_local_step: Optional[int] = None
    wandb_project: Optional[str] = None
    max_minibatch_size: Optional[int] = None
    port: int = 12355
    devices: Optional[list[int]] = None
    max_norm: Optional[float] = None
    async_sparta_delay: int = 0