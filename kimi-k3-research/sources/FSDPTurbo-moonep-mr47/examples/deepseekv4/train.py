import argparse
import logging
import os

import torch

from fsdp_turbo.fsdp_turbo import FSDPTurbo
from fsdp_turbo.fsdp_turbo_config import load_config_from_yaml
from fsdp_turbo.training.trainer import BaseTrainer
from fsdp_turbo.utils.log import print_rank

logger = logging.getLogger(__name__)

def convert_model_dtype(model, dtype):
    for param in model.parameters():
        param.data = param.data.to(dtype)
    for buffer in model.buffers():
        buffer.data = buffer.data.to(dtype)

class DeepseekV4Trainer(BaseTrainer):

    def __init__(self, config):
        super().__init__(config)
        self.config = config

    def build_model(self):
        from transformers.models.deepseek_v4 import DeepseekV4ForCausalLM, DeepseekV4Config
        import modeling_deepseek_v4

        print_rank(logger.info, "> Building DeepseekV4 Model...")
        model_config = DeepseekV4Config()

        # reduce layer and expert for demo
        model_config.num_hidden_layers = 4
        model_config.n_routed_experts = 16
        model_config.index_topk = 16

        with torch.device('meta'):
            model = DeepseekV4ForCausalLM(model_config)
        convert_model_dtype(model, self.config.model.torch_dtype)

        print_rank(logger.info, "> Applying FSDPTurbo parallelism...")
        model = FSDPTurbo(self.config, model)
        model.to_empty(device='npu')

        # indexer grad bypass
        for name, module in model.named_modules():
            if "indexer" in name:
                for n, p in module.named_parameters():
                    p.grad = torch.zeros_like(p.data)
        return model

    def train_step(self, batch):
        batch['use_cache'] = False
        return super().train_step(batch)


def parse_args():
    parser = argparse.ArgumentParser(description="DeepSeekV4 Training")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config file")
    return parser.parse_args()


def main():
    args = parse_args()

    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.config)
    config = load_config_from_yaml(config_path)

    trainer = DeepseekV4Trainer(config=config)
    trainer.setup()
    trainer.train()


if __name__ == "__main__":
    main()
