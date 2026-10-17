#!/usr/bin/env python3
"""Export an N2 DWL checkpoint as a stateless ONNX LSTM step.

The controller owns h/c state and passes them through each policy invocation.
"""
import argparse
import json
from pathlib import Path

import torch
from torch import nn


class PolicyStep(nn.Module):
    def __init__(self, cfg, weights, normalizer):
        super().__init__()
        policy = cfg["policy"]
        env = cfg["env"]
        if (env["num_observations"], env["num_actions"], env["frame_stack"]) != (39, 10, None):
            raise ValueError("Expected a 39-observation, 10-action policy without frame stacking")
        if policy["rnn_type"].lower() != "lstm":
            raise ValueError("Expected an LSTM policy")
        activation = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh}[policy["activation"].lower()]
        hidden = policy.get("rnn_hidden_dim", policy.get("rnn_hidden_size", 256))
        embedding = policy["num_embedding"]
        self.rnn = nn.LSTM(39, hidden, num_layers=policy["rnn_num_layers"])

        def mlp(input_dim, middle, output_dim):
            layers = []
            for width in middle:
                layers.extend((nn.Linear(input_dim, width), activation()))
                input_dim = width
            layers.append(nn.Linear(input_dim, output_dim))
            return nn.Sequential(*layers)

        self.embodied = mlp(hidden, policy["enc_hidden_dims"], embedding)
        self.actor = mlp(embedding, policy["actor_hidden_dims"], 10)
        for module, prefix in ((self.rnn, "dwl_model.encoder.rnn."),
                               (self.embodied, "embodied."), (self.actor, "actor.")):
            module.load_state_dict({key[len(prefix):]: value for key, value in weights.items()
                                    if key.startswith(prefix)}, strict=True)
        if normalizer is None:
            self.register_buffer("mean", torch.zeros(39))
            self.register_buffer("std", torch.ones(39))
            self.normalize = False
        else:
            self.register_buffer("mean", normalizer["_mean"].reshape(39))
            self.register_buffer("std", normalizer["_std"].reshape(39))
            self.normalize = True

    def forward(self, obs, h_in, c_in):
        if self.normalize:
            obs = (obs - self.mean) / (self.std + 1e-2)
        encoded, (h_out, c_out) = self.rnn(obs.unsqueeze(0), (h_in, c_in))
        actions = self.actor(self.embodied(encoded.squeeze(0)))
        return actions, h_out, c_out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="model_*.pt from n2_dwl training")
    parser.add_argument("--config", type=Path, help="train_cfg.json; defaults to checkpoint directory")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "policy/policy_user.onnx")
    args = parser.parse_args()
    config_path = args.config or args.checkpoint.parent / "train_cfg.json"
    with config_path.open(encoding="utf-8") as stream:
        cfg = json.load(stream)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise ValueError("Expected a DWL model_*.pt training checkpoint")
    normalizer = None
    if cfg["runner"]["empirical_normalization"]:
        normalizer = checkpoint["obs_norm_state_dict"]
    model = PolicyStep(cfg, checkpoint["model_state_dict"], normalizer).eval()
    hidden = model.rnn.hidden_size
    layers = model.rnn.num_layers
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(model,
                          (torch.zeros(1, 39), torch.zeros(layers, 1, hidden),
                           torch.zeros(layers, 1, hidden)),
                          str(args.output), opset_version=11, do_constant_folding=True,
                          input_names=["policy_input", "h_in", "c_in"],
                          output_names=["policy_output", "h_out", "c_out"])
    import onnx
    graph = onnx.load(str(args.output))
    onnx.checker.check_model(graph)
    print(f"Exported {args.output} from {args.checkpoint} (iteration {checkpoint.get('iter', '?')})")
    print(f"Input: [1,39], [{layers},1,{hidden}] x2; output: [1,10], [{layers},1,{hidden}] x2")


if __name__ == "__main__":
    main()
