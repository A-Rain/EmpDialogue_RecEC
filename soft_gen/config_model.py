"""Configurations of Transformer model
"""
import copy

import texar.torch as tx

random_seed = 1024 # 123

beam_width = 5

length_penalty = 0.6
hidden_dim = 300
word_dim = 300
num_experts = 4
num_tasks = 2
dropout_rate = 0.1
use_mmoe=False

emb = {
    "name": "lookup_table",
    "dim": word_dim,
    "initializer": {
        "type": "normal_",
        "kwargs": {"mean": 0.0, "std": word_dim ** -0.5},
    },
}

position_embedder_hparams = {"dim": hidden_dim}

poswise_feedforward_hparams = {
        "layers": [
            {
                "type": "Linear",
                "kwargs": {
                    "in_features": hidden_dim,
                    "out_features": hidden_dim,
                    "bias": True,
                }
            },
            {
                "type": "Dropout",
                "kwargs": {
                    "p": 0.1,
                }
            },
            {
                "type": "Linear",
                "kwargs": {
                    "in_features": hidden_dim,
                    "out_features": hidden_dim,
                    "bias": True,
                }
            },

        ],
        "name": "ffn"
    }


emotion_encoder = {
    "dim": hidden_dim,
    "num_blocks": 1,
    "embedding_dropout": dropout_rate,
    "residual_dropout": dropout_rate,
    "multihead_attention": {
        "num_heads": 1,
        "output_dim": hidden_dim,
        'dropout_rate': dropout_rate,
        'use_bias': False,
        # See documentation for more optional hyperparameters
    },
    "initializer": {
        "type": "variance_scaling_initializer",
        "kwargs": {"factor": 1.0, "mode": "FAN_AVG", "uniform": False},
    },
    "poswise_feedforward": poswise_feedforward_hparams,
}

encoder = {
    "dim": hidden_dim,
    "num_blocks": 2,
    "multihead_attention": {
        "num_heads": 2,
        "output_dim": hidden_dim
        # See documentation for more optional hyperparameters
    },
    "initializer": {
        "type": "variance_scaling_initializer",
        "kwargs": {"factor": 1.0, "mode": "FAN_AVG", "uniform": True},
    },
    "poswise_feedforward": tx.modules.default_transformer_poswise_net_hparams(
        input_dim=hidden_dim,
        output_dim=hidden_dim
    ),
}

decoder = copy.deepcopy(encoder)

loss_label_confidence = 0.9

opt = {
    "optimizer": {
        "type": "Adam",
        "kwargs": {"beta1": 0.9, "beta2": 0.997, "epsilon": 1e-9},
    }
}

# lr_config = {
#     "learning_rate_schedule": "constant.linear_warmup.rsqrt_decay.rsqrt_depth",
#     "lr_constant": 2 * (hidden_dim ** -0.5),
#     "static_lr": 1e-4,
#     "warmup_steps": 100,
# }

lr_config = {
    "learning_rate_schedule": "static",
    "lr_constant": 2 * (hidden_dim ** -0.5),
    "static_lr": 5e-4,
    "warmup_steps": 100,
}