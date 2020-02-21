experiment_name = 'densenet201-loss-7-2-1'

experiment_config = {
    "monitoring_params": {
        "name": experiment_name,
        "tags": ["pytorch", "catalyst", "torchvision", "densenet201"],
        "project": "bengali-ai"
    },
    "model_params": {
        "model_name": "densenet201",
        "output_classes": [
            7,
            168,
            11
        ],
        "pretrained": True
    },
    "runner_params": {},
    "args": {
        "logdir": "./logs",
        "verbose": True,
        "main_metric": "hmar_avg",
        "minimize_metric": False
    },
    "stages": {
        "data_params": {
            "batch_size": 128,
            "num_workers": 4
        },
        "state_params": {
            "num_epochs": 80,
            "checkpoint_data": {}
        },
        "criterion_params": {
            "criterion": "CrossEntropyLoss"
        },
        "scheduler_params": {
            "scheduler": "ReduceLROnPlateau",
            "mode": "min",
            "factor": 0.5,
            "patience": 2,
            "min_lr": 0.00001
        },
        "callbacks_params": {
            "loss_gr": {
                "callback": "CriterionCallback",
                "input_key": "grapheme_root",
                "output_key": "logit_grapheme_root",
                "prefix": "loss_gr",
                "multiplier": 0.7
            },
            "loss_vd": {
                "callback": "CriterionCallback",
                "input_key": "vowel_diacritic",
                "output_key": "logit_vowel_diacritic",
                "prefix": "loss_wd",
                "multiplier": 0.2
            },
            "loss_cd": {
                "callback": "CriterionCallback",
                "input_key": "consonant_diacritic",
                "output_key": "logit_consonant_diacritic",
                "prefix": "loss_cd",
                "multiplier": 0.1
            },
            "loss": {
                "callback": "CriterionAggregatorCallback",
                "prefix": "loss",
                "loss_aggregate_fn": "sum",
                "loss_keys": ["loss_gr", "loss_wd", "loss_cd"]
            },
            "early_stopping": {
                "callback": "EarlyStoppingCallback",
                "patience": 4,
                "metric": "hmar_avg",
                "minimize": False
            },
            "hmar_gr": {
                "callback": "HMacroAveragedRecall",
                "prefix": "hmar_gr",
                "input_key": "grapheme_root",
                "output_key": "logit_grapheme_root"
            },
            "hmar_wd": {
                "callback": "HMacroAveragedRecall",
                "prefix": "hmar_wd",
                "input_key": "vowel_diacritic",
                "output_key": "logit_vowel_diacritic"
            },
            "hmar_cd": {
                "callback": "HMacroAveragedRecall",
                "prefix": "hmar_cd",
                "input_key": "consonant_diacritic",
                "output_key": "logit_consonant_diacritic"
            },
            "hmar_avg": {
                "callback": "AverageMetric",
                "prefix": "hmar_avg",
                "metrics": [
                    "hmar_gr",
                    "hmar_wd",
                    "hmar_cd"
                ],
                "weights": [
                    2,
                    1,
                    1
                ]
            }
        },
        "stage1": {
            "optimizer_params": {
                "optimizer": "Adam",
                "lr": 0.001
            }
        }
    }
}
