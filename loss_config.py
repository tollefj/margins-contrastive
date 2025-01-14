from sentence_transformers.losses import (
    ContrastiveLoss,
    MultipleNegativesRankingLoss,
    OnlineContrastiveLoss,
    TripletLoss,
)

loss_config_all = {
    "Triplet": {"triplet_margin": [0.01, 0.1, 1, 5, 7.5, 10], "lossfn": TripletLoss},
    "Contrastive": {"margin": [0.1, 0.25, 0.5, 0.75, 1], "lossfn": ContrastiveLoss},
    "OnlineContrastive": {
        "margin": [0.1, 0.25, 0.5, 0.75, 1],
        "lossfn": OnlineContrastiveLoss,
    },
    "MultipleNegatives": {"lossfn": MultipleNegativesRankingLoss},
}

loss_config_default = {
    "Triplet": {"lossfn": TripletLoss},
    "Contrastive": {"lossfn": ContrastiveLoss},
    "OnlineContrastive": {"lossfn": OnlineContrastiveLoss},
    "MultipleNegatives": {"lossfn": MultipleNegativesRankingLoss},
}
loss_config_best = {"Triplet": {"lossfn": TripletLoss, "triplet_margin": [0.01]}}

available_configs = {
    "default": loss_config_default,
    "all": loss_config_all,
    "best": loss_config_best,
}


def get_available_loss_configs() -> list:
    return list(available_configs.keys())


def get_preset_loss_config(preset: str) -> dict:
    return available_configs[preset]


def get_loss_config(default_lambda: bool = False) -> dict:
    return loss_config_default if default_lambda else loss_config_all
