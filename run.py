import argparse
from typing import Dict

import wandb

from loss_config import get_available_loss_configs, get_preset_loss_config
from model_config import (
    ModelConfig,
    ParameterConfig,
    generate_model_configs,
    get_parameters,
)
from model_trainer import train_model
from util import (
    get_datasets,
    get_loss_data,
    get_models,
    get_original_datasets,
    get_train_test,
    make_wandb_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run model training")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        help=f"Any sentence-transformer model (from Hugging Face hub). Tested models: {get_models()}",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        help="Choose an existing dataset",
        choices=get_datasets(),
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="all",
        help="A predefined loss configuration",
        choices=get_available_loss_configs(),
    )
    parser.add_argument(
        "--samples",
        type=int,
        nargs="+",
        default=[50000],
        help="The number of training samples",
    )
    parser.add_argument(
        "--project", type=str, default="tmp", help="The name of the wandb project"
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Boolean variable to decide whether to upload the project. Requires --user argument.",
    )
    parser.add_argument(
        "--user",
        type=str,
        help="Huggingface user to publish models. Log in with `huggingface-cli login`",
    )
    print("See `params.yml` for additional training parameters")
    return parser.parse_args()


def main():
    args = parse_args()

    ################## DATA ##################
    loss_datas = get_loss_data()
    original_datasets = get_original_datasets()
    ################# SETUP ##################
    # wandb.login()

    datasets = get_datasets()
    if args.dataset != "all":
        datasets = [args.dataset]

    models = get_models()
    if args.model != "all":
        if args.model not in models:
            print(
                f"Model {args.model} is not in the tested models. Training will however proceed as normal with your selection."
            )
            model_name = args.model
            if "/" in args.model:
                model_name = args.model.split("/")[-1]
            models = {model_name: args.model}
        else:
            models = {args.model: models[args.model]}

    ################# CONFIG #################
    loss_config = get_preset_loss_config(args.loss)
    model_configs: Dict[str, ModelConfig] = generate_model_configs(models, loss_config)
    params: ParameterConfig = get_parameters()

    ################ TRAINING ################
    print(f"Training on {args.samples} samples")

    for dataset in datasets:
        loss_data = loss_datas[dataset]
        print(f"Training on {dataset}")
        train_data, test_data = get_train_test(original_datasets, dataset)

        for samples in args.samples:
            for m_name, m_config in model_configs.items():
                wandb_config = make_wandb_config(
                    m_name, m_config, params, dataset, samples
                )
                print(wandb_config["model"])
                with wandb.init(project=args.project, config=wandb_config, name=m_name):
                    ######## MAIN TRAINING STEP ########
                    model, all_scores = train_model(
                        m_name,
                        m_config,
                        params,
                        loss_data,
                        train_data,
                        test_data,
                        samples,
                    )
                    sample_str = str(samples)
                    if "000" in sample_str:
                        sample_str = sample_str[:2] + "k"
                    model_id = f"{wandb_config['model']}-{dataset}-{sample_str}-{wandb_config['loss']}"
                    print(f"Publishing model: {model_id}")
                    model.save(path=f"finetuned_model_store/{model_id}")

                    ############# LOGGING ##############
                    for ith_epoch, scores in enumerate(all_scores):
                        scores["epoch"] = ith_epoch
                        wandb.log(scores)


if __name__ == "__main__":
    main()
