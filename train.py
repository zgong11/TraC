import argparse
import os

from research.utils.config import Config


def try_wandb_setup(path, config):
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if wandb_api_key is not None and wandb_api_key != "":
        try:
            import wandb
        except ImportError:
            return
        # project_dir = os.path.dirname(os.path.dirname(__file__))
        project_dir = os.path.dirname(__file__)
        group = "-".join(path.split("/")[-3:-2])
        name = path.split("/")[-1]
        wandb.init(
            project=os.path.basename(project_dir),
            # name=os.path.basename(path),
            name=name,
            config=config.flatten(separator="-"),
            # dir=os.path.join(os.path.dirname(project_dir), "wandb"),
            dir=os.path.join(project_dir, "exp"),
            group=group,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default=None)
    parser.add_argument("--path", "-p", type=str, default=None)
    parser.add_argument("--device", "-d", type=str, default="auto")
    parser.add_argument("--seed", "-s", type=int, default=0)
    parser.add_argument("--eval_env", "-e", type=str, default=None)
    parser.add_argument("--dataset_path", "-dp", type=str, default=None)
    parser.add_argument("--target_cost", "-tc", type=float, default=None)
    parser.add_argument("--alpha", "-a", type=float, default=None)
    parser.add_argument("--gamma", "-g", type=float, default=None)
    parser.add_argument("--eta", type=float, default=None)
    parser.add_argument("--seg_ratio", type=float, default=None)
    parser.add_argument("--kappa", type=float, default=None)
    parser.add_argument("--safe_top_perc", type=float, default=None)
    parser.add_argument("--safe_bottom_perc", type=float, default=None)
    args = parser.parse_args()

    config = Config.load(args.config)
    config["seed"] = args.seed
    config["eval_env"] = args.eval_env
    config["dataset_kwargs"]["path"] = args.dataset_path
    config["trainer_kwargs"]["target_cost"] = args.target_cost  # BulletGym: [10, 20, 40]; SafetyGym: [20, 40, 80]
    config["alg_kwargs"]["alpha"] = args.alpha
    config["alg_kwargs"]["gamma"] = args.gamma
    config["alg_kwargs"]["eta"] = args.eta
    config["dataset_kwargs"]["seg_ratio"] = args.seg_ratio
    config["dataset_kwargs"]["kappa"] = args.kappa
    config["dataset_kwargs"]["safe_top_perc"] = args.safe_top_perc
    config["dataset_kwargs"]["safe_bottom_perc"] = args.safe_bottom_perc

    log_path = os.path.join(args.path, config["eval_env"], config["dataset_kwargs"]["path"].split("/")[-1][:-5], config["alg"], 
                            f"cost-{str(config['trainer_kwargs']['target_cost'])}", f"seed-{str(config['seed'])}-seg")
    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)  # Change this to false temporarily so we don't recreate experiments
    # try_wandb_setup(log_path, config)
    config.save(log_path)  # Save the config

    # Parse the config file to resolve names.
    config = config.parse()
    trainer = config.get_trainer(device=args.device)
    # Train the model
    trainer.train(log_path)
