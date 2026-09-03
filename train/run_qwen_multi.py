import argparse
import os
import yaml

from src.train import SelfTrainPipeline, ensure_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    exp_dir = os.path.join(cfg["yaml_file"], cfg["exp_name"])
    save_path = os.path.join(exp_dir, "config.yaml")
    ensure_dir(os.path.dirname(save_path))

    with open(save_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    pipe = SelfTrainPipeline(cfg)
    pipe.run()


if __name__ == "__main__":
    main()


