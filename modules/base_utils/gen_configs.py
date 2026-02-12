#!/usr/bin/env python3
import os
from pathlib import Path

NUM_POISONED = 3
NUM_CLEAN = 7
ATTACK = "backdoor"
DATASETS = ["cifar", "svhn"]
POISONERS = ["1xs", "optimized", "4xl", "1xp"]
INIT="stripe"
MODEL_FLAG = "r32p"
AGGREGATORS = ["mean", "median", "krum", "trmean"]
BUDGETS = [150, 300, 500, 1000, 1500, 2000, 2500, 5000]
N_CYCLES = 10
GAMMA = 1.0

BASE_DIR = Path("experiments/federated_experiments").resolve()

GEN_LABEL_TEMPLATE = """# Module to train and record an expert trajectory.
[train_expert]
output_dir = "/Data/mb/flip/out/checkpoints/{model_flag}_{poisoner}/0/"
model = "{model_flag}"
trainer = "sgd"
dataset = "{dataset}"
source_label = 9
target_label = 4
poisoner = "{poisoner}"
delta = "optimized_trigger/fed_opt_trig_{init}_{model_flag}_{dataset}_{aggregator}_{num_poisoned}vs{num_clean}.pt"
epochs = 20
checkpoint_iters = 50

# Module to generate attack labels from the expert trajectories.
[federated_generate_labels]
input_pths = "/Data/mb/flip/out/checkpoints/{model_flag}_{poisoner}/{{}}/model_{{}}_{{}}.pth"
opt_pths = "/Data/mb/flip/out/checkpoints/{model_flag}_{poisoner}/{{}}/model_{{}}_{{}}_opt.pth"
expert_model = "{model_flag}"
trainer = "sgd"
dataset = "{dataset}"
source_label = 9
target_label = 4
poisoner = "{poisoner}"
delta = "optimized_trigger/fed_opt_trig_{init}_{model_flag}_{dataset}_{aggregator}_{num_poisoned}vs{num_clean}.pt"
output_dir = "out/{model_flag}/{num_poisoned}vs{num_clean}/{dataset}/{attack}/{aggregator}/{poisoner}/{run_id}/"
lambda = 0.0
num_honests = {num_clean}
num_poisoned = {num_poisoned}
agg_method = "{aggregator}"
attack = "{attack}"
gamma = {gamma}

[federated_generate_labels.expert_config]
experts = 1
min = 0
max = 20
trajectories = [50, 100, 150, 200]

[federated_generate_labels.attack_config]
iterations = 15
one_hot_temp = 5
alpha = 0
label_kwargs = {{lr = 150, momentum = 0.5}}

# Module to flip labels at the provided budgets.
[federated_select_flips]
budgets = {budgets}
input_label_glob = "out/{model_flag}/{num_poisoned}vs{num_clean}/{dataset}/{attack}/{aggregator}/{poisoner}/{run_id}/labels.npy"
true_labels = "out/{model_flag}/{num_poisoned}vs{num_clean}/{dataset}/{attack}/{aggregator}/{poisoner}/{run_id}/true.npy"
output_dir = "out/{model_flag}/{num_poisoned}vs{num_clean}/{dataset}/{attack}/{aggregator}/{poisoner}/{run_id}"
num_honests = {num_clean}
num_poisoned = {num_poisoned}
"""

TRAIN_USER_TEMPLATE = """[federated_train_user]
input_labels = "out/{model_flag}/{num_poisoned}vs{num_clean}/{dataset}/{attack}/{aggregator}/{poisoner}/{run_id}/"
budget = {budget}
user_model = "{model_flag}"
trainer = "sgd"
dataset = "{dataset}"
source_label = 9
target_label = 4
poisoner = "{poisoner}"
delta = "optimized_trigger/fed_opt_trig_{init}_{model_flag}_{dataset}_{aggregator}_{num_poisoned}vs{num_clean}.pt"
output_dir = "out/{model_flag}/{num_poisoned}vs{num_clean}/{dataset}/{attack}/{aggregator}/{poisoner}/{run_id}/{budget}"
soft = false
alpha = 0.0
num_honests = {num_clean}
num_poisoned = {num_poisoned}
agg_method = "{aggregator}"
"""

def write_config(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    print(f"[OK] Config written to {path}")

def generate_all_configs():
    for dataset in DATASETS:
        for poisoner in POISONERS:
            for aggregator in AGGREGATORS:
                # Gen_label configs
                for run_id in range(1, N_CYCLES + 1):
                    gen_label_dir = BASE_DIR / f"{MODEL_FLAG}/{NUM_POISONED}vs{NUM_CLEAN}/{dataset}/{ATTACK}/{aggregator}/{poisoner}/gen_labels/{run_id}"
                    gen_label_config = GEN_LABEL_TEMPLATE.format(
                        dataset=dataset,
                        model_flag=MODEL_FLAG,
                        num_poisoned=NUM_POISONED,
                        num_clean=NUM_CLEAN,
                        attack=ATTACK,
                        gamma=GAMMA,
                        poisoner=poisoner,
                        aggregator=aggregator,
                        run_id=run_id,
                        budgets=BUDGETS, 
                        init=INIT
                    )
                    write_config(gen_label_dir / "config.toml", gen_label_config)
                    for budget in BUDGETS:
                        train_user_dir = BASE_DIR / f"{MODEL_FLAG}/{NUM_POISONED}vs{NUM_CLEAN}/{dataset}/{ATTACK}/{aggregator}/{poisoner}/train_user_{budget}/{run_id}"
                        train_user_config = TRAIN_USER_TEMPLATE.format(
                            dataset=dataset,
                            model_flag=MODEL_FLAG,
                            num_poisoned=NUM_POISONED,
                            num_clean=NUM_CLEAN,
                            attack=ATTACK,
                            aggregator=aggregator,
                            poisoner=poisoner,
                            run_id=run_id,
                            budget=budget,
                            init=INIT
                        )
                        write_config(train_user_dir / "config.toml", train_user_config)

if __name__ == "__main__":
    generate_all_configs()
