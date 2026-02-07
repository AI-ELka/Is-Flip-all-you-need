#!/usr/bin/env python3
import os
from pathlib import Path

# ==========================
# Paramètres à configurer
# ==========================
NUM_POISONED = 3
NUM_CLEAN = 7
ATTACK = "stealthy_backdoor"
DATASET = "cifar"
AGGREGATORS = ["mean", "median", "krum", "trmean"]
BUDGETS = [150, 300, 500, 1000, 1500, 2000, 2500, 5000]
N_CYCLES = 5
GAMMA = 0.5  

BASE_DIR = Path("experiments/federated_experiments").resolve()


# ==========================
# Templates de config
# ==========================
GEN_LABEL_TEMPLATE = """# Module to train and record an expert trajectory.
[train_expert]
output_dir = "/Data/mb/flip/out/checkpoints/r32p_1xs/0/"
model = "r32p"
trainer = "sgd"
dataset = "{dataset}"
source_label = 9
target_label = 4
poisoner = "1xs"
epochs = 20
checkpoint_iters = 50

# Module to generate attack labels from the expert trajectories.
[federated_generate_labels]
input_pths = "/Data/mb/flip/out/checkpoints/r32p_1xs/{{}}/model_{{}}_{{}}.pth"
opt_pths = "/Data/mb/flip/out/checkpoints/r32p_1xs/{{}}/model_{{}}_{{}}_opt.pth"
expert_model = "r32p"
trainer = "sgd"
dataset = "{dataset}"
source_label = 9
target_label = 4
poisoner = "1xs"
output_dir = "out/{num_poisoned}vs{num_clean}/{dataset}/{attack}/{aggregator}/{run_id}/"
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
input_label_glob = "out/{num_poisoned}vs{num_clean}/{dataset}/{attack}/{aggregator}/{run_id}/labels.npy"
true_labels = "out/{num_poisoned}vs{num_clean}/{dataset}/{attack}/{aggregator}/{run_id}/true.npy"
output_dir = "out/{num_poisoned}vs{num_clean}/{dataset}/{attack}/{aggregator}/{run_id}"
num_honests = {num_clean}
num_poisoned = {num_poisoned}
"""

TRAIN_USER_TEMPLATE = """[federated_train_user]
input_labels = "out/{num_poisoned}vs{num_clean}/{dataset}/{attack}/{aggregator}/{run_id}/"
budget = {budget}
user_model = "r32p"
trainer = "sgd"
dataset = "{dataset}"
source_label = 9
target_label = 4
poisoner = "1xs"
output_dir = "out/{num_poisoned}vs{num_clean}/{dataset}/{attack}/{aggregator}/{run_id}/{budget}"
soft = false
alpha = 0.0
num_honests = {num_clean}
num_poisoned = {num_poisoned}
agg_method = "{aggregator}"
"""

# ==========================
# Fonction pour créer les configs
# ==========================
def write_config(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    print(f"[OK] Config written to {path}")


def generate_all_configs():
    for aggregator in AGGREGATORS:
        # Gen_label configs
        for run_id in range(1, N_CYCLES + 1):
            gen_label_dir = BASE_DIR / f"{NUM_POISONED}vs{NUM_CLEAN}/{DATASET}/{ATTACK}/{aggregator}/gen_labels/{run_id}"
            gen_label_config = GEN_LABEL_TEMPLATE.format(
                dataset=DATASET,
                num_poisoned=NUM_POISONED,
                num_clean=NUM_CLEAN,
                attack=ATTACK,
                gamma=GAMMA,
                aggregator=aggregator,
                run_id=run_id,
                budgets=BUDGETS
            )
            write_config(gen_label_dir / "config.toml", gen_label_config)

        # Train_user configs
        for budget in BUDGETS:
            for run_id in range(1, N_CYCLES + 1):
                train_user_dir = BASE_DIR / f"{NUM_POISONED}vs{NUM_CLEAN}/{DATASET}/{ATTACK}/{aggregator}/train_user_{budget}/{run_id}"
                train_user_config = TRAIN_USER_TEMPLATE.format(
                    dataset=DATASET,
                    num_poisoned=NUM_POISONED,
                    num_clean=NUM_CLEAN,
                    attack=ATTACK,
                    aggregator=aggregator,
                    run_id=run_id,
                    budget=budget
                )
                write_config(train_user_dir / "config.toml", train_user_config)


if __name__ == "__main__":
    generate_all_configs()
