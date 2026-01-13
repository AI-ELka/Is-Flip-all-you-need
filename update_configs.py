from pathlib import Path
import re

NUM_POISONED = 1
NUM_CLEAN = 2
ATTACK = "backdoor"
DATASET = "cifar"

BASE_DIR = Path(f"experiments/federated_experiments/{NUM_POISONED}_vs_{NUM_CLEAN}/{ATTACK}/{DATASET}").resolve()


def extract_run_id(text: str):
    for line in text.splitlines():
        if "output_dir" not in line:
            continue
        m = re.search(r'/(\d+)/\d+\s*"?$', line)
        if m:
            return int(m.group(1))
    return None


def update_output_dir():
    configs = list(BASE_DIR.glob("train_user_*/config.toml"))

    if not configs:
        raise RuntimeError("None config.toml found")

    run_ids = {}

    for cfg in configs:
        text = cfg.read_text()
        run = extract_run_id(text)

        if run is None:
            raise RuntimeError(f"Impossible to detect output_dir in {cfg}")

        run_ids[cfg] = run

    unique_runs = set(run_ids.values())
    if len(unique_runs) != 1:
        raise RuntimeError(
            "Inconsistency detected in runs:\n"
            + "\n".join(f"{c.parent.name}: run {r}" for c, r in run_ids.items())
        )

    current_run = unique_runs.pop()
    next_run = current_run + 1

    print(f"[INFO] Detected run: {current_run} → {next_run}")

    # --- update
    for cfg in configs:
        text = cfg.read_text()

        new_text, n = re.subn(
            rf'/{current_run}/(\d+)',
            rf'/{next_run}/\1',
            text
        )

        if n != 1:
            raise RuntimeError(f"Unexpected replacement in {cfg}")

        cfg.write_text(new_text)
        print(f"[OK] {cfg.parent.name}")

    print("✅ All configs updated")


if __name__ == "__main__":
    update_output_dir()
