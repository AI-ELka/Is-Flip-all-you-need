from pathlib import Path
import re


BASE_DIR = Path("experiments/federated_experiments")


def extract_run_id(text: str):
    for line in text.splitlines():
        if "output_dir" not in line:
            continue
        m = re.search(r'/(\d+)/\d+\s*"?$', line)
        if m:
            return int(m.group(1))
    return None


def update_output_dir():
    configs = list(BASE_DIR.glob("train_cifar_backdoor_*/config.toml"))

    if not configs:
        raise RuntimeError("Aucun config.toml trouvé")

    run_ids = {}

    # --- détection du run courant
    for cfg in configs:
        text = cfg.read_text()
        run = extract_run_id(text)

        if run is None:
            raise RuntimeError(f"Impossible de détecter output_dir dans {cfg}")

        run_ids[cfg] = run

    unique_runs = set(run_ids.values())
    if len(unique_runs) != 1:
        raise RuntimeError(
            "Incohérence des runs détectés:\n"
            + "\n".join(f"{c.parent.name}: run {r}" for c, r in run_ids.items())
        )

    current_run = unique_runs.pop()
    next_run = current_run + 1

    print(f"[INFO] Run détecté : {current_run} → {next_run}")

    # --- mise à jour
    for cfg in configs:
        text = cfg.read_text()

        new_text, n = re.subn(
            rf'/{current_run}/(\d+)',
            rf'/{next_run}/\1',
            text
        )

        if n != 1:
            raise RuntimeError(f"Remplacement inattendu dans {cfg}")

        cfg.write_text(new_text)
        print(f"[OK] {cfg.parent.name}")

    print("✅ Tous les configs mis à jour")


if __name__ == "__main__":
    update_output_dir()
