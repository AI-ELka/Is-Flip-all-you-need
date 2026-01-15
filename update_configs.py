from pathlib import Path
import re
import argparse
import sys

# ---------- CLI ----------

def parse_args():
    parser = argparse.ArgumentParser(description="Safely update run indices in configs")

    parser.add_argument("--num_poisoned", type=int, required=True)
    parser.add_argument("--num_clean", type=int, required=True)
    parser.add_argument("--attack", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--aggregator", type=str, required=True)

    parser.add_argument("--reset", action="store_true", help="Reset run index to 1")

    return parser.parse_args()


# ---------- Utils ----------

RUN_REGEX = re.compile(r'/(\d+)(?:/[^"]*)?"?$')


def extract_run_from_path(path: str):
    m = RUN_REGEX.search(path)
    if not m:
        return None
    return int(m.group(1))


def replace_run_in_path(path: str, old: int, new: int):
    return re.sub(rf'/{old}(/|")', rf'/{new}\1', path)


# ---------- Train user ----------

def process_train_user_configs(base_dir):
    configs = list(base_dir.glob("train_user_*/config.toml"))
    if not configs:
        raise RuntimeError("No train_user configs found")

    run_ids = {}

    for cfg in configs:
        text = cfg.read_text()
        input_line = None
        output_line = None

        for line in text.splitlines():
            if line.strip().startswith("input_labels"):
                input_line = line
            elif line.strip().startswith("output_dir"):
                output_line = line

        if input_line is None or output_line is None:
            raise RuntimeError(f"Missing keys in {cfg}")

        run_in = extract_run_from_path(input_line)
        run_out = extract_run_from_path(output_line)

        if run_in is None or run_out is None:
            raise RuntimeError(f"Cannot parse run id in {cfg}")

        if run_in != run_out:
            raise RuntimeError(f"Run mismatch in {cfg}")

        run_ids[cfg] = run_in

    return run_ids


def update_train_user_configs(configs, old_run, new_run):
    for cfg in configs:
        text = cfg.read_text()
        new_lines = []
        replacements = 0

        for line in text.splitlines():
            if line.strip().startswith("input_labels") or line.strip().startswith("output_dir"):
                new_line, n = re.subn(
                    rf'/{old_run}(/|")',
                    rf'/{new_run}\1',
                    line
                )
                if n == 0:
                    raise RuntimeError(f"Failed replacement in {cfg}")
                replacements += n
                new_lines.append(new_line)
            else:
                new_lines.append(line)

        if replacements != 2:
            raise RuntimeError(f"Expected 2 replacements in {cfg}, got {replacements}")

        cfg.write_text("\n".join(new_lines) + "\n")
        print(f"[OK] {cfg.parent.name} (train_user)")


# ---------- Gen labels ----------

GEN_KEYS = {
    "federated_generate_labels": ["output_dir"],
    "federated_select_flips": ["input_label_glob", "true_labels", "output_dir"],
}


def process_gen_labels_config(gen_labels_path):
    if not gen_labels_path.exists():
        raise RuntimeError(f"Missing {gen_labels_path}")

    text = gen_labels_path.read_text()
    lines = text.splitlines()

    current_section = None
    found = []

    for line in lines:
        line_stripped = line.strip()

        if line_stripped.startswith("[") and line_stripped.endswith("]"):
            current_section = line_stripped.strip("[]")
            continue

        if current_section in GEN_KEYS:
            for key in GEN_KEYS[current_section]:
                if line_stripped.startswith(key):
                    run = extract_run_from_path(line)
                    if run is None:
                        raise RuntimeError(f"Cannot parse run in: {line}")
                    found.append(run)

    if not found:
        raise RuntimeError("No run ids found in gen_labels")

    if len(set(found)) != 1:
        raise RuntimeError("Inconsistent run ids in gen_labels")

    return found[0]


def update_gen_labels_config(gen_labels_path, old_run, new_run):
    text = gen_labels_path.read_text()
    lines = text.splitlines()

    current_section = None
    new_lines = []
    replacements = 0

    for line in lines:
        line_stripped = line.strip()

        if line_stripped.startswith("[") and line_stripped.endswith("]"):
            current_section = line_stripped.strip("[]")
            new_lines.append(line)
            continue

        if current_section in GEN_KEYS:
            for key in GEN_KEYS[current_section]:
                if line_stripped.startswith(key):
                    new_line, n = re.subn(
                        rf'/{old_run}(/|")',
                        rf'/{new_run}\1',
                        line
                    )
                    if n == 0:
                        raise RuntimeError(f"Failed replacement in gen_labels: {line}")
                    replacements += n
                    line = new_line

        new_lines.append(line)

    if replacements != 4:
        raise RuntimeError(f"Expected 4 replacements in gen_labels, got {replacements}")

    gen_labels_path.write_text("\n".join(new_lines) + "\n")
    print("[OK] gen_labels/config.toml")


# ---------- Main ----------

def main():
    args = parse_args()

    base_dir = Path(
        f"experiments/federated_experiments/"
        f"{args.num_poisoned}vs{args.num_clean}/"
        f"{args.dataset}/{args.attack}/{args.aggregator}"
    ).resolve()

    train_user_pattern = "train_user_*/config.toml"
    gen_labels_path = base_dir / "gen_labels" / "config.toml"

    print(f"📁 Base dir: {base_dir}")

    if not base_dir.exists():
        print("❌ Base directory does not exist")
        sys.exit(1)

    print("🔍 Scanning train_user configs...")
    train_user_runs = process_train_user_configs(base_dir)

    train_runs = set(train_user_runs.values())
    if len(train_runs) != 1:
        raise RuntimeError("Inconsistent runs in train_user")

    train_run = train_runs.pop()

    print("🔍 Scanning gen_labels config...")
    gen_run = process_gen_labels_config(gen_labels_path)

    if train_run != gen_run:
        raise RuntimeError(
            f"Mismatch: train_user run={train_run}, gen_labels run={gen_run}"
        )

    if args.reset:
        next_run = 1
        print("\n⚠️ RESET MODE: forcing run = 1\n")
    else:
        next_run = train_run + 1

    print(f"[INFO] Detected run: {train_run} → {next_run}\n")

    update_train_user_configs(train_user_runs.keys(), train_run, next_run)
    update_gen_labels_config(gen_labels_path, train_run, next_run)

    print("\n✅ All configs updated safely")


if __name__ == "__main__":
    main()
