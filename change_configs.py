import shutil
from pathlib import Path

def duplicate_cifar_to_svhn(experiments_dir: str):
    root = Path(experiments_dir)

    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")

    for src_dir in root.iterdir():
        # On ne traite que les dossiers contenant "cifar"
        if not src_dir.is_dir() or "cifar" not in src_dir.name:
            continue

        config_path = src_dir / "config.toml"
        if not config_path.exists():
            print(f"[SKIP] {src_dir.name} (no config.toml)")
            continue

        # Nouveau nom de dossier
        dst_dir = src_dir.with_name(src_dir.name.replace("cifar", "svhn"))

        if dst_dir.exists():
            print(f"[SKIP] {dst_dir.name} already exists")
            continue

        print(f"[COPY] {src_dir.name} → {dst_dir.name}")
        dst_dir.mkdir(parents=True)

        # Copier uniquement config.toml
        dst_config = dst_dir / "config.toml"
        shutil.copy2(config_path, dst_config)

        # Modifier le contenu du config.toml
        text = dst_config.read_text()
        text = text.replace("cifar", "svhn")
        dst_config.write_text(text)


if __name__ == "__main__":
    duplicate_cifar_to_svhn("./federated_experiments")
