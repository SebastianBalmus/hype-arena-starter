"""Build-time download of POTION weights for offline inference."""

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from model2vec import StaticModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default="main", help="Hub revision; pin the saved commit for reproducible builds")
    args = parser.parse_args()
    repo_id = "minishlab/potion-base-8M"
    target = Path(__file__).resolve().parents[1] / "models" / "potion-base-8M"
    revision = HfApi().model_info(repo_id, revision=args.revision).sha
    snapshot_download(
        repo_id, revision=revision, local_dir=target,
        allow_patterns=["model.safetensors", "config.json", "tokenizer.json", "README.md", "LICENSE*"],
    )
    # Confirm all files required by Model2Vec are present before recording success.
    model = StaticModel.from_pretrained(target)
    shape = model.encode(["monthly subscription price"]).shape
    files = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(target.iterdir()) if path.is_file() and path.name != "manifest.json"
    }
    (target / "manifest.json").write_text(
        json.dumps({"repo_id": repo_id, "revision": revision, "sha256": files}, indent=2) + "\n"
    )
    print(f"Saved {repo_id} at {revision} to {target}; embedding shape: {shape}")


if __name__ == "__main__":
    main()
