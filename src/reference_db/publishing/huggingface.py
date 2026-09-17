from __future__ import annotations

from pathlib import Path

from huggingface_hub import HfApi


def upload_files(
    repo_id: str,
    files: list[str | Path],
    path_prefix: str = "",
    commit_message: str = "Upload pipeline outputs",
) -> list[str]:
    api = HfApi()
    uploaded = []

    for file in files:
        path = Path(file)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        path_in_repo = f"{path_prefix.strip('/')}/{path.name}".lstrip("/")

        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=commit_message,
        )

        uploaded.append(path_in_repo)

    return uploaded