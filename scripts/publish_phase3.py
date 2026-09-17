from pathlib import Path

from reference_db.publishing.huggingface import upload_files


OUTPUT_DIR = Path("phase_3")

FILES = [
    OUTPUT_DIR / "product_classification.parquet",
    OUTPUT_DIR / "product_taxonomy.parquet",
    OUTPUT_DIR / "product_groups.parquet",
    OUTPUT_DIR / "grouping_review_queue.parquet",
    OUTPUT_DIR / "grouping_contradictions.parquet",
]


def main() -> None:
    missing = [str(path) for path in FILES if not path.exists()]

    if missing:
        raise FileNotFoundError(f"Missing Phase 3 outputs: {missing}")

    print("Phase 3 outputs are ready for publishing:")
    for path in FILES:
        print(f"- {path}")


if __name__ == "__main__":
    main()
