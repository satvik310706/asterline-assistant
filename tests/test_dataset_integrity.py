import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_immutable_dataset_hashes() -> None:
    expected = {
        "data/corpus.jsonl": "f63fc05cf43204485c98bd8fb6d89860a32c16a7e906bc90975e7c3954fe9fb1",
        "eval/datasets/research.jsonl": "67d7e6e0ed5541a5ee8295b53c4978017af21f6a8fd19822cd16bd1ad0e7f417",
        "eval/datasets/memory.jsonl": "09d32b417dacc408d443f1d0a79b972dde84aee3e0ed317b2776034798cca68a",
    }
    assert {name: sha256(ROOT / name) for name in expected} == expected
