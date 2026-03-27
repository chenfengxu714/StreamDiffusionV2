"""Minimal dataset helpers used by inference entrypoints."""

from torch.utils.data import Dataset


class TextDataset(Dataset):
    """Load one prompt per line from a UTF-8 text file."""

    def __init__(self, data_path: str):
        with open(data_path, "r", encoding="utf-8") as handle:
            self.texts = [line.strip() for line in handle]

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> str:
        return self.texts[idx]
