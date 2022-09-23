from configparser import ConfigParser
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Cfg:
    config_file: str

    def __post_init__(self):
        if Path(self.config_file).exists():
            self.config = ConfigParser()
            with open(self.config_file, "r") as f:
                self.config.read_file(f)
        else:
            raise FileNotFoundError

    def get_params(self, type: str) -> dict[str, Any]:
        if type == "dataset":
            return self._dataset_params()
        elif type == "dataloader":
            return self._dataloader_params()
        elif type == "model":
            return self._model_params()
        elif type == "learning":
            return self._learning_params()
        else:
            raise KeyError

    def _dataset_params(self):
        return {
            "dir": self.config.get("dataset", "dir"),
            "name": self.config.get("dataset", "name"),
            "in_channels": self.config.getint("dataset", "in_channels"),
            "image_size": self.config.getint("dataset", "image_size"),
            "num_classes": self.config.getint("dataset", "num_classes"),
        }

    def _dataloader_params(self):
        return {
            "batch_size": self.config.getint("dataloader", "batch_size"),
            "shuffle": self.config.getboolean("dataloader", "shuffle"),
        }

    def _model_params(self):
        return {
            "patch_size": self.config.getint("model", "patch_size"),
            "embed_dim": self.config.getint("model", "embed_dim"),
            "num_blocks": self.config.getint("model", "num_blocks"),
            "heads": self.config.getint("model", "heads"),
            "hidden_dim": self.config.getint("model", "hidden_dim"),
            "dropout": self.config.getfloat("model", "dropout"),
        }

    def _learning_params(self):
        return {
            "epochs": self.config.getint("learning", "epochs"),
            "learning_rate": self.config.getfloat("learning", "learning_rate"),
        }
