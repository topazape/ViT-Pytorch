from dataclasses import dataclass
from pathlib import Path

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


@dataclass
class ViTData:
    save_dir: str

    def __post_init__(self) -> None:
        save_dir_path = Path(self.save_dir)
        if save_dir_path.exists():
            self.save_dir_path = save_dir_path
        else:
            raise FileNotFoundError

        self.transform = transforms.Compose(
            [
                # [0, 1]
                transforms.ToTensor(),
                # [0, 1] -> [-1, 1]
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def create_CIFAR10_dataloader(
        self, batch_size: int, shuffle: bool
    ) -> tuple[DataLoader, DataLoader]:
        save_dir = self.save_dir_path.joinpath("CIFAR10")
        save_dir = str(save_dir)

        train_set = torchvision.datasets.CIFAR10(
            root=save_dir, train=True, download=True, transform=self.transform
        )
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)

        test_set = torchvision.datasets.CIFAR10(
            root=save_dir, train=False, download=True, transform=self.transform
        )
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader
