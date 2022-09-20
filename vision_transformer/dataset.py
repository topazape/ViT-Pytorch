from pathlib import Path
from dataclasses import dataclass

from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms


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
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def create_CIFAR10_dataloader(
        self, batch_size: int, shuffle: bool
    ) -> tuple[DataLoader, DataLoader, tuple]:
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

        classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

        return train_loader, test_loader, classes
