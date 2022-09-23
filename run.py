import argparse
import random

import torch
import torch.nn as nn
import torch.optim as optim
from vision_transformer import Trainer, ViT, ViTData


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str)
    parser.add_argument(
        "--seed", default=0, type=int, help="seed for initializing training"
    )
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--shuffle", default=True, type=bool)
    parser.add_argument("--in-channels", type=int)
    parser.add_argument("--num-classes", type=int)
    parser.add_argument("--embed-dim", default=384, type=int)
    parser.add_argument("--patch-size", type=int)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--num-blocks", type=int)
    parser.add_argument("--heads", type=int)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--dropout", type=float)

    return parser.parse_args()


def main():
    args = make_parser()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = ViTData("./datasets")
    train_loader, valid_loader = data.create_CIFAR10_dataloader(
        batch_size=args.batch_size, shuffle=args.shuffle
    )

    model = ViT(
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        embed_dim=args.embed_dim,
        patch_size=args.patch_size,
        image_size=args.image_size,
        num_blocks=args.num_blocks,
        nb_head=args.heads,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0002)
    trainer = Trainer(
        50,
        train_loader=train_loader,
        valid_loader=valid_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        save_dir=args.save_dir,
    )
    trainer.fit(model)


if __name__ == "__main__":
    main()
