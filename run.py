import torch
import torch.nn as nn
import torch.optim as optim
from vision_transformer import ViT
from vision_transformer import ViTData, Trainer

def main():
    data = ViTData("./datasets")
    tr_l, te_l, classes = data.create_CIFAR10_dataloader(batch_size=32, shuffle=True)

    model = ViT()
    model.to("cuda")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0002)
    trainer = Trainer(50, train_loader=tr_l, valid_loader=te_l, criterion=criterion, optimizer=optimizer, device="cuda", save_dir=".")
    trainer.fit(model)

if __name__ == "__main__":
    main()
