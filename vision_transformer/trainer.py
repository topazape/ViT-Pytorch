from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from vision_transformer.utils import AverageMeter, get_logger


class Trainer:
    def __init__(
        self,
        epochs: int,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        criterion,
        optimizer,
        device,
        save_dir,
    ):
        self.epochs = epochs
        self.train_loader, self.valid_loader = train_loader, valid_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.save_dir = save_dir

        self.logger = get_logger(str(Path(self.save_dir).joinpath("log.txt")))
        self.best_loss = float("inf")

    def fit(self, model: nn.Module):
        for epoch in range(self.epochs):
            model.train()
            losses = AverageMeter("train_loss")

            with tqdm(self.train_loader, dynamic_ncols=True) as pbar:
                pbar.set_description(f"[Epoch {epoch + 1}/{self.epochs}]")

                for tr_data in pbar:
                    tr_X = tr_data[0].to(self.device)
                    tr_y = tr_data[1].to(self.device)

                    self.optimizer.zero_grad()
                    out = model(tr_X)
                    # CrossEntropy
                    loss = self.criterion(out, tr_y)
                    loss.backward()
                    self.optimizer.step()

                    losses.update(loss.item())

                    pbar.set_postfix(loss=losses.value)

            self.logger.info(f"(train) epoch: {epoch} loss: {losses.avg}")
            self.evaluate(model, epoch)

    @torch.no_grad()
    def evaluate(self, model: nn.Module, epoch: Optional[int] = None) -> None:
        model.eval()
        losses = AverageMeter("valid_loss")

        for va_data in tqdm(self.valid_loader):
            va_X = va_data[0].to(self.device)
            va_y = va_data[1].to(self.device)

            out = model(va_X)
            loss = self.criterion(out, va_y)
            losses.update(loss.item())

        if epoch:
            self.logger.info(f"(valid) epoch: {epoch} loss: {losses.avg}")

            if losses.avg <= self.best_loss:
                self.best_acc = losses.avg
                torch.save(model.state_dict(), Path(self.save_dir).joinpath("best.pth"))
