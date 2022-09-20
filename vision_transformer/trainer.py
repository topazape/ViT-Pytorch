from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


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
        self.criterion, self.optimizer = criterion, optimizer
        self.device = device
        self.save_dir = save_dir

        # self.logger = get_logger(str(Path(self.save_dir).joinpath("log.txt")))
        self.best_loss = float("inf")


    def fit(self, model: nn.Module):
        for epoch in range(self.epochs):
            model.train()
            # losses = AverageMeter("train_loss")

            with tqdm(self.train_loader, dynamic_ncols=True) as pbar:
                pbar.set_description(f"[Epoch {epoch + 1}/{self.epochs}]")

                for tr_data in pbar:
                    tr_X = [d.to(self.device) for d in tr_data[:-1]]
                    tr_y = tr_data[-1].to(self.device)

                    self.optimizer.zero_grad()
                    out = model(*tr_X)
                    # RMSE
                    loss = self.criterion(out, tr_y).sqrt()
                    loss.backward()
                    self.optimizer.step()

                    # losses.update(rmse)

                    # pbar.set_postfix(loss=losses.value)

            self.evaluate(model, epoch)

    @torch.no_grad()
    def evaluate(self, model: nn.Module, epoch: Optional[int] = None) -> None:
        model.eval()
        # losses = AverageMeter("valid_loss")

        for va_data in tqdm(self.valid_loader):
            va_X = [d.to(self.device) for d in va_data[:-1]]
            va_y = va_data[-1].to(self.device)

            out = model(*va_X)

            # losses.update(rmse)

        # self.logger.info(f"loss: {losses.avg}")

        # if epoch is not None:
        #     if losses.avg <= self.best_loss:
        #         self.best_acc = losses.avg
        #         torch.save(model.state_dict(), Path(self.save_dir).joinpath("best.pth"))
