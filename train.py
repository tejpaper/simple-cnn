from cnn.constants import *
from cnn.model import *

import typing

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tqdm.autonotebook import tqdm, trange

# Bayesian optimization
from skopt import space, Optimizer

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

np.int = np.int_


def train_epoch(model: CNN,
                optimizer: optim.Optimizer,
                criterion: nn.Module,
                train_dl: DataLoader,
                device: torch.device) -> float:
    train_score = 0

    for samples, targets in tqdm(train_dl, desc='Batches', leave=False, position=2):
        samples = samples.to(device)
        targets = targets.unsqueeze(-1).type(torch.float32).to(device)

        model.zero_grad()

        logits = model(samples)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        train_score += ((torch.sigmoid(logits) > 0.5) == targets).sum()
    return train_score.item() / len(train_dl.dataset)


@torch.no_grad()
def val_epoch(model: nn.Module, val_dl: DataLoader, device: torch.device) -> float:
    val_score = 0

    for samples, targets in val_dl:
        samples = samples.to(device)
        targets = targets.unsqueeze(-1).type(torch.float32).to(device)

        val_score += ((model(samples) > 0.5) == targets).sum()
    return val_score.item() / len(val_dl.dataset)


def train(model: CNN,
          train_ds: Dataset,
          val_ds: Dataset,
          lr: float,
          swa_lr: float,
          batch_size: int,
          max_epochs: int,
          ) -> typing.Iterator[tuple[float, float] | tuple[nn.Module, float]]:

    is_cuda = next(model.parameters()).is_cuda
    device = torch.device('cuda' if is_cuda else 'cpu')

    train_dl = DataLoader(dataset=train_ds,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=2,
                          pin_memory=is_cuda)
    val_dl = DataLoader(dataset=val_ds,
                        batch_size=256,
                        num_workers=2,
                        pin_memory=is_cuda)

    optimizer = optim.Adam(model.parameters(), lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    swa_start = int(max_epochs * 0.5)
    swa_model = optim.swa_utils.AveragedModel(model)
    swa_scheduler = optim.swa_utils.SWALR(optimizer, swa_lr)

    criterion = nn.BCEWithLogitsLoss()

    for epoch in (pbar := trange(max_epochs, desc='Epochs', leave=False, position=1)):
        model.train()
        train_acc = train_epoch(model, optimizer, criterion, train_dl, device)

        model.eval()
        val_acc = val_epoch(model, val_dl, device)

        if epoch > swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step(val_acc)

        pbar.set_description(f'Epochs ({train_acc=:.4f}; {val_acc=:.4f})')
        yield train_acc, val_acc

    optim.swa_utils.update_bn(train_dl, swa_model, device)
    swa_model.eval()
    yield swa_model, val_epoch(swa_model, val_dl, device)


def learning_curve(logs: pd.DataFrame, max_epochs: int) -> plt.Axes:
    _, ax = plt.subplots(figsize=(8, 6))

    sns.lineplot(logs, x='epoch', y='train_acc', marker='o', label='Train', ax=ax)
    sns.lineplot(logs, x='epoch', y='val_acc', marker='o', label='Validation', ax=ax)

    # title and labels
    ax.set_title('Learning curves')
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Epoch')

    # x axis
    ax.grid(axis='x', color='black', alpha=.2, linewidth=.5)
    locator = ticker.FixedLocator(range(1, max_epochs + 1))
    ax.xaxis.set_major_locator(locator)
    ax.set_xlim((1, max_epochs))

    # y axis
    ax.grid(axis='y', color='black', alpha=.2, linewidth=.5)
    locator = ticker.FixedLocator(np.linspace(0, 1, 11))
    ax.yaxis.set_major_locator(locator)
    ax.set_ylim((0, 1))

    return ax


def hyperparameter_optimization(search_space: list,
                                n_iter: int,
                                max_epochs: int,
                                train_ds: Dataset,
                                val_ds: Dataset,
                                random_seed: int,
                                device: torch.device,
                                ) -> None:
    optimizer = Optimizer(search_space, random_state=random_seed, n_jobs=-1)

    for config_idx in trange(n_iter, desc='Configurations', position=0):
        params_list = optimizer.ask()

        dropout, skip_connections, lr, swa_lr, batch_size = params_list
        batch_size = batch_size.item()

        torch.manual_seed(random_seed)
        model = CNN(dropout, skip_connections).apply(init_weights).to(device)

        logs_dir = os.path.join(LOGS_DIR, f'{config_idx:03}')
        if not os.path.exists(logs_dir):
            os.mkdir(logs_dir)

        logs_file = os.path.join(logs_dir, 'epochs.csv')
        plot_file = os.path.join(logs_dir, 'learning curve.svg')
        logs = pd.DataFrame()

        model_file = os.path.join(logs_dir, 'model.pt')

        training_loop = train(model, train_ds, val_ds, lr, swa_lr, batch_size, max_epochs)
        for epoch in range(max_epochs):
            train_acc, val_acc = next(training_loop)

            logs = pd.concat([
                logs,
                pd.DataFrame([dict(
                    epoch=epoch + 1,
                    dropout=dropout,
                    skip_connections=skip_connections,
                    lr=lr,
                    swa_lr=swa_lr,
                    batch_size=batch_size,
                    train_acc=train_acc,
                    val_acc=val_acc,
                )])
            ], ignore_index=True)
            logs.to_csv(logs_file, index=False)

            ax = learning_curve(logs, max_epochs)
            ax.figure.savefig(plot_file)
            plt.close()

        swa_model, swa_val_acc = next(training_loop)
        logs['swa_val_acc'] = swa_val_acc
        logs.to_csv(logs_file, index=False)
        torch.save(swa_model.state_dict(), model_file)

        optimizer.tell(params_list, -swa_val_acc)


def main():
    plt.ioff()
    datasets = torch.load(DATA_INTERFACE_PATH)

    search_space = [
        # dropout
        space.Real(0.1, 0.9),
        # skip_connections
        [True, False],
        # lr
        space.Real(5e-6, 0.01, 'log-uniform'),
        # swa_lr
        space.Real(1e-4, 0.05),
        # batch_size
        [8, 16, 32, 64, 128],
    ]

    hyperparameter_optimization(
        search_space,
        25,
        12,
        datasets['train_ds'],
        datasets['val_ds'],
        RANDOM_SEED,
        DEVICE,
    )


if __name__ == '__main__':
    main()
