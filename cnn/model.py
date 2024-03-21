import torch
import torch.nn as nn


def init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear | nn.Conv2d):
        torch.nn.init.kaiming_normal_(module.weight, nonlinearity='relu')


class CNN(nn.Module):
    def __init__(self, dropout: float = None, skip_connections: bool = False) -> None:
        self.skip_connections_ = skip_connections
        super().__init__()

        self.input_conv = nn.Sequential(
            nn.Conv2d(3, 8, (7, 7), stride=2, padding=3, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )

        self.intermediate_conv = nn.ModuleList()
        for i in range(3, 6):
            in_channels = 2 ** i
            out_channels = 2 * in_channels

            self.intermediate_conv.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, (3, 3), stride=2, padding=1, bias=skip_connections),
                nn.Identity() if skip_connections else nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            ))

        self.conv = nn.ModuleList()
        for i in range(3, 7):
            channels = 2 ** i

            self.conv.append(nn.Sequential(
                nn.Conv2d(channels, channels, (3, 3), padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
                nn.Conv2d(channels, channels, (3, 3), padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
            ))

        self.intermediate_conv.append(nn.Sequential(
            nn.Conv2d(64, 128, (3, 3), stride=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        ))

        self.fc = nn.Sequential(
            nn.Flatten(),

            nn.Linear(128 * 3 * 3, 1024),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout else nn.Identity(),

            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout else nn.Identity(),

            nn.Linear(1024, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x = self.input_conv(x)

        for conv, int_conv in zip(self.conv, self.intermediate_conv):
            x = conv(x)
            res = x = int_conv(x + res if self.skip_connections_ else x)

        logits = self.fc(x)
        if self.training:
            return logits
        else:
            return torch.sigmoid(logits)
