import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class NVIDIA(nn.Module):
    def __init__(self, no_outputs, use_speed=False):
        super(NVIDIA, self).__init__()
        self.no_outputs = no_outputs
        self.use_speed = use_speed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # RGB img
        self.input_channels = 3
        self.features = nn.Sequential(
            nn.Conv2d(self.input_channels, 24, (5, 5), padding=2),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(24, 36, (5, 5), padding=2),
            nn.BatchNorm2d(36),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(36, 48, (5, 5), padding=2),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(48, 64, (3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 64, (3,3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * 4 * 8 + (1 if self.use_speed else 0), 64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, self.no_outputs)
        )


    def forward(self, data):
        B, _, H, W = data["img"].shape

        # meand and stdev for image
        mean_rgb = torch.tensor([0.47, 0.44, 0.45]).view(1, 3, 1, 1).to(self.device)
        std_rgb = torch.tensor([0.22, 0.22, 0.22]).view(1, 3, 1, 1).to(self.device)

        # make data unit normal
        img = data["img"]
        img = (img - mean_rgb) / std_rgb

        # feature extractor
        input = self.features(img)
        input = input.reshape(input.shape[0], -1)

        # append speed if necessary
        if self.use_speed:
            input = torch.cat((input, data["speed"]), dim=1)
        
        # probability distribution
        output = self.classifier(input)
        return output
