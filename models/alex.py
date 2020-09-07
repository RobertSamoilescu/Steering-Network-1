import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models


class Alex(nn.Module):
	def __init__(self, no_outputs, use_speed=False):
		super(Alex, self).__init__()
		self.no_outputs = no_outputs
		self.use_speed = use_speed
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		self.alexnet = models.alexnet(pretrained=True)
		self.features = self.alexnet.features

		size = 4096
		self.classifier = nn.Sequential(
			nn.Dropout(0.5, inplace=False),
			nn.Linear(256 * 1 * 3 + (1 if self.use_speed else 0), size),
			nn.ReLU(inplace=True),
			nn.Dropout(0.5, inplace=False),
			nn.Linear(size, size),
			nn.ReLU(inplace=True),
			nn.Linear(size, no_outputs)
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
