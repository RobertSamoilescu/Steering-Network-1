import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class WAYVE(nn.Module):
	def __init__(self, no_outputs, use_speed=False):
		super(WAYVE, self).__init__()
		self.no_outputs = no_outputs
		self.use_speed = use_speed
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		# RGB img
		self.input_channels = 3
		self.features = nn.Sequential(
			nn.Conv2d(self.input_channels, 32, (3, 3), stride=2),
			nn.ReLU(),

			nn.Conv2d(32, 32, (3, 3), stride=2),
			nn.ReLU(),

			nn.Conv2d(32, 32, (3, 3), stride=2),
			nn.ReLU(),

			nn.Conv2d(32, 32, (3, 3), stride=2),
			nn.ReLU(),
		)

		self.classifier = nn.Sequential(
			nn.Linear(672 + (1 if self.use_speed else 0), 256),
			nn.ReLU(),
			nn.Linear(256, self.no_outputs),
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
		# print(input.shape)

		# append speed if necessary
		if self.use_speed:
			input = torch.cat((input, data["speed"]), dim=1)
		
		# probability distribution
		output = self.classifier(input)
		return output
