import torchvision.models
import torch
from fastai import *
from fastai.vision import *

__all__ = ['SiameseNetwork', 'contrastive_loss']


class SiameseNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        # self.cnn1 = create_body(torchvision.models.densenet121(True))
        self.cnn1 = create_body(models.resnet50(pretrained=True))
        self.fc1 = nn.Sequential(
            nn.Linear(1000, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 5)
        )

    def forward_once(self, x):
        x = self.cnn1(x)
        x = F.relu(x)
        x = self.fc1(x)

        return x

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


def contrastive_loss(outputs, label,  margin=2.0):
    output1, output2 = outputs
    euclidean_distance = F.pairwise_distance(output1, output2)
    label = torch.from_numpy(label.cpu().data.numpy()).to(torch.float).cpu()

    loss_contrastive = torch.mean(
        (1-label) * torch.pow(euclidean_distance, 2).cpu()
        + (label) * torch.pow(torch.clamp(margin - euclidean_distance, min=0.0), 2).cpu()
    )
    return loss_contrastive
