import torchvision.models
import torch
from fastai import *
from fastai.vision import *

__all__ = ['SiameseNetwork', 'contrastive_loss']


class SiameseNetwork(nn.Module):

    def __init__(self, architecture: nn.Module = models.resnet50,
                 pretrained: bool = True,
                 cut: int = None,
                 nf: int = 1000,
                 nc: int = 2,
                 lin_ftrs: Optional[Collection[int]] = None,
                 ps: Floats = 0.5):
        super().__init__()
        self.body = create_body(architecture, pretrained=pretrained, cut=cut)
        self.head = create_head(nf, nc, lin_ftrs, ps)

    def forward_once(self, x):
        x = self.body(x)
        x = self.head(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


def contrastive_loss(outputs, label,  margin=2.0):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    output1, output2 = outputs
    euclidean_distance = F.pairwise_distance(output1, output2)
    label = torch.from_numpy(label.cpu().data.numpy()).to(torch.float).cpu()

    loss_contrastive = torch.mean(
        (1-label) * torch.pow(euclidean_distance, 2).cpu()
        + (label) * torch.pow(torch.clamp(margin -
                                          euclidean_distance, min=0.0), 2).cpu()
    )
    return loss_contrastive
