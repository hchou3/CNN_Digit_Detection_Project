
import time
 
import torch
import torch.nn as nn
from torchvision.models import vgg16
 
NUM_CLASSES = 11  # digits 0-9 and background (10)
 
def conv_block(in_channels, out_channels):
    """Two 3x3 convolutions with BatchNorm and ReLU, then 2x2 max pooling.
 
    Args:
        in_channels (int): channels coming in.
        out_channels (int): channels going out.
 
    Returns:
        nn.Sequential: the block. Halves the height and width.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )
 
 
class ModelA(nn.Module):
    """Small CNN for 32x32 crops, trained from scratch.
 
    32x32 -> block 1 (32 filters) -> 16x16 -> block 2 (64) -> 8x8
          -> block 3 (128) -> 4x4 -> global average pool -> dropout -> 11 classes
 
    Args:
        widths (tuple): filters in each of the three blocks.
        dropout (float): dropout probability before the last layer.
    """
 
    def __init__(self, widths=(32, 64, 128), dropout=0.3):
        super(ModelA, self).__init__()

        self.features = nn.Sequential(
            conv_block(3, widths[0]),
            conv_block(widths[0], widths[1]),
            conv_block(widths[1], widths[2]),
        )
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(widths[2], NUM_CLASSES)
 
    def forward(self, x):
        x = self.features(x)        # (N, 128, 4, 4)
        x = self.pool(x)            # (N, 128, 1, 1)
        x = torch.flatten(x, 1)     # (N, 128)
        x = self.dropout(x)
        return self.classifier(x)   # (N, 11) class scores
 
 
def count_parameters(model):
    """Number of trainable weights in a model.
 
    Args:
        model (nn.Module): the model.
 
    Returns:
        int: number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_vgg16(pretrained):
    """Builds torchvision's VGG16, with or without ImageNet weights.

    Without pretrained weights nothing is downloaded, so run.py can build the
    model and then load our own trained weights from a file.

    Args:
        pretrained (bool): load ImageNet weights (needs internet the first time).

    Returns:
        nn.Module: torchvision VGG16.
    """
    if not pretrained:
        return vgg16()

    return vgg16(pretrained=True)


class ModelB(nn.Module):
    """VGG16 convolution layers with a new small head for 64x64 crops.

    64x64 -> VGG16 features (13 conv layers, 5 pools) -> 2x2x512
          -> global average pool -> 512 -> 256 -> dropout -> 11 classes

    Args:
        pretrained (bool): start from ImageNet weights (True for training).
        hidden (int): units in the hidden layer of the head.
        dropout (float): dropout probability before the last layer.
    """

    def __init__(self, pretrained=False, hidden=256, dropout=0.5):
        super(ModelB, self).__init__()
        vgg = load_vgg16(pretrained)
        self.features = vgg.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(512, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, NUM_CLASSES),
        )

    def forward(self, x):
        x = self.features(x)        # (N, 512, 2, 2) for 64x64 input
        x = self.pool(x)            # (N, 512, 1, 1)
        x = torch.flatten(x, 1)     # (N, 512)
        return self.head(x)         # (N, 11) class scores

    def freeze_features(self):
        """Stage 1: keep all VGG16 convolution weights fixed, train only the head."""
        for p in self.features.parameters():
            p.requires_grad = False

    def unfreeze_last_block(self):
        """Stage 2: also train the last convolution block (layers 24-30 of features)."""
        for p in self.features[24:].parameters():
            p.requires_grad = True


def check_model(model, size):
    model.eval()
    print("parameters:", count_parameters(model))

    with torch.no_grad():
        out = model(torch.randn(4, 3, size, size))
    print("output shape:", tuple(out.shape))

    batch = torch.randn(200, 3, size, size)
    with torch.no_grad():
        start = time.time()
        model(batch)
        seconds = (time.time() - start) * 5
    print("CPU: {:.2f} s per 1000 crops".format(seconds))

 
if __name__ == "__main__":
    print("Model A")
    check_model(ModelA(), 32)

    print("Model B")
    model_b = ModelB(pretrained=False)
    check_model(model_b, 64)
    model_b.freeze_features()
    print("trainable after freeze_features:", count_parameters(model_b))
    model_b.unfreeze_last_block()
    print("trainable after unfreeze_last_block:", count_parameters(model_b))