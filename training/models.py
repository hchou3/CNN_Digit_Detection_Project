
import time
 
import torch
import torch.nn as nn
 
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
 
 
if __name__ == "__main__":
    model = ModelA()
    model.eval()
    print("parameters:", count_parameters(model))
 
    batch = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        out = model(batch)
    print("output shape:", tuple(out.shape))
 
    # Rough CPU speed: one forward pass on 1000 crops
    batch = torch.randn(1000, 3, 32, 32)
    with torch.no_grad():
        start = time.time()
        model(batch)
        seconds = time.time() - start
    print("CPU: {:.2f} s per 1000 crops".format(seconds))