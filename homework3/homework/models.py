from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class Classifier(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 6,
    ):
        """
        A more complex convolutional network for image classification with additional layers and filters.
        """
        super().__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))

        # Increased number of filters in each layer
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1)  # 64x64 -> 32x32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # 32x32 -> 16x16
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # 16x16 -> 8x8
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # 8x8 -> 4x4
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)  # 4x4 -> 2x2

        # Batch Normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)

        # Global Average Pooling to reduce feature map size
        self.pool = nn.AdaptiveAvgPool2d(1)  # Output size will be (512, 1, 1)
        
        # Fully connected layer after pooling
        self.fc = nn.Linear(512, num_classes)  # 512 features after pooling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        """
        # Normalize input
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # Convolutional layers with Batch Normalization and ReLU activation
        x = F.relu(self.bn1(self.conv1(z)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))

        # Global average pooling
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten for the fully connected layer

        # Fully connected layer
        logits = self.fc(x)

        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Used for inference, returns class labels
        """
        return self(x).argmax(dim=1)



class Detector(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 3):
        super().__init__()

        # Downsampling with extra convolutional layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),  # 64x64 -> 32x32
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),  # Extra conv layer
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 32x32 -> 16x16
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  # Extra conv layer
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 16x16 -> 8x8
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # Extra conv layer
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 16x16 -> 8x8
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # Extra conv layer
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # Upsampling with extra convolutional layers
        self.upconv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 6x8 -> 12x16
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # Extra conv layer
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.upconv2 = nn.Sequential(
            nn.ConvTranspose2d(64 + 64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 12x16 -> 24x32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  # Extra conv layer
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.upconv3 = nn.Sequential(
            nn.ConvTranspose2d(32 + 32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # 24x32 -> 48x64
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),  # Extra conv layer
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        self.upconv4 = nn.Sequential(
            nn.ConvTranspose2d(16 + 16, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # 48x64 -> 96x128
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),  # Extra conv layer
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        # Classification head
        self.class_head = nn.Conv2d(19, num_classes, kernel_size=1)

        # Depth head with more convolutional layers
        self.depth_head = nn.Sequential(
            nn.Conv2d(19, 1, kernel_size=1),

        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        # Downsampling (Encoder)
        x1 = self.conv1(x)  # 48x64
        x2 = self.conv2(x1)  # 24x32
        x3 = self.conv3(x2)  # 12x16
        x4 = self.conv4(x3)  # 6x8

        # Upsampling (Decoder) with Skip Connections
        x5 = self.upconv1(x4)  # 12x16
        x5 = torch.cat([x5, x3], dim=1)  # Skip connection

        x6 = self.upconv2(x5)  # 24x32
        x6 = torch.cat([x6, x2], dim=1)  # Skip connection

        x7 = self.upconv3(x6)  # 48x64
        x7 = torch.cat([x7, x1], dim=1)  # Skip connection

        x8 = self.upconv4(x7)  # 96x128
        x8 = torch.cat([x8, x], dim=1)  # Skip connection with input

        # Output heads
        logits = self.class_head(x8)  # Segmentation output
        raw_depth = self.depth_head(x8).squeeze(1)  # Depth estimation output

        return logits, raw_depth




    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used for inference, takes an image and returns class labels and normalized depth.
        This is what the metrics use as input (this is what the grader will use!).

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            tuple of (torch.LongTensor, torch.FloatTensor):
                - pred: class labels {0, 1, 2} with shape (b, h, w)
                - depth: normalized depth [0, 1] with shape (b, h, w)
        """
        logits, raw_depth = self(x)
        
        # Get the predicted class labels by taking argmax on logits
        pred = logits.argmax(dim=1)  # (b, h, w)

        # Normalize depth to range [0, 1] if needed
        depth = raw_depth.squeeze(1)  # Remove the channel dimension, shape (b, h, w)
        depth = (depth - depth.min()) / (depth.max() - depth.min())  # Normalize to [0, 1]

        return pred, depth





# Use the tiny models instead of original ones
MODEL_FACTORY = {
    "classifier": Classifier,
    "detector": Detector,
   
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"
        print (model_path.resolve())
        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            print (e)
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Args:
        model: torch.nn.Module

    Returns:
        float, size in megabytes
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024