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
        A tiny convolutional network for image classification.
        """
        super().__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))

        # Use even smaller architecture with depthwise separable convolutions
        self.conv1 = nn.Conv2d(in_channels, 4, kernel_size=3, stride=2, padding=1)  # 64x64 -> 32x32
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1)  # 32x32 -> 16x16
        self.conv3 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)  # 16x16 -> 8x8

        # Use global average pooling to reduce feature map size before fully connected layer
        self.pool = nn.AdaptiveAvgPool2d(1)  # Output size will be (16, 1, 1)
        self.fc = nn.Linear(16, num_classes)  # Only 16 features after pooling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, h, w) image

        Returns:
            tensor (b, num_classes) logits
        """
        # Normalize input
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]
        
        # Forward pass
        x = F.relu(self.conv1(z))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Apply global average pooling
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten for the fully connected layer
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

        # Downsampling
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1)  # 64x64 -> 32x32
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)  # 32x32 -> 16x16
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # 16x16 -> 8x8

        # Upsampling
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)  # 8x8 -> 16x16
        self.upconv2 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)  # 16x16 -> 32x32
        self.upconv3 = nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2, padding=1, output_padding=1)  # 32x32 -> 64x64

        # Classification head
        self.class_head = nn.Conv2d(16, num_classes, kernel_size=1)

        # Depth head
        self.depth_head = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Downsampling
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))

        # Upsampling
        x4 = F.relu(self.upconv1(x3))
        x5 = F.relu(self.upconv2(x4))
        x6 = F.relu(self.upconv3(x5))

        # Output heads
        logits = self.class_head(x6)
        raw_depth = self.depth_head(x6)

        return logits, raw_depth

    def predict(self, x):
        """Method required by the grader. Runs forward pass and returns predictions."""
        self.eval()  # Ensure the model is in evaluation mode
        with torch.no_grad():
            logits, raw_depth = self.forward(x)
            pred_classes = torch.argmax(logits, dim=1)  # Convert logits to class labels
        return pred_classes, raw_depth




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

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
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



