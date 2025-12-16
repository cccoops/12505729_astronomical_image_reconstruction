import pytest
import torch
from torchvision import transforms
from PIL import Image
import torch
from PIL import Image
import torch.nn as nn
from tqdm import tqdm
from torchvision import transforms


class DoubleConv(nn.Module):
    """(Conv2d -> ReLU) x2"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


#TODO: Be able to reason about the network here
class CnnUNet(nn.Module):
    def __init__(self):
        super(CnnUNet, self).__init__()

        # Encoder (Downsampling)
        self.enc1 = DoubleConv(3, 64)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(256, 512)

        # Decoder (Upsampling + SKIP CONNECTIONS)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256) # Input is 512 because 256 (up3) + 256 (skip)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128) # Input is 256 because 128 (up2) + 128 (skip)
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)  # Input is 128 because 64 (up1) + 64 (skip)

        self.final_conv = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        # Bottleneck
        b = self.bottleneck(p3)

        # Decoder with Skips
        # We concatenate the encoder output (e3) with the upsampled bottleneck (x)
        x = self.up3(b)
        x = torch.cat([x, e3], dim=1) 
        x = self.dec3(x)

        x = self.up2(x)
        x = torch.cat([x, e2], dim=1)
        x = self.dec2(x)

        x = self.up1(x)
        x = torch.cat([x, e1], dim=1)
        x = self.dec1(x)

        return self.final_conv(x)


# Helper to create a fake image if one doesn't exist
@pytest.fixture
def fake_image_path(tmp_path):
    # Create a dummy white image 500x500
    img = Image.new('RGB', (500, 500), color='white')
    path = tmp_path / "test_galaxy.jpg"
    img.save(path)
    return str(path)

def test_preprocessing_shape(fake_image_path):
    """Test that images are resized to 256x256 correctly."""
    # Mimic your Dataset logic
    upscale_factor = 4
    transform = transforms.Compose([
        transforms.Resize((256 // upscale_factor, 256 // upscale_factor)),
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    img = Image.open(fake_image_path).convert("RGB")
    tensor = transform(img)
    
    # Check dimensions: [Channels, Height, Width]
    assert tensor.shape == (3, 256, 256), "Pre-processing failed: Shape is wrong!"

def test_model_output_shape():
    """Test that the model accepts input and returns correct output shape."""
    model = CnnUNet()
    
    # Create a random 'fake' batch of 2 images (Batch=2, C=3, H=256, W=256)
    dummy_input = torch.randn(2, 3, 256, 256)
    
    # Pass through model
    output = model(dummy_input)
    
    # Verify input shape == output shape (U-Net requirement)
    assert output.shape == dummy_input.shape, "Post-processing failed: Output shape mismatch!"