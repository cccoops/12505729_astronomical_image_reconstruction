import os 
import torch
import gradio as gr
from PIL import Image
import torch.nn as nn
from torchvision import transforms

# Load weights from git
def load_model_weights(model):
    filename = "best_model.pth"
    GITHUB_MODEL_URL = "https://github.com/cccoops/12505729_astronomical_image_reconstruction/releases/download/v1.0/best_model.pth"
    print("Downloading model weights...")
    torch.hub.download_url_to_file(GITHUB_MODEL_URL, filename)
    print("Model weights downloaded.")

    # Load the weights into the model
    state_dict = torch.load(filename, map_location=torch.device('cpu')) 
    model.load_state_dict(state_dict)
    model.eval()
    return model

# Double convolution class
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

# U-Net class
class CnnUNet(nn.Module):
    def __init__(self):
        super(CnnUNet, self).__init__()

        # Encoder (downsampling)
        self.enc1 = DoubleConv(3, 64)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(256, 512)

        # Decoder (upsampling and skipp connections)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)

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
        # Concatenate the encoder output (e3) with the bottleneck (x)
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

# Setup and config
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Load model
model = CnnUNet().to(DEVICE)

# Load weights
load_model_weights(model)

model.eval()

# Pre-processing
inference_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# The inference function
def deblur_galaxy(image):
    if image is None:
        return None
    
    # Convert Gradio image (numpy) to PIL
    img_pil = Image.fromarray(image).convert("RGB")
    
    # Transform to Tensor
    input_tensor = inference_transform(img_pil).unsqueeze(0).to(DEVICE)
    
    # Run Model
    with torch.no_grad():
        output_tensor = model(input_tensor)
    
    # Clamp to 0-1, move to CPU, convert to Numpy
    output_tensor = output_tensor.squeeze(0).cpu().clamp(0, 1)
    output_img = output_tensor.permute(1, 2, 0).numpy()
    
    return output_img

with gr.Blocks(title="De-noising Astronomical Images Demo") as demo:    
    with gr.Row():
        with gr.Column(scale=1): pass 
        with gr.Column(scale=10):     
            gr.Markdown("# De-noising Astronomical Images Demo")
            gr.Markdown("### Upload a blurry astronomical image, and the U-Net will attempt to restore the details.")
        with gr.Column(scale=1): pass 

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Blurry Input", type="numpy", height=400)
        
        with gr.Column():
            output_image = gr.Image(label="AI Restoration", type="numpy", height=400)

    with gr.Row():
        with gr.Column(scale=1): pass  # Invisible Left Spacer
        
        with gr.Column(scale=1):       # Middle Column for Button
            run_btn = gr.Button("Start Restoration", variant="primary", size="lg")
            
        with gr.Column(scale=1): pass  # Invisible Right Spacer
    
    run_btn.click(fn=deblur_galaxy, inputs=input_image, outputs=output_image)


# Launch the browser app
if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())