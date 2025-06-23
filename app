import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F

# Load the trained generator model
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(100 + 10, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 784),
            nn.Tanh()
        )
    
    def forward(self, z, labels):
        labels = F.one_hot(labels, num_classes=10).float()
        z = torch.cat([z, labels], dim=1)
        return self.main(z).view(-1, 1, 28, 28)

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator().to(device)
generator.load_state_dict(torch.load('generator.pth', map_location=device))
generator.eval()

# Streamlit app
st.title("Handwritten Digit Generator")
st.write("Generate MNIST-style handwritten digits (0-9)")

# User input
digit = st.selectbox("Select a digit to generate:", options=list(range(10)))

if st.button("Generate 5 Images"):
    # Generate noise vectors
    noise = torch.randn(5, 100, device=device)
    labels = torch.full((5,), digit, dtype=torch.long, device=device)
    
    # Generate images
    with torch.no_grad():
        generated_images = generator(noise, labels).cpu().numpy()
    
    # Display images
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i, ax in enumerate(axes):
        ax.imshow(generated_images[i][0], cmap='gray')
        ax.axis('off')
    st.pyplot(fig)
    
    st.success(f"Generated 5 variations of digit {digit}")

st.write("Note: This app uses a Conditional GAN trained on the MNIST dataset.")
