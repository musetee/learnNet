import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import radon, iradon

class RadonSelfAttention(nn.Module):
    def __init__(self, angles=180, projection_dim=50, num_heads=4):
        super(RadonSelfAttention, self).__init__()
        self.angles = np.linspace(0., 180., angles, endpoint=False)
        self.projection_dim = projection_dim
        self.multihead_attn = nn.MultiheadAttention(embed_dim=projection_dim, num_heads=num_heads, batch_first=True)
    
    def forward(self, x):
        batch_size, channels, height, width = x.shape
        output = torch.zeros_like(x)
        
        for b in range(batch_size):
            for c in range(channels):
                img = x[b, c].detach().cpu().numpy()
                radon_transformed = radon(img, self.angles, circle=True)
                radon_tensor = torch.tensor(radon_transformed, dtype=torch.float32, device=x.device)
                
                # Multihead Self-Attention in Radon space
                attn_output, _ = self.multihead_attn(radon_tensor, radon_tensor, radon_tensor)
                
                # Inverse Radon Transform (FBP)
                inverse_img = iradon(attn_output.detach().cpu().numpy(), self.angles, circle=True)
                inverse_img = torch.tensor(inverse_img, dtype=torch.float32, device=x.device)
                
                output[b, c] = inverse_img[:height, :width]  # Crop to original size
        
        return output

# Example usage
if __name__ == "__main__":
    image = torch.rand(1, 1, 32, 32)  # Simulated grayscale image (1 batch, 1 channel, 32x32)
    radon_attention = RadonSelfAttention()
    output = radon_attention(image)
    
    # Visualize Input and Output
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(image.squeeze().numpy(), cmap='gray')
    axs[0].set_title('Original Image')
    axs[1].imshow(output.squeeze().detach().numpy(), cmap='gray')
    axs[1].set_title('Radon Self-Attention Output')
    plt.show()
