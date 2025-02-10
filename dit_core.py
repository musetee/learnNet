import torch
import torch.nn as nn
import math
from timm.models.vision_transformer import Attention, Mlp
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchinfo import summary
import numpy as np

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class DitBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super(DitBlock, self).__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size*mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        # pointwise feedforward
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim,act_layer=approx_gelu) # out_features not defined, thus = in_features, hidden_features means the processing features in between
        # another simple Mlp to generate 6 parameters for scaling and shifting
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6*hidden_size, bias=True)
        ) 
    
    def forward(self, x, c):
        # x is input tokens, c is conditioning
        # adaLN_modulation output size of 6*hidden_size, alpha_1, alpha_2, beta_1, beta_2, gamma_1, gamma_2
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1) 
        x = x + self.attn(modulate(self.norm1(x), shift_msa, scale_msa)) * gate_msa.unsqueeze(1)
        output_point_feedforward = self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp)) * gate_mlp.unsqueeze(1)
        x = x + output_point_feedforward
        #print("MLP output: ")
        #print("shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp \n", shift_msa.shape, scale_msa.shape, gate_msa.shape, shift_mlp.shape, scale_mlp.shape, gate_mlp.shape)
        #print("pointwise feedforward output: ", output_point_feedforward.shape)
        return x 
    
class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super(FinalLayer, self).__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2*hidden_size, bias=True)
        )
    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2,dim=1)
        x = self.linear(modulate(self.norm_final(x),shift,scale))
        return x
    
def test_simple_function():
    # Define input parameters
    batch_size = 2
    seq_length = 16  # Example token sequence length
    hidden_size = 128
    num_heads = 8

    # Create random input tensors
    x = torch.randn(batch_size, seq_length, hidden_size)  # Input tokens
    c = torch.randn(batch_size, hidden_size)  # Conditioning tensor

    # Initialize and test the DitBlock
    model = DitBlock(hidden_size, num_heads)
    output = model(x, c)

    # Print output shape to verify consistency
    print("Input shape:", x.shape)
    print("Conditioning shape:", c.shape)
    print("Output shape:", output.shape)


def test_train_dit():
    

    # Define parameters
    batch_size = 64
    hidden_size = 128
    num_heads = 8
    num_classes = 10
    epochs = 5
    learning_rate = 0.001
    patch_size = 1  # MNIST is 28x28, patch_size is set accordingly
    out_channels = 1

    download = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=download)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=download)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Define simple model using DitBlock
    # Define simple model using DitBlock and FinalLayer
    # Define Conditional DiT Model for Image Generation
    class ConditionalDiT(nn.Module):
        def __init__(self, hidden_size, num_heads, num_classes, patch_size, out_channels):
            super(ConditionalDiT, self).__init__()
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(28 * 28, hidden_size)
            self.label_embedding = nn.Embedding(num_classes, hidden_size)
            self.dit_block = DitBlock(hidden_size, num_heads)
            self.final_layer = FinalLayer(hidden_size, patch_size, out_channels)
        
        def forward(self, x, labels):
            x = self.flatten(x)
            x = self.fc1(x)
            label_emb = self.label_embedding(labels)  # Embed class label
            c = label_emb  # Use label as conditioning
            x = self.dit_block(x, c)
            x = self.final_layer(x, c)
            return x.view(-1, 1, 28, 28)  # Reshape to image size

    # Initialize model, loss function, and optimizer
    model = ConditionalDiT(hidden_size, num_heads, num_classes, patch_size, out_channels).to(device)
    print(summary(model, input_size=[(batch_size, 1, 28, 28), (batch_size,)]))
    print(summary(model, input_size=(batch_size, 1, 28, 28)))

    criterion = nn.CrossEntropyLoss()
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    def train():
        model.train()
        losses = []
        for epoch in range(epochs):
            total_loss = 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                x = images
                beta_a = 1.0
                beta_b = 2.5
                noise_level = torch.tensor(
                np.random.beta(beta_a, beta_b, len(x)), device=device
                 )
                signal_level = 1 - noise_level
                noise = torch.randn_like(x)

                x_noisy = noise_level.view(-1, 1, 1, 1) * noise + signal_level.view(-1, 1, 1, 1) * x

                x_noisy = x_noisy.float()
                noise_level = noise_level.float()
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            losses.append(avg_loss)
            print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')
        return losses

    # Train the model
    losses = train()

    # Plot training loss
    plt.plot(range(1, epochs+1), losses, marker='o', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.show()

    # Test the model and visualize some predictions
    def test():
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'Accuracy: {100 * correct / total:.2f}%')

    test()

if __name__ == "__main__":
    #test_simple_function()
    test_train_dit()