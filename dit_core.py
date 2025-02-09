import torch
import torch.nn as nn
import math
from timm.models.vision_transformer import Attention, Mlp

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
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim,act_layer=approx_gelu)
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
        x = x + self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp)) * gate_mlp.unsqueeze(1)
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
    
if __name__ == "__main__":
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision
    import torchvision.transforms as transforms
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    # Define parameters
    batch_size = 64
    hidden_size = 128
    num_heads = 8
    num_classes = 10
    epochs = 5
    learning_rate = 0.001

    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Define simple model using DitBlock
    class MNISTModel(nn.Module):
        def __init__(self, hidden_size, num_heads, num_classes):
            super(MNISTModel, self).__init__()
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(28 * 28, hidden_size)
            self.dit_block = DitBlock(hidden_size, num_heads)
            self.fc2 = nn.Linear(hidden_size, num_classes)
        
        def forward(self, x):
            x = self.flatten(x)
            x = self.fc1(x)
            c = torch.randn(x.shape[0], x.shape[1], device=x.device)  # Conditioning vector
            x = self.dit_block(x, c)
            x = self.fc2(x[:, 0, :])  # Take only the first token output for classification
            return x

    # Initialize model, loss function, and optimizer
    model = MNISTModel(hidden_size, num_heads, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    def train():
        model.train()
        losses = []
        for epoch in range(epochs):
            total_loss = 0
            for images, labels in train_loader:
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

