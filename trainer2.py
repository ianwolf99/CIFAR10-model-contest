import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import math
import re
import onnx_tool
import torch.onnx
import yaml
import os

###################################################################################################
##############################  This is just a crap code made by me. ############################## 
###################################################################################################


# ========== Modifiable Parameters ==========
num_classes = 10  # Number of classes for CIFAR-10
epochs = 50  # Number of training epochs
batch_size = 128  # Batch size for DataLoader
width_factor = 0.36  # Width scaling factor for the model layers
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use CUDA if available, else CPU
lr_values = [0.1]  # List of learning rates to try
# ============================================

# Function to format numbers for filenames
def format_number_filename(num):
    if abs(num) >= 1_000_000:
        return f'{int(num / 1_000_000)}M'  # Round to the nearest million
    elif abs(num) >= 1_000:
        return f'{int(num / 1_000)}k'  # Round to the nearest thousand
    else:
        return str(num)

# Function to format large numbers for readability
def format_number(num):
    if abs(num) >= 1_000_000:
        return f'{num / 1_000_000:.1f}M'  # Format in millions
    elif abs(num) >= 1_000:
        return f'{num / 1_000:.1f}k'  # Format in thousands
    else:
        return str(num)

# Function to round numbers to the nearest significant digit
def round_significant(x, digits=2):
    if x == 0:
        return 0
    else:
        return round(x, -int(math.floor(math.log10(abs(x))) - (digits - 1)))

# Function to calculate FLOPs using ONNX
def calculate_flops_onnx(model):
    # Generate a dummy input for the model with CIFAR-10 dimensions (3x32x32)
    dummy_input = torch.randn(1, 3, 32, 32).to(device)
    
    # Paths to save the ONNX model and profile
    onnx_path = "tmp.onnx"
    profile_path = "profile.txt"
    
    # Export the PyTorch model to ONNX format
    torch.onnx.export(model,
                      dummy_input,
                      onnx_path,
                      export_params=True,
                      opset_version=12,
                      do_constant_folding=True,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes=None)
    
    # Profile the ONNX model to calculate the number of MACs
    onnx_tool.model_profile(onnx_path, save_profile=profile_path)
    
    # Read and parse the profile to extract total MACs
    with open(profile_path, 'r') as file:
        profile = file.read()
    
    # Use regex to find the total MACs in the profile
    match = re.search(r'Total\s+_\s+([\d,]+)\s+100%', profile)
    
    if match:
        total_macs = match.group(1)
        total_macs = int(total_macs.replace(',', ''))  # Remove commas for calculation
        total_macs = round_significant(total_macs)
        return total_macs
    else:
        return None

# ECABlock class that adds channel-wise attention to the model
class ECABlock(nn.Module):
    def __init__(self, channels, gamma=4, b=24):
        super(ECABlock, self).__init__()
        
        # Calculate kernel size based on input channel size
        t = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = t if t % 2 else t + 1
        
        # Define average pooling and 1D convolution
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Apply global average pooling and convolution to calculate channel-wise attention
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = self.sigmoid(y)
        y = y.transpose(-1, -2).unsqueeze(-1)
        
        return x * y.expand_as(x)  # Element-wise multiplication for channel attention

# InvertedResidual block that can optionally use ECABlock for attention
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, use_eca=False):
        super(InvertedResidual, self).__init__()
        
        # Hidden dimension after expansion
        hidden_dim = int(inp * expand_ratio)
        
        # Check if residual connection is applicable
        self.use_res_connect = (stride == 1 and inp == oup)

        # Build layers: expansion, depthwise convolution, pointwise convolution
        layers = []
        if expand_ratio != 1:
            layers.extend([nn.Conv2d(inp, hidden_dim, 1, bias=False),
                           nn.BatchNorm2d(hidden_dim),
                           nn.GELU()])  # Use GELU activation
        
        # Add depthwise convolution and batch norm
        layers.extend([nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                       nn.BatchNorm2d(hidden_dim),
                       nn.GELU()])
        
        # Optionally add ECABlock for attention
        if use_eca:
            layers.append(ECABlock(hidden_dim))

        # Add final pointwise convolution
        layers.extend([nn.Conv2d(hidden_dim, oup, 1, bias=False),
                       nn.BatchNorm2d(oup)])

        self.conv = nn.Sequential(*layers)  # Define the sequential model

    def forward(self, x):
        # Forward pass through convolution layers
        out = self.conv(x)
        
        # Add residual connection if applicable
        if self.use_res_connect:
            return x + out
        else:
            return out


# Define the MobileNetECA architecture
#This class takes as input a  block_settings.yml which contains a list of different block settings, trains each block setting with a learning rate of 0.1, and saves the trained models.
#But  now  learning rate is set to 0.1, trained, and later on  re-trained with different learning rates.
class MobileNetECA(nn.Module):
    def __init__(self, num_classes=10, width_mult=0.2, block_settings=None):
        super(MobileNetECA, self).__init__()

        # Default block_settings if not provided
        if block_settings is None:
            block_settings = [
                [2, 24, 2, 1, True],  # Block 1
                [4, 24, 3, 2, True],  # Block 2
                [8, 36, 3, 2, True],  # Block 3
                [8, 44, 3, 1, True],  # Block 4
            ]

        
        # Calculate input and output channel sizes based on width factor
        input_channel = max(int(36 * width_mult), 8)
        last_channel = max(int(144 * width_mult), 8)

        # First convolution layer
        self.features = [nn.Sequential(
            nn.Conv2d(3, input_channel, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.GELU()
        )]

        # Add inverted residual blocks
        for idx, (t, c, n, s, use_eca) in enumerate(block_settings):
            output_channel = max(int(c * width_mult), 8)
            for i in range(n):
                stride = s if i == 0 else 1  # First layer in block may have stride > 1
                self.features.append(InvertedResidual(input_channel, output_channel, stride, expand_ratio=t, use_eca=use_eca))
                input_channel = output_channel

        # Final convolution layer
        self.features.append(nn.Sequential(
            nn.Conv2d(input_channel, last_channel, 1, bias=False),
            nn.BatchNorm2d(last_channel),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1)
        ))

        self.features = nn.Sequential(*self.features)  # Combine all layers

        # Final classifier layer
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(last_channel, num_classes)

        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        # Forward pass through feature extractor and classifier
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.classifier(x)
        return x

    # Function to initialize model weights
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

# Data augmentation and normalization for CIFAR-10 training dataset
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Random horizontal flip
    transforms.RandomCrop(32, padding=4),  # Random crop with padding
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))  # Normalize based on dataset statistics
])

# Normalization for test data
test_transforms = transforms.Compose([
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))  # Normalize based on dataset statistics
])

# Load CIFAR-10 training and test datasets
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transforms)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# Load block_settings from external YAML file
with open('block_settings.yaml', 'r') as f:
    block_settings_dict = yaml.safe_load(f)



# Function to train and evaluate the model with each learning rate
for block_settings_name, block_settings in block_settings_dict.items():
    for lr in lr_values:
        # Create the MobileNetECA model with the provided block_settings and move it to the appropriate device
        model = MobileNetECA(num_classes=num_classes, width_mult=width_factor, block_settings=block_settings).to(device)
        
        # Calculate the number of parameters and FLOPs of the model
        params = sum(p.numel() for p in model.parameters())
        macs = calculate_flops_onnx(model)
        

        formatted_params = format_number(params)
        formatted_macs = format_number(macs)
        print(f"Total number of parameters for lr={lr}: {formatted_params}")
        
        # Set up the optimizer (SGD) and learning rate scheduler (Cosine Annealing)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=3e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

        # Loss function: Cross-Entropy Loss
        criterion = nn.CrossEntropyLoss()

        # Function to train for one epoch
        def train():
            model.train()  # Set model to training mode
            correct = 0
            total = 0

            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()  # Reset gradients
                outputs = model(inputs)  # Forward pass
                loss = criterion(outputs, targets)  # Calculate loss
                loss.backward()  # Backpropagation
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)  # Gradient clipping
                optimizer.step()  # Update weights

                _, predicted = outputs.max(1)  # Get predicted class
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            accuracy = 100. * correct / total  # Calculate accuracy
            return accuracy

        # Function to validate on the test set
        def validate():
            model.eval()  # Set model to evaluation mode
            correct = 0
            total = 0

            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)  # Forward pass
                    loss = criterion(outputs, targets)  # Calculate loss

                    _, predicted = outputs.max(1)  # Get predicted class
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

            accuracy = 100. * correct / total  # Calculate accuracy
            return accuracy

        # Calculate and print parameters and MACs
        print(f"------ Rounded Parameters for lr={lr} ------")
        params = sum(param.numel() for param in model.parameters())  # Total number of parameters
        params = round_significant(params)
        macs = calculate_flops_onnx(model)  # Calculate FLOPs using ONNX
        formatted_params = format_number(params)
        formatted_macs = format_number(macs)
        print(f"Params: {formatted_params}  MACS: {formatted_macs}")

        # Training loop for multiple epochs
        for epoch in range(epochs):
            acc_train = train()  # Train for one epoch
            acc_valid = validate()  # Validate on the test set
            scheduler.step()  # Update learning rate
            print(f'Epoch {epoch+1} - Training Accuracy: {acc_train:.2f}% - Validation Accuracy: {acc_valid:.2f}%')

        # Format parameters, MACs, accuracy, and learning rate for saving the model
        params_str = format_number_filename(params)
        macs_str = format_number_filename(macs)
        acc_str = f"{acc_valid:.1f}".replace('.', '_')  # Format accuracy like 84.4% -> 84_4
        lr_str = f"{lr:.2f}".replace('.', '_').rstrip('0').rstrip('_')  # Format learning rate

        # Add the block_settings name to the filename
        block_name_str = re.sub(r'\W+', '_', block_settings_name)

        # Save the trained model using TorchScript with a formatted filename
        model_save_dir = '/workspace/Dima/saved/try1/'
        os.makedirs(model_save_dir, exist_ok=True)
        model_path = os.path.join(model_save_dir, f'{block_name_str}_{params_str}_{macs_str}_{acc_str}_{lr_str}.pt')
        scripted_model = torch.jit.script(model)
        scripted_model.save(model_path)
        print(f"Model saved as '{model_path}'")
