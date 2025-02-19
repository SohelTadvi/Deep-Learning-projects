# Check if the script is being run as the main program
if __name__ == '__main__':
    
    # Import necessary libraries
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision
    import torchvision.transforms as transforms
    import matplotlib.pyplot as plt
    import torch.nn.functional as F

    # Step 1: Load and Preprocess the Data
    # Define the transformations to be applied to the CIFAR-10 dataset images
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to tensor format
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the images to have mean 0.5 and std deviation 0.5
    ])

    # Load the CIFAR-10 training dataset and apply the transformations
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    # Load the CIFAR-10 training data using DataLoader for batching
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)  # Set num_workers=0 for single-threaded loading

    # Load the CIFAR-10 test dataset and apply the same transformations
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    # Load the CIFAR-10 test data using DataLoader for batching
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=0)

    # Define the class labels for CIFAR-10 dataset
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Step 2: Define the CNN Model
    class Net(nn.Module):
        def __init__(self):
            # Initialize the layers of the CNN
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)  # First convolution layer with 6 filters, 5x5 kernel
            self.pool = nn.MaxPool2d(2, 2)  # Max pooling layer with 2x2 window
            self.conv2 = nn.Conv2d(6, 16, 5)  # Second convolution layer with 16 filters, 5x5 kernel
            self.fc1 = nn.Linear(16 * 5 * 5, 120)  # Fully connected layer with 120 output nodes
            self.fc2 = nn.Linear(120, 84)  # Fully connected layer with 84 output nodes
            self.fc3 = nn.Linear(84, 10)  # Output layer with 10 output nodes (for 10 classes)

        def forward(self, x):
            # Forward pass through the layers
            x = self.pool(F.relu(self.conv1(x)))  # Apply ReLU activation after the first convolution
            x = self.pool(F.relu(self.conv2(x)))  # Apply ReLU activation after the second convolution
            x = x.view(-1, 16 * 5 * 5)  # Flatten the feature maps to 1D vector
            x = F.relu(self.fc1(x))  # Apply ReLU activation after the first fully connected layer
            x = F.relu(self.fc2(x))  # Apply ReLU activation after the second fully connected layer
            x = self.fc3(x)  # Output layer (no activation function, raw scores)
            return x

    # Create the CNN model instance
    net = Net()

    # Step 3: Define Loss Function and Optimizer
    # Use CrossEntropyLoss for multi-class classification
    criterion = nn.CrossEntropyLoss()
    # Use Stochastic Gradient Descent (SGD) optimizer with learning rate of 0.001 and momentum of 0.9
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Step 4: Train the Model
    num_epochs = 10  # Set the number of epochs (iterations over the dataset)
    train_losses = []  # List to store the training loss at each epoch
    train_accuracies = []  # List to store the training accuracy at each epoch

    for epoch in range(num_epochs):  
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Iterate over the training data
        for inputs, labels in trainloader:
            optimizer.zero_grad()  # Clear gradients from previous step
            outputs = net(inputs)  # Pass inputs through the model
            loss = criterion(outputs, labels)  # Calculate the loss
            loss.backward()  # Backpropagate the gradients
            optimizer.step()  # Update the model parameters
            
            running_loss += loss.item()  # Accumulate the loss
            _, predicted = torch.max(outputs.data, 1)  # Get the predicted class by finding the max logit
            total += labels.size(0)  # Total number of images
            correct += (predicted == labels).sum().item()  # Count correct predictions
        
        # Calculate average loss and accuracy for the epoch
        epoch_loss = running_loss / len(trainloader)
        epoch_accuracy = correct / total
        train_losses.append(epoch_loss)  # Save loss for plotting
        train_accuracies.append(epoch_accuracy)  # Save accuracy for plotting
        
        # Print the training loss and accuracy for this epoch
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')

    # Step 5: Evaluate the Model
    correct = 0
    total = 0

    # Disable gradient calculation for evaluation
    with torch.no_grad():
        # Iterate over the test data
        for inputs, labels in testloader:
            outputs = net(inputs)  # Get the model's predictions
            _, predicted = torch.max(outputs.data, 1)  # Get the predicted class
            total += labels.size(0)  # Total number of images
            correct += (predicted == labels).sum().item()  # Count correct predictions

    # Print the test accuracy
    print(f'Test Accuracy: {100 * correct / total:.2f}%')

    # Step 6: Visualize the Results
    # Create a figure to plot the training loss and accuracy
    plt.figure(figsize=(10, 5))

    # Plot the training loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot the training accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Adjust layout and display the plots
    plt.tight_layout()
    plt.show()
