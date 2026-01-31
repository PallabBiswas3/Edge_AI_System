import torch
import matplotlib.pyplot as plt
import torchvision
from models.simple_cnn import SimpleCNN
from data.dataloader import get_dataloaders

classes = ['airplane','automobile','bird','cat','deer',
           'dog','frog','horse','ship','truck']

def visualize_predictions():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load("baseline_model.pth"))
    model.eval()

    _, test_loader = get_dataloaders(batch_size=8)
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

    images = images.cpu()
    preds = preds.cpu()
    labels = labels.cpu()

    fig, axs = plt.subplots(2,4, figsize=(10,5))
    for i, ax in enumerate(axs.flat):
        ax.imshow(images[i].permute(1,2,0))
        ax.set_title(f"P: {classes[preds[i]]}\nT: {classes[labels[i]]}")
        ax.axis("off")

    plt.show()

if __name__ == "__main__":
    visualize_predictions()
