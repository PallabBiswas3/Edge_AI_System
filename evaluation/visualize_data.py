import matplotlib.pyplot as plt
import torchvision
from data.dataloader import get_dataloaders

def show_samples():
    train_loader, _ = get_dataloaders(batch_size=8)
    images, labels = next(iter(train_loader))

    img_grid = torchvision.utils.make_grid(images)
    plt.figure(figsize=(8,4))
    plt.imshow(img_grid.permute(1,2,0))
    plt.title(f"Labels: {labels.tolist()}")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    show_samples()
