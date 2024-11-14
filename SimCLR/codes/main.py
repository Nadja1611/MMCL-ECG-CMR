import torch
import torch.nn as nn
from torch.optim import Adam
import fire
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset, RandomSampler
from tqdm import tqdm
from simclr import SimCLR, N_XENT
from torch.utils.tensorboard import SummaryWriter
import os
import torch.nn.functional as F

class CustomTensorDataset(Dataset):
    def __init__(self, data_tensors, labels_tensors, transform=None):
        """
        Args:
            data_tensors (torch.Tensor): Tensor containing image data, 
                                         shape (num_samples, channels, height, width).
            labels_tensors (torch.Tensor): Tensor containing labels, shape (num_samples,).
            transform (callable, optional): Optional transform to apply to each sample.
        """
        self.data = data_tensors
        self.targets = labels_tensors
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img1 = self.transform(img)
            img2 = self.transform(img)
        else:
            img1, img2 = img, img  # If no transform is provided

        return torch.stack([img1, img2]), target

    def __len__(self):
        return len(self.data)

def main(epochs: int = 1, batch_size=128):
    # Replace CIFAR-10 transforms with appropriate ones for your data
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=32),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

    # Load your custom tensor data
    # Assuming data and labels are already tensors
    train_data = torch.load('path/to/your_train_data.pt')  # Tensor of shape (num_samples, channels, height, width)
    train_labels = torch.load('path/to/your_train_labels.pt')  # Tensor of shape (num_samples,)
    val_data = torch.load('path/to/your_val_data.pt')
    val_labels = torch.load('path/to/your_val_labels.pt')

    # Initialize custom datasets
    dataset_train = CustomTensorDataset(data_tensors=train_data, labels_tensors=train_labels, transform=train_transform)
    dataset_val = CustomTensorDataset(data_tensors=val_data, labels_tensors=val_labels, transform=train_transform)

    sampler_train = RandomSampler(dataset_train)
    sampler_val = RandomSampler(dataset_val)

    train_loader = DataLoader(dataset=dataset_train, sampler=sampler_train, batch_size=batch_size)
    val_loader = DataLoader(dataset=dataset_val, sampler=sampler_val, batch_size=batch_size)

    model = SimCLR(device="cuda")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.3 * (batch_size / 256), momentum=0.9)
    loss_func = N_XENT()
    writer = SummaryWriter(log_dir="checkpoints")
    device = model.device
    model.train()

    for i in range(1, epochs + 1):
        total_loss = torch.tensor(0.).to(device)
        total_num = 0
        for j, (img, target) in enumerate(tqdm(train_loader, desc=f'training epoch: {i}')):
            img = img.view(img.shape[0] * 2, img.shape[2], img.shape[3], img.shape[4])

            out = model(img.to(device))

            optimizer.zero_grad()
            loss = loss_func(out)
            total_num += img.size(0)
            loss.backward()
            optimizer.step()

            total_loss += loss.detach().item() * img.size(0)
        print(f"Epoch {i} training loss: {total_loss / total_num}")
        writer.add_scalar("train_loss", total_loss / total_num, global_step=i)

        val_loss = 0.0
        total_num = 0
        model.eval()
        for j, (img, target) in enumerate(tqdm(val_loader, desc=f'validation epoch: {i}')):
            with torch.no_grad():
                img = img.view(img.shape[0] * 2, img.shape[2], img.shape[3], img.shape[4])
                out = model(img.to(device))
                loss = loss_func(out)
                val_loss += loss.detach().item() * img.size(0)
                total_num += img.size(0)
        print(f"Epoch {i} validation loss: {val_loss / total_num}")
        writer.add_scalar("val_loss", val_loss / total_num, global_step=i)
        model.train()
        if i % 10 == 0:
            torch.save(model.state_dict(), os.path.join("checkpoints", f"model-{i}.pt"))

if __name__ == "__main__":
    fire.Fire(main)
