import torch
from torch.utils.data import DataLoader
from model import NvidiaModel
from data import DrivingDataset
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os

def train(csv_train, img_dir, csv_val=None, epochs=20, batch_size=64, lr=1e-4, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    train_ds = DrivingDataset(csv_train, img_dir, augment=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    if csv_val:
        val_ds = DrivingDataset(csv_val, img_dir, augment=False)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    model = NvidiaModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    writer = SummaryWriter()

    global_step = 0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for imgs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            imgs = imgs.to(device)
            targets = targets.to(device)
            preds = model(imgs)
            loss = criterion(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if global_step % 50 == 0:
                writer.add_scalar('train/loss', loss.item(), global_step)
            global_step += 1
        avg_train_loss = running_loss/len(train_loader)
        print(f"Epoch {epoch+1} train_loss: {avg_train_loss:.5f}")
        if csv_val:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for imgs, targets in val_loader:
                    imgs = imgs.to(device); targets = targets.to(device)
                    preds = model(imgs)
                    val_loss += criterion(preds, targets).item()
            val_loss /= len(val_loader)
            print(f"Epoch {epoch+1} val_loss: {val_loss:.5f}")
            writer.add_scalar('val/loss', val_loss, epoch)
        # checkpoint
        os.makedirs('checkpoints', exist_ok=True)
        torch.save(model.state_dict(), f'checkpoints/model_epoch{epoch+1}.pt')
    writer.close()

if __name__ == '__main__':
    train('data/drive_log.csv', 'data/IMG', epochs=20)
