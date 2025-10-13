import torch
from torch.utils.data import DataLoader

def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y, x)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()*x.size(0)
    return total_loss/len(loader.dataset)

def validate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y, x)
            total_loss += loss.item()*x.size(0)
    return total_loss/len(loader.dataset)
