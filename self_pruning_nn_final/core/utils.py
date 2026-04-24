import torch
from model import PrunableConv

def compute_sparsity_loss(model):
    loss = 0
    count = 0

    for module in model.modules():
        if hasattr(module, "gate_scores"):
            gates = torch.sigmoid(module.gate_scores)
            loss += gates.mean()
            count += 1

    return loss / (count + 1e-8)

def calculate_sparsity(model, threshold):
    total, pruned = 0, 0
    for module in model.modules():
        if isinstance(module, PrunableConv):
            gates = torch.sigmoid(module.gate_scores)
            total += gates.numel()
            pruned += (gates < threshold).sum().item()
    return 100 * pruned / total

def evaluate(model, dataloader, device):
    correct, total = 0, 0
    model.eval()
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    return 100 * correct / total
