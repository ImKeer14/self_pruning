import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from model import SelfPruningCNN
from utils import compute_sparsity_loss, calculate_sparsity, evaluate
import config


# ------------------ DATA ------------------
def get_data():
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=config.BATCH_SIZE, shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=config.BATCH_SIZE, shuffle=False
    )

    return train_loader, test_loader


# ------------------ TRAIN ------------------
def train_model():

    device = config.DEVICE if torch.cuda.is_available() else "cpu"

    model = SelfPruningCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    train_loader, test_loader = get_data()

    # 🔥 Lambda Scheduling
    start_lam = 0.002
    end_lam = 0.02

    train_losses = []

    for epoch in range(config.EPOCHS):

        model.train()
        total_loss = 0

        # 🔥 Dynamic lambda per epoch
        lam = start_lam + (end_lam - start_lam) * (epoch / (config.EPOCHS - 1))

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()

            output = model(data)

            loss_cls = F.cross_entropy(output, target)
            loss_sparse = compute_sparsity_loss(model)

            # ✅ Correct loss
            loss = loss_cls + lam * loss_sparse

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        train_losses.append(total_loss)

        print(f"Epoch {epoch+1} | Loss: {total_loss:.2f} | Lambda: {lam:.5f}")

    # -------- Evaluation --------
    accuracy = evaluate(model, test_loader, device)
    sparsity = calculate_sparsity(model, config.SPARSITY_THRESHOLD)

    print(f"\nFinal Accuracy: {accuracy:.2f}%")
    print(f"Sparsity: {sparsity:.2f}%")

    # Save model
    torch.save(model.state_dict(), "outputs/model.pth")

    return model, accuracy, sparsity, train_losses


# ------------------ PLOTS ------------------
def plot_loss(train_losses):
    plt.plot(train_losses)
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    plt.savefig("outputs/loss_curve.png")
    plt.show()


def plot_gate_distribution(model):
    all_gates = []

    for module in model.modules():
        if hasattr(module, "gate_scores"):
            gates = torch.sigmoid(module.gate_scores).detach().cpu().numpy().flatten()
            all_gates.extend(gates)

    plt.hist(all_gates, bins=50)
    plt.title("Gate Distribution")
    plt.xlabel("Gate Value")
    plt.ylabel("Frequency")
    plt.grid()
    plt.savefig("outputs/gate_distribution.png")
    plt.show()


# ------------------ MAIN ------------------
if __name__ == "__main__":

    model, acc, sp, losses = train_model()

    plot_loss(losses)
    plot_gate_distribution(model)