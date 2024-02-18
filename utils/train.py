import torch
import matplotlib.pyplot as plt
import seaborn as sns

def train(model, epoch, optimizer, train_dataloader, valid_dataloader, device):
    save_path = f'./saved_model/best_{model.__class__.__name__}.pth'
    loss_func = torch.nn.CrossEntropyLoss()
    train_acc, valid_acc = [], []
    best_valid_acc = 0.0  # 用于跟踪最佳验证准确率
    best_epoch = 0

    train_count, valid_count = len(train_dataloader.dataset), len(valid_dataloader.dataset)

    for ite in range(epoch):
        train_cor_count = evaluate(model, train_dataloader, optimizer, loss_func, device)
        valid_cor_count = evaluate(model, valid_dataloader, None, loss_func, device)

        train_acc.append(train_cor_count / train_count)
        valid_acc.append(valid_cor_count / valid_count)

        print(f'epoch {ite + 1}/{epoch} -> train acc: {train_acc[-1]:.4f}, valid acc: {valid_acc[-1]:.4f}')

        # 如果当前验证准确率是历史最佳，则保存模型
        if valid_acc[-1] > best_valid_acc:
            best_valid_acc = valid_acc[-1]
            best_epoch = ite + 1
            torch.save(model.state_dict(), save_path)

    print(f'Best validation accuracy achieved at epoch {best_epoch}. Model saved to {save_path}')
    plot_results(train_acc, valid_acc, epoch, model.__class__.__name__)

def evaluate(model, dataloader, optimizer, loss_func, device):
    model.to(device)
    model.train() if optimizer else model.eval()

    cor_count, total_loss = 0, 0.0

    with torch.set_grad_enabled(optimizer is not None):
        for img, target, idx, mask in dataloader:
            img, target, mask, idx = img.to(device), target.to(device), mask.to(device), idx.to(device)
            output = model(idx, mask, img)
            loss = loss_func(output, target)

            if optimizer:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            cor_count += (output.argmax(dim=1) == target).sum().item()

    return cor_count


def plot_results(train_acc, valid_acc, epoch, model_name):
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))

    plt.plot(train_acc, label="train accuracy", marker='o', linestyle='-', color='b')
    plt.plot(valid_acc, label="valid accuracy", marker='o', linestyle='-', color='r')

    plt.title(f'{model_name} Training Progress', fontsize=16)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.xticks(range(epoch), range(1, epoch + 1))
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)

    plt.tight_layout()
    plt.show()