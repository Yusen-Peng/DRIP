import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from open_clip_local import create_model_and_transforms, CLIP, DTP_ViT


def finetuning_ViT():

    BATCH_SIZE = 128
    NUM_CLASSES = 1000
    EPOCHS = 10
    LR = 1e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, _, preprocess = create_model_and_transforms(
        model_name="ViT-B-32", 
        pretrained="laion2b_s34b_b79k", 
        DTP_ViT=False
    )

    # remove projection head and add classifier
    model.visual.head = nn.Linear(model.visual.output_dim, NUM_CLASSES)
    model = model.visual.to(DEVICE)

    print("Model loaded!")


    train_dataset = datasets.ImageFolder("/fs/scratch/PAS2836/yusenpeng_dataset/train", transform=preprocess)
    val_dataset = datasets.ImageFolder("/fs/scratch/PAS2836/yusenpeng_dataset/val", transform=preprocess)




    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (output.argmax(1) == labels).sum().item()

        train_acc = correct / len(train_loader.dataset)
        print(f"Epoch {epoch}: Train Loss {total_loss:.2f}, Train Acc {train_acc:.4f}, Val Acc {val_acc:.4f}")
    

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    # report final finetuning test-set accuracy
    val_acc = correct / total
    print("⭐" * 20)
    print(f"Final Validation Accuracy: {val_acc:.4f}")
    print("⭐" * 20)



if __name__ == "__main__":
    finetuning_ViT()