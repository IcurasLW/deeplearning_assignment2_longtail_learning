import torch 
import torch.nn as nn
import torchvision.models as models
import argparse
from utils import *
import torch.nn.functional as F
from models import *
from tqdm import tqdm
from dataset import *




DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'



parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")


parser.add_argument(
    "-sp",
    "--save_path",
    default="./results/"
)


parser.add_argument(
    "-ls",
    "--loss",
    default="crossentropy",
    help='focalloss, crossentropy, focallossv1, classdistance, cbloss'
)


parser.add_argument(
    "-mn",
    "--modelname", 
    default='efficientnet_b0', 
    type=str
)


parser.add_argument(
    "-dn",
    "--dataname", 
    default='cifar100', 
    type=str
)


parser.add_argument(
    "-nc",
    "--num_class", 
    default=100
)


parser.add_argument(
    "-bs",
    "--batch_size",
    default=256,
    type=int
)


parser.add_argument(
    "-e",
    "--epochs", 
    default=50
)


def train(args, model, loss_fn, loader, optimizer, epoch):
    model.train()
    y_pred = []
    y_true = []
    losses = 0
    loop = tqdm(loader)
    
    for i, (image, label) in enumerate(loop):
        image = image.to(DEVICE)
        output = model(image)
        optimizer.zero_grad()
        label = label.to(DEVICE)
        loss = loss_fn(output, label)
        y_pred.append(F.softmax(output, dim=1).detach().cpu().numpy())
        y_true.extend(label.detach().cpu().numpy())
        losses += loss.item()
        loss.backward()
        optimizer.step()

    y_pred = np.concatenate(y_pred, axis=0)
    metrics = evaluate(args, y_pred, y_true)
    metrics['loss'] = losses / (i+1)

    progress = ProgressMeter(
        args,
        mode='train',
        meters=metrics,
        prefix= f'Epoch [{epoch}] --->  '
    )
    
    progress.display()



def validate(args, model, loss_fn, loader):
    model.eval()
    y_pred = []
    y_true = []
    losses = 0
    loop = tqdm(loader)
    with torch.no_grad():
        for i, (image, label) in enumerate(loop):
            image = image.to(DEVICE)
            label = label.to(DEVICE)
            output = model(image)
            
            loss = loss_fn(output, label)
            y_pred.append(F.softmax(output, dim=1).detach().cpu().numpy())
            y_true.extend(label.detach().cpu().numpy())
            losses += loss.item()

    y_pred = np.concatenate(y_pred, axis=0)
    metrics = evaluate(args, y_pred, y_true)
    metrics['loss'] = losses / (i+1)

    progress = ProgressMeter(
        args,
        mode='test',
        meters=metrics,
        prefix= f'Testing --->  '
    )
    
    progress.display()
    return metrics['loss']



def main(args):
    # Prepared model
    model = models.__dict__[args.modelname]()
    in_feature = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Linear(in_feature, args.num_class))
    
    
    # in_feature = model.fc.in_features
    # model.fc = nn.Linear(in_feature, args.num_class)
    
    print(model)
    
    model = model.to(DEVICE)
    # optimizer = torch.optim.SGD(model.parameters(), lr=5e-4, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience = 3, threshold=1e-4, verbose=2)
    train_loader, val_loader = prepare_data(args)
    loss_fn = prepare_loss(args, train_loader)
    
    
    # Prepare data
    best_loss = 100000
    for epoch in range(args.epochs):
        train(args, model, loss_fn, train_loader, optimizer, epoch)
        torch.cuda.empty_cache()
        val_loss = validate(args, model, loss_fn, val_loader)
        scheduler.step(val_loss)
        best_loss = save_model(val_loss, best_loss, model, args)



if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
    