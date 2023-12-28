import os
import argparse
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
from dataset import WakeWordData, collate_fn
from model import LSTMWW
from sklearn.metrics import classification_report
from tabulate import tabulate


def save_checkpoint(checkpoint_path, model, optimizer, scheduler, model_params, notes=None):
    torch.save({
        "notes": notes,
        "model_params": model_params,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict()
    }, checkpoint_path)


def binary_accuracy(preds, y):
    rounded_preds = torch.round(preds)
    acc = rounded_preds.eq(y.view_as(rounded_preds)).sum().item() / len(y)

    return acc


def test(test_loader, model, device, epoch):
    print(f'\n Starting the test for season {epoch}')
    accs = []
    preds = []
    labels = []
    model.eval()

    with torch.no_grad():
        for idx, (mfcc, label) in enumerate(test_loader):
            mfcc, label = mfcc.to(device), label.to(device)
            output = model(mfcc)
            pred = torch.sigmoid(output)
            acc = binary_accuracy(pred, label)
            preds += torch.flatten(torch.round(pred)).cpu()
            labels += torch.flatten(label).cpu()
            accs.append(acc)
            print(f'Iter: {idx}/{len(test_loader)}, Precision: {acc}', end="\r")

    average_acc = sum(accs) / len(accs)
    print('Average test accuracy: ', average_acc, "\n")
    report = classification_report(labels, preds)
    print(report)

    return average_acc, report


def train(train_loader, model, optimizer, loss_fn, device, epoch):
    print(f'\n Starting to train epoch {epoch}')
    losses = []
    preds = []
    labels = []
    model.train()

    for idx, (mfcc, label) in enumerate(train_loader):
        mfcc, label = mfcc.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(mfcc)
        pred = torch.round(torch.sigmoid(output))
        loss = loss_fn(torch.flatten(output), label)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        pred = torch.sigmoid(output)
        preds += torch.flatten(torch.round(pred)).cpu()
        labels += torch.flatten(label).cpu()

        print(f'epoch: {epoch}, Iter: {idx}/{len(train_loader)}, loss: {loss.item()}', end='\r')

    avg_train_loss = sum(losses) / len(losses)
    acc = binary_accuracy(torch.Tensor(preds), torch.Tensor(labels))
    print('Train loss average:', avg_train_loss, 'Train precision average', acc)
    report = classification_report(torch.Tensor(labels).numpy(), torch.Tensor(preds).numpy())
    print(report)

    return acc, report


def main(args):
    use_cuda = True
    torch.manual_seed(1)
    device = torch.device('cuda' if args.no_cuda else 'cpu')
    train_dataset = WakeWordData(data_json=args.train_data_json, sample_rate=args.sample_rate, valid=False)
    test_dataset = WakeWordData(data_json=args.test_data_json, sample_rate=args.sample_rate, valid=True)
    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if use_cuda else {}
    train_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        **kwargs
    )
    test_loader = data.DataLoader(
        dataset=test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        **kwargs
    )
    model_params = {
        'num_classes': 1, 'feature_size': 40, 'hidden_size': args.hidden_size,
        'num_layers': 1, 'dropout': 0.1, 'bidirectional': False
    }
    model = LSTMWW(**model_params, device=device)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    best_train_acc, best_train_report = 0, None
    best_test_acc, best_test_report = 0, None
    best_epoch = 0

    for epoch in range(args.epochs):
        print(f'\nStarting training with learning rate', optimizer.param_groups[0]['lr'])
        train_acc, train_report = train(train_loader, model, optimizer, loss_fn, device, epoch)
        test_acc, test_report = test(test_loader, model, device, epoch)

        if train_acc > best_train_acc:
            best_train_acc = train_acc
        if test_acc > best_test_acc:
            best_test_acc = test_acc

        if args.save_checkpoint_path and test_acc >= best_test_acc:
            checkpoint_path = os.path.join(args.save_checkpoint_path, args.model_name + '.pt')
            print('Found the best checkpoint. Save model as', checkpoint_path)
            save_checkpoint(
                checkpoint_path, model, optimizer, scheduler, model_params,
                notes=f'precision_training: {best_train_acc}, precision_test: {best_test_acc}, epoch: {epoch}'
            )
            best_train_report = train_report
            best_test_report = test_report
            best_epoch = epoch

        table = [
            ['Precision train', train_acc],
            ['Precision test', test_acc],
            ['Better training accuracy', best_train_acc],
            ['Best testing accuracy', best_test_acc],
            ['Best epoch', best_epoch]
        ]
        print(tabulate(table))
        scheduler.step(train_acc)

    print('Training completed...')
    print('Best model saved in', checkpoint_path)
    print('Best epoch', best_epoch)
    print('\nTrain Report \n')
    print(best_train_report)
    print('\nTest Report\n')
    print(best_test_report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Wake Word Training Script')
    parser.add_argument('--sample_rate', type=int, default=8000, help='sample_rate for data')
    parser.add_argument('--epochs', type=int, default=50, help='epoch size')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--eval_batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disable CUDA training')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of data loading workers')
    parser.add_argument('--hidden_size', type=int, default=128, help='lstm hidden size')
    parser.add_argument(
        '--model_name',
        type=str,
        default="wakeword",
        required=False,
        help='Model name to save'
    )
    parser.add_argument(
        '--save_checkpoint_path',
        type=str,
        default=None,
        help='path to save the best checkpoint'
    )
    parser.add_argument(
        '--train_data_json',
        type=str,
        default=None,
        required=True,
        help='Path to train data json file'
    )
    parser.add_argument(
        '--test_data_json',
        type=str,
        default=None,
        required=True,
        help='Path to test data json file'
    )

    args = parser.parse_args()

    main(args)
