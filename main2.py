import argparse
import copy
import os
import subprocess
import time
from itertools import chain, count

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from astropy.io import fits
import tqdm

from dataset import GG2, BalancedBatchSampler, image_transform, target_transform


def execute(args):
    # define model
    torch.manual_seed(args.init_seed)
    f = torch.hub.load(args.github, args.model, pretrained=True)
    f.conv_stem = torch.nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1, bias=False)
    f.classifier = torch.nn.Linear(1280, 1)
    f.to(args.device)

    # evaluation
    def evaluate(dataset, desc):
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.bs, num_workers=2)

        with torch.no_grad():
            ote = []
            yte = []
            for x, y in tqdm.tqdm(loader, desc=desc):
                x, y = x.to(args.device), y.to(dtype=x.dtype, device=args.device)
                f.train()
                ote += [f(x).flatten()]
                yte += [y]

        return {
            'output': torch.cat(ote).cpu(),
            'labels': torch.cat(yte).cpu(),
        }

    # criterion and optimizer
    criterion = nn.SoftMarginLoss()
    optimizer = torch.optim.SGD(f.parameters(), lr=args.lr, momentum=args.mom)

    # datasets
    dataset = GG2(args.root, transform=image_transform, target_transform=target_transform)
    print("{} images in total".format(len(dataset)))

    torch.manual_seed(args.data_seed)
    trainset, testset, _ = torch.utils.data.random_split(dataset, (args.ntr, args.nte, len(dataset) - args.ntr - args.nte))

    # training
    dummy_trainset = copy.deepcopy(trainset)
    dummy_trainset.dataset.transform = None

    trainloader = torch.utils.data.DataLoader(trainset, sampler=BalancedBatchSampler(dummy_trainset), batch_size=args.bs, drop_last=True, num_workers=2)

    results = []
    torch.manual_seed(args.batch_seed)

    for epoch in range(args.epoch):
        t = tqdm.tqdm(total=len(trainloader), desc='[epoch {}] training'.format(epoch + 1))
        for x, y in trainloader:
            x, y = x.to(args.device), y.to(dtype=x.dtype, device=args.device)

            f.train()
            out = f(x).flatten()
            loss = criterion(out, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t.update(1)
            t.set_postfix_str("loss={0[loss]:.2f} acc={0[acc]:.2f}".format({
                'loss': args.alpha * loss.item(),
                'acc': (out * y > 0).double().mean().item(),
            }))

        t.close()

        results += [{
            'epoch': epoch,
            'train': evaluate(trainset, '[epoch {}] eval trainset'.format(epoch + 1)),
            'test': evaluate(testset, '[epoch {}] eval testset'.format(epoch + 1)),
        }]

        yield {
            'args': args,
            'epochs': results,
            'state': f.state_dict(),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_seed", type=int, required=True)
    parser.add_argument("--data_seed", type=int, required=True)
    parser.add_argument("--batch_seed", type=int, required=True)

    parser.add_argument("--ntr", type=int, required=True)
    parser.add_argument("--nte", type=int, required=True)

    parser.add_argument("--bs", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--mom", type=float, required=True)

    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--root", type=str, required=True)

    parser.add_argument("--github", type=str, default='rwightman/gen-efficientnet-pytorch')
    parser.add_argument("--model", type=str, default='efficientnet_b0')

    parser.add_argument("--pickle", type=str, required=True)
    args = parser.parse_args()

    torch.save(args, args.pickle)
    try:
        for res in execute(args):
            with open(args.pickle, 'wb') as f:
                torch.save(args, f)
                torch.save(res, f)
    except:
        os.remove(args.pickle)
        raise


if __name__ == "__main__":
    main()
