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

from dataset import GG2, BalancedBatchSampler


def image_transform(images):
    images = [fits.open(file, memmap=False)[0].data for file in images]
    images = [torch.from_numpy(x.byteswap().newbyteorder()) for x in images]

    # normalize the second moment of the channels to 1
    normalize = [3.5239e+10, 1.5327e+09, 1.8903e+09, 1.2963e+09]
    images = [x.mul(n) for x, n in zip(images, normalize)]

    # stack the 3 channels of small resolution together
    vis, j, y, h = images
    vis, jyh = vis[None], torch.stack([j, y, h])

    upsample = torch.nn.Upsample(200, mode='bilinear', align_corners=True)
    jyh = upsample(jyh[None])[0]

    return torch.cat([vis, jyh])


def target_transform(prop):
    if prop['n_sources'] > 0:
        return 1.0
    return -1.0


def execute(args):
    dataset = GG2(args.root, transform=image_transform, target_transform=target_transform)
    print("{} images in total".format(len(dataset)))

    torch.manual_seed(args.data_seed)
    trainset, testset, _ = torch.utils.data.random_split(dataset, (args.ntr, args.nte, len(dataset) - args.ntr - args.nte))

    dummy_trainset = copy.deepcopy(trainset)
    dummy_trainset.dataset.transform = None

    trainloader = torch.utils.data.DataLoader(trainset, sampler=BalancedBatchSampler(dummy_trainset), batch_size=args.bs, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs, num_workers=2)

    torch.manual_seed(args.init_seed)
    f = torch.hub.load('rwightman/gen-efficientnet-pytorch', 'efficientnet_b0', pretrained=True)
    f.conv_stem = torch.nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1, bias=False)
    f.classifier = torch.nn.Linear(1280, 1)
    f.to(args.device)

    torch.manual_seed(args.batch_seed)
    criterion = nn.SoftMarginLoss()
    optimizer = torch.optim.SGD(f.parameters(), lr=args.lr, momentum=args.mom)

    for epoch in tqdm.tqdm(range(args.epoch)):
        t = tqdm.tqdm(total=len(trainloader), desc='training')
        for x, y in trainloader:
            x, y = x.to(args.device), y.to(dtype=x.dtype, device=args.device)

            f.train()
            out = f(x).flatten()
            loss = criterion(out, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t.update(1)
            t.set_postfix({
                'loss': loss.item(),
                'acc': (out * y > 0).double().mean().item(),
            })

        t.close()

        with torch.no_grad():
            ote = []
            yte = []
            for x, y in tqdm.tqdm(testloader, desc='testing'):
                x, y = x.to(args.device), y.to(dtype=x.dtype, device=args.device)
                f.eval()
                ote += [f(x).flatten()]
                yte += [y]
            yield {
                'args': args,
                'test': {
                    'output': torch.cat(ote),
                    'labels': torch.cat(yte),
                }
            }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_seed", type=int, required=True)
    parser.add_argument("--data_seed", type=int, required=True)
    parser.add_argument("--batch_seed", type=int, required=True)

    parser.add_argument("--ntr", type=int, required=True)
    parser.add_argument("--nte", type=int, required=True)

    # parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--bs", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--mom", type=float, required=True)

    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--root", type=str, required=True)

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
