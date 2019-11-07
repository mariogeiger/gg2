import argparse
import copy
import subprocess
import time
from itertools import chain, count

import matplotlib.pyplot as plt
import torch.nn as nn
from tqdm import tqdm, tqdm_notebook

from dataset import *


class Model(nn.Module):
    def __init__(self, L1, L2, h):
        super().__init__()

        self.L1 = L1
        self.L2 = L2
        self.h = h

        h = 1

        layers1 = []
        for l in range(self.L1):
            layers1 += [nn.Parameter(torch.randn(self.h, h, 3, 3))]
            h = self.h
        self.layers1 = nn.ParameterList(layers1)

        self.W1 = nn.Parameter(torch.randn(self.h, self.h, 3, 3))
        h = self.h

        h += 3

        layers2 = []
        for l in range(self.L2):
            layers2 += [nn.Parameter(torch.randn(self.h, h, 3, 3))]
            h = self.h
            layers2 += [nn.Parameter(torch.randn(self.h, h, 3, 3))]
            h = self.h
        self.layers2 = nn.ParameterList(layers2)

        self.W2 = nn.Parameter(torch.randn(h))

    def forward(self, vis, jyh):
        z = vis

        for W in self.layers1:
            z = nn.functional.conv2d(z, W / z.size(1) ** 0.5, None, stride=1, padding=1)
            z = z.relu().mul(2 ** 0.5)

        z = nn.functional.conv2d(z, self.W1 / z.size(1) ** 0.5, None, stride=3, padding=0)
        z = z.relu().mul(2 ** 0.5)

        z = torch.cat([z, jyh], 1)

        for l, W in enumerate(self.layers2):
            stride = 1 if l % 2 == 0 else 2
            z = nn.functional.conv2d(z, W / z.size(1) ** 0.5, None, stride=stride, padding=1)
            z = z.relu().mul(2 ** 0.5)

        z = z.mean([2, 3])

        z = z @ (self.W2 / z.size(1) ** 0.5)

        return z


def execute(args):
    dataset = GG2('gg2/', transform=None)

    dataset = [(x, y['n_sources']) for x, y in dataset]
    classes = sorted({y for x, y in dataset})

    sets = [[(x, y) for x, y in dataset if y == i] for i in classes]

    torch.manual_seed(args.data_seed)
    sets = [
        [x[i] for i in torch.randperm(len(x))]
        for x in sets
    ]

    dataset = list(chain(*zip(*sets)))
    dataset = dataset[:args.ptr + args.pte]

    x = [load_GG2_images(x) for x, y in tqdm_notebook(dataset)]
    y = torch.tensor([2 * y - 1 for x, y in dataset], dtype=torch.float32)

    torch.manual_seed(args.init_seed)
    f0 = Model(args.L1, args.L2, args.h).cuda()

    vis = torch.stack([vis for vis, jyh in x]).cuda()
    jyh = torch.stack([jyh for vis, jyh in x]).cuda()

    with torch.no_grad():
        out0 = torch.cat([f0(vis[i:i+args.bs], jyh[i:i+args.bs]) for i in tqdm_notebook(range(0, len(vis), args.bs))])

    f = copy.deepcopy(f0)
    optim = torch.optim.SGD(f.parameters(), lr=args.lr, momentum=args.mom)

    t0 = time.perf_counter()
    dynamics = []

    torch.manual_seed(args.batch_seed)
    for step in count():
        indices = torch.randperm(args.ptr)[:args.bs]

        vis = torch.stack([vis for vis, jyh in (x[i] for i in indices)]).cuda()
        jyh = torch.stack([jyh for vis, jyh in (x[i] for i in indices)]).cuda()

        out = f(vis, jyh) - out0[indices]
        loss = torch.nn.functional.softplus(-args.alpha * out * y[indices].cuda()).mean().div(args.alpha)

        optim.zero_grad()
        loss.backward()
        optim.step()

        dynamics.append({
            'step': step,
            'loss': loss.item(),
        })

        print("[{:d}] [L={:.2f}]".format(step, loss.item()), flush=True)

        if time.perf_counter() - t0 > args.train_time:
            break

    return {
        'f0': f0.state_dict(),
        'f': f.state_dict(),
        'dynamics': dynamics,
    }



def main():
    git = {
        'log': subprocess.getoutput('git log --format="%H" -n 1 -z'),
        'status': subprocess.getoutput('git status -z'),
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--init_seed", type=int, required=True)
    parser.add_argument("--data_seed", type=int, required=True)
    parser.add_argument("--batch_seed", type=int, required=True)

    parser.add_argument("--ptr", type=int, required=True)
    parser.add_argument("--pte", type=int, required=True)

    parser.add_argument("--L1", type=int, required=True)
    parser.add_argument("--L2", type=int, required=True)
    parser.add_argument("--h", type=int, required=True)

    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--bs", type=float, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--mom", type=float, required=True)

    parser.add_argument("--train_time", type=float, required=True)

    parser.add_argument("--pickle", type=str, required=True)
    args = parser.parse_args()

    torch.save(args, args.pickle)
    try:
        for res in execute(args):
            res['git'] = git
            with open(args.pickle, 'wb') as f:
                torch.save(args, f)
                torch.save(res, f)
    except:
        os.remove(args.pickle)
        raise


if __name__ == "__main__":
    main()
