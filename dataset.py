import csv
import glob
import math
import os

import torch
from astropy.io import fits
from six.moves import urllib


def load_GG2_images(images):
    images = [fits.open(file, memmap=False)[0].data for file in images]
    images = [torch.from_numpy(x.byteswap().newbyteorder()) for x in images]

    # normalize the second moment of the channels to 1
    normalize = [3.5239e+10, 1.5327e+09, 1.8903e+09, 1.2963e+09]
    images = [x.mul(n) for x, n in zip(images, normalize)]

    # stack the 3 channels of small resolution together
    vis, j, y, h = images
    return vis[None], torch.stack([j, y, h])


class GG2(torch.utils.data.Dataset):
    url_train = 'http://metcalf1.difa.unibo.it/DATA3/datapack2.0train.tar.gz'
    url_train_log = 'http://metcalf1.difa.unibo.it/DATA3/image_catalog2.0train.csv'


    def __init__(self, root, transform=load_GG2_images, target_transform=None):
        self.root = os.path.expanduser(root)
        self.files = None
        self.data = None
        self.download()
        self.transform = transform
        self.target_transform = target_transform


    def __getitem__(self, index):
        images = self.files[index]
        ID = int(images[0].split('-')[-1].split('.')[0])

        if self.transform:
            images = self.transform(images)

        labels = self.data[ID]

        if self.target_transform:
            labels = self.target_transform(labels)

        return images, labels


    def __len__(self):
        return len(self.files)


    def download(self):
        if not os.path.isdir(self.root):
            os.makedirs(self.root)

        log_path = os.path.join(self.root, "train.csv")
        if not os.path.isfile(log_path):
            print("Download log...", flush=True)
            data = urllib.request.urlopen(self.url_train_log)
            with open(log_path, 'wb') as f:
                f.write(data.read())

        keys = [
            '',              'ID',           'x_crit',            'y_crit',
            'source_ID',     'z_source',     'z_lens',            'mag_source',
            'ein_area',      'n_crit',       'r_source',          'crit_area',
            'n_pix_source',  'source_flux',  'n_pix_lens',        'lens_flux',
            'n_source_im',   'mag_eff',      'sb_contrast',       'color_diff',
            'n_gal_3',       'n_gal_5',      'n_gal_10',          'halo_mass',
            'star_mass',     'mag_lens',     'n_sources'
        ]
        assert len(keys) == 27
        with open(log_path, 'rt') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            data = [x for x in reader if len(x) == 27 and not 'ID' in x]
            data = [{k: float(x) if x else math.nan for k, x in zip(keys, xs)} for xs in data]
            self.data = {x['ID']: x for x in data}

        gz_path = os.path.join(self.root, "datapack2.0train.tar.gz")
        if not os.path.isfile(gz_path):
            print("Download...", flush=True)
            data = urllib.request.urlopen(self.url_train)
            with open(gz_path, 'wb') as f:
                f.write(data.read())

        tar_path = os.path.join(self.root, "datapack2.0train.tar")
        if not os.path.isfile(tar_path):
            print("Decompress...", flush=True)
            import gzip
            import shutil
            with gzip.open(gz_path, 'rb') as f_in:
                with open(tar_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

        dir_path = os.path.join(self.root, "datapack2.0train")
        if not os.path.isdir(dir_path):
            print("Extract...", flush=True)
            import tarfile
            tar = tarfile.open(tar_path)
            tar.extractall(dir_path)
            tar.close()

        self.files = list(zip(*(
            sorted(glob.glob(os.path.join(dir_path, "Public/{}/*.fits".format(band))))
            for band in ("EUC_VIS", "EUC_J", "EUC_Y", "EUC_H")
        )))
        assert all(len({x.split('-')[-1] for x in fs}) == 1 for fs in self.files)
