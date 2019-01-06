from torchvision.transforms import *
import PIL
import os
import pandas as pd
from fastai.vision import *
from fastai import *


__all__ = ['compile_faces_dataset', 'create_bunch', 'TwinDataset']


def _select_and_count_only_images(path):
    identity_images = [f for f in os.listdir(path) if '.jpg' in f]
    return identity_images, len(identity_images)


def compile_faces_dataset(path, pairs=10, folder='valid', qty=500):
    from random import choice

    df = pd.DataFrame(columns=['source', 'target', 'similarity'])

    face_identities = [d for d in os.listdir(path+'/'+folder) if 'n' in d]

    faces_num = len(face_identities)
    counter = 0

    image_registry = {}

    for i in range(0, faces_num):
        # cache images for later speedup
        if face_identities[i] not in image_registry:
            src_id_imgs, src_id_imgs_num = _select_and_count_only_images(path+'/%s/%s' %
                                                                         (folder, face_identities[i]))

            image_registry[face_identities[i]] = src_id_imgs, src_id_imgs_num
        else:
            src_id_imgs, src_id_imgs_num = image_registry[face_identities[i]]

        for ii in range(0, pairs):
            if qty > 0 and counter >= qty:
                return df

            base_path = '%s/%s' % (folder, face_identities[i])
            source = '%s/%s' % (base_path, src_id_imgs[ii])
            target = '%s/%s' % (base_path, src_id_imgs[src_id_imgs_num-ii-1])
            df.loc[counter] = [source, target, 'genuine']
            counter = counter+1

            # get another identities
            trg_id = faces_num-i-ii if faces_num-i-ii <= 0 else faces_num-ii-1

            # cache images for later speedup
            if face_identities[trg_id] not in image_registry:
                trg_id_imgs, trg_id_imgs_num = _select_and_count_only_images(path+'/%s/%s' %
                                                                             (folder, face_identities[trg_id]))

                image_registry[face_identities[trg_id]
                               ] = trg_id_imgs, trg_id_imgs_num
            else:
                trg_id_imgs, trg_id_imgs_num = image_registry[face_identities[trg_id]]

            target_img = choice(trg_id_imgs)
            trg_base_path = '%s/%s' % (folder, face_identities[trg_id])
            target = '%s/%s' % (trg_base_path, target_img)
            df.loc[counter] = [source, target, 'imposter']
            counter = counter+1

    return df


def create_bunch(df, cols, path: PathOrStr = '.', tfms=None, bs=1):
    return (CustomImageItemList.from_df(df, path=path, cols=cols)
            .no_split()
            .label_from_df(cols='similarity')
            # .transform(get_transforms(), tfm_y=True)
            .databunch(bs=bs, tfms=tfms)
            )


class CustomImageItemList(ImageItemList):

    def _resizeable_create_func(self, fn: PathOrStr):
        normalize = Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])

        transform = Compose([
            RandomGrayscale(),
            Resize(224),
            ToTensor(),
            normalize,
        ])

        x = PIL.Image.open(fn).convert('RGB')
        x = transform(x)
        x = Image(x).resize(224)

        return x

    def open(self, fn): return self._resizeable_create_func(fn)


class TwinDataset(Dataset):

    def __init__(self, source: Dataset, target: Dataset):
        self.source = source
        self.target = target
        self.c = source.c

    def __getitem__(self, idx):
        src = self.source[idx]
        trg = self.target[idx]

        return [src[0], trg[0]], src[1]

    def __len__(self):
        return len(self.source)
