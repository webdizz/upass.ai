from random import choice
import random
import pandas as pd

MAX_IMGS_PER_IDNT = 50
MAX_PAIRS_FOR_IDNT_PER_CLS = int(MAX_IMGS_PER_IDNT/2)


def prepare_idnt_image_idxs(filtered_df):
    # preapre image registry
    image_registry = {}
    counter = 0
    for idx, row in filtered_df.iterrows():
        if counter > MAX_IMGS_PER_IDNT:
            counter = 0
            continue
        idnt = row['identity']
        if idnt not in image_registry:
            image_registry[idnt] = []
        image_registry[idnt].append(row['img_path'])
        counter = counter+1

    # index identities
    idnt_idxs = []
    for idnt in image_registry:
        idnt_idxs.append(idnt)
    return image_registry, idnt_idxs


def combine_pairs(image_registry, idnt_idxs):

    print('source,target,similarity')
    n_idnts = len(image_registry)
    counter = 0
    idnt_i = 0
    for idnt in image_registry:
        for pair_i in range(MAX_PAIRS_FOR_IDNT_PER_CLS):
            # genuine pairs
            source = image_registry[idnt][pair_i]
            target = image_registry[idnt][MAX_PAIRS_FOR_IDNT_PER_CLS-pair_i]
            print('{},{},genuine'.format(source, target))
            counter = counter+1
            # imposter pairs
            imposter_idnt_idx = random.randint(0, n_idnts-1)
            while imposter_idnt_idx == idnt_i:
                imposter_idnt_idx = random.randint(0, n_idnts-1)

            imposter_idnt = idnt_idxs[imposter_idnt_idx]
            target = choice(image_registry[imposter_idnt])
            print('{},{},imposter'.format(source, target))
            counter = counter+1
        idnt_i = idnt_i+1


train_df = pd.read_csv('dataset/train.csv')
train_df.columns = ['img_path', 'identity', 'width', 'hight']
train_df.describe()

train_counted_df = train_df.groupby('identity')[['identity']].count()
train_counted_df.columns = ['cnt']

train_counted_df = train_counted_df.loc[train_counted_df['cnt']
                                        >= MAX_IMGS_PER_IDNT]

train_counted_df.head()


train_filtered_df = train_df[train_df.identity.isin(train_counted_df.index)]
train_filtered_df.describe()

train_img_reg, train_idnt_idx = prepare_idnt_image_idxs(train_filtered_df)

combine_pairs(train_img_reg, train_idnt_idx)
