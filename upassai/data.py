import os
import pandas as pd


__all__ = ['compile_faces_dataset']


def select_and_count_only_images(path):
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
            src_id_imgs, src_id_imgs_num = select_and_count_only_images(path+'/%s/%s' %
                                                                        (folder, face_identities[i]))

            image_registry[face_identities[i]] = src_id_imgs, src_id_imgs_num
        else:
            src_id_imgs, src_id_imgs_num = image_registry[face_identities[i]]

        for ii in range(0, pairs):
            if counter >= qty:
                return df

            base_path = '%s/%s' % (folder, face_identities[i])
            source = '%s/%s' % (base_path, src_id_imgs[ii])
            target = '%s/%s' % (base_path, src_id_imgs[src_id_imgs_num-ii-1])
            df.loc[counter] = [source, target, 0]
            counter = counter+1

            # get another identities
            trg_id = faces_num-i-ii if faces_num-i-ii <= 0 else faces_num-ii-1

            # cache images for later speedup
            if face_identities[trg_id] not in image_registry:
                trg_id_imgs, trg_id_imgs_num = select_and_count_only_images(path+'/%s/%s' %
                                                                            (folder, face_identities[trg_id]))

                image_registry[face_identities[trg_id]
                               ] = trg_id_imgs, trg_id_imgs_num
            else:
                trg_id_imgs, trg_id_imgs_num = image_registry[face_identities[trg_id]]

            target_img = choice(trg_id_imgs)
            trg_base_path = '%s/%s' % (folder, face_identities[trg_id])
            target = '%s/%s' % (trg_base_path, target_img)
            df.loc[counter] = [source, target, 1]
            counter = counter+1

    return df
