from PIL import Image
import os
import sys

def iterate_over_folders(dataset_folder: str = '../data/vggface2', data_folder: str = 'train', start='n000002', identities: int = 6000, img_width_min: int = 200, img_width_max: int = 300):
    "Helps to collect images of similar size across folders to simplify images collection + speedup process of training data preprocessing"
    data_list_file = '%s_list.txt' % data_folder
    counter = 0
    current_identity = start
    with open(dataset_folder + '/' + data_list_file, 'r') as file:

        for line in file:
            face_img_path = line.strip()
            image_identity, image_name = face_img_path.split('/')

            # let's start processing
            if image_identity == start:
                counter = 1

            if counter > 0:
                if current_identity != image_identity:
                    counter = counter+1
                    current_identity = image_identity
                # stop when we reach interesting limit
                if counter >= identities:
                    return
                # let's read image
                im = Image.open(dataset_folder + '/' +
                                data_folder + '/' + face_img_path)
                width, height = im.size
                if width >= img_width_min and width <= img_width_max:
                    print(
                        ','.join((face_img_path, image_identity, str(width), str(height))))

# Valid iterations
# iterate_over_folders(data_folder='test', start='n000001')

# Train iterations
# iterate_over_folders(start='n001056')
# iterate_over_folders(start='n000067')
# iterate_over_folders(start='n003212')