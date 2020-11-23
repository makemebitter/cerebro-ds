import os
import random
import sys

import h5py
import numpy as np

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise Exception("An image directory must be specified...")
    input_path = sys.argv[1]

    file_paths = [];
    labels = []
    label_map = {}
    label_count = 0
    for label in os.listdir(input_path + "/"):
        if label.startswith("n"):
            if label in label_map:
                label_index = label_map[label]
            else:
                label_map[label] = label_count
                label_index = label_count
                label_count += 1

            for f_name in os.listdir(input_path + "/" + label):
                if f_name.endswith("JPEG"):
                    file_path = input_path + "/" + label + "/" + f_name
                    file_paths.append(file_path)
                    labels.append(label_index)

    data = zip(file_paths, labels)
    random.shuffle(data)

    h5f = h5py.File(input_path + ".h5", "w")

    image_dtype = h5py.special_dtype(vlen=np.dtype('uint8'))

    images = h5f.create_dataset('images', (len(data),), dtype=image_dtype)
    labels = h5f.create_dataset('labels', (len(data),))

    for i, (file_path, label) in enumerate(data):
        with open(file_path, mode='rb') as file:
            images[i] = np.fromstring(file.read(), dtype=np.uint8)
        labels[i] = label

    h5f.close()


# from PIL import Image
# import io
# import h5py
# import sys
# import numpy as np
#
# file_paths = [
#     '/data/imagenet/train/train_0.h5',
#     '/data/imagenet/train/train_1.h5',
#     '/data/imagenet/train/train_2.h5',
#     '/data/imagenet/train/train_3.h5',
#     '/data/imagenet/train/train_4.h5',
#     '/data/imagenet/train/train_5.h5',
#     '/data/imagenet/train/train_6.h5',
#     '/data/imagenet/train/train_7.h5',
# ]
#
# def transform_data(i):
#     file_path = file_paths[i]
#     h5f_in = h5py.File(file_path, 'r')
#     images = h5f_in.get("images")
#     labels = h5f_in.get("labels")[:]
#     partition_size = images.shape[0]
#
#     np_images = np.zeros((partition_size, 112, 112, 3), dtype=np.float32)
#
#     for i in range(partition_size):
#         image = np.asarray(Image.open(io.BytesIO(images[i].tostring())).convert('RGB').resize((112, 112)))
#         image = image / 255.0
#         image = image - [0.485, 0.456, 0.406]
#         image = image / [0.229, 0.224, 0.225]
#         np_images[i] = image
#
#     h5f_in.close()
#     h5f_out = h5py.File(file_path, "w")
#     h5f_out.create_dataset('images', data=np_images)
#     h5f_out.create_dataset('labels', data=labels)
#     h5f_out.close()
#
# transform_data(int(sys.argv[1]))