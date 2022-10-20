#!/usr/bin/env python

import glob
import tarfile
import os
import shutil


def untar(fname, targetd_dir):
    """

    :param fname:
    :param targetd_dir:
    """
    with tarfile.open(fname) as tar:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar, path=targetd_dir)


untar("ILSVRC2012_img_val.tar", "valid_temp")

mapping_file = open("ILSVRC2012_mapping.txt")
valid_files = open("ILSVRC2012_valid_ground_truth.txt")

mappings = {}

count = 0
for line in mapping_file.readlines():
    x = line.strip()
    mappings[str(count)] = x
    count += 1

labels = {}
for line in valid_files.readlines():
    temp = line.strip()
    if temp != "":
        x, y = temp.split(" ")
        labels[x] = mappings[y]

for fname in os.listdir("valid_temp"):
    if fname.endswith(".JPEG"):
        if not os.path.exists("valid/" + labels[fname]):
            os.makedirs("valid/" + labels[fname])

        shutil.copyfile("valid_temp/" + fname, "valid/" + labels[fname] + "/" + fname)

shutil.rmtree("valid_temp", ignore_errors=True)
