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


untar("ILSVRC2012_img_train.tar", "ILSVRC2012_img_train")

files = glob.glob("ILSVRC2012_img_train/*.tar")

for f in files:
    prefix = f.split("/")[1].split(".")[0]
    if not os.path.exists("train/" + prefix):
        os.makedirs("train/" + prefix)
    untar(f, "train/" + prefix)

shutil.rmtree("ILSVRC2012_img_train", ignore_errors=True)
