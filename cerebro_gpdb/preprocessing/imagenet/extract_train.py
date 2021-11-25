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
        tar.extractall(path=targetd_dir)


untar("ILSVRC2012_img_train.tar", "ILSVRC2012_img_train")

files = glob.glob("ILSVRC2012_img_train/*.tar")

for f in files:
    prefix = f.split("/")[1].split(".")[0]
    if not os.path.exists("train/" + prefix):
        os.makedirs("train/" + prefix)
    untar(f, "train/" + prefix)

shutil.rmtree("ILSVRC2012_img_train", ignore_errors=True)
