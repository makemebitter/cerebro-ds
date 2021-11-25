from __future__ import print_function, division
import random


if __name__ == "__main__":

    random.seed(2018)

    valid_fraction = 0.1

    train_file = open('train.tsv', 'w')
    valid_file = open('valid.tsv', 'w')
    file_names = ['day_1', 'day_2', 'day_3']
    for file_name in file_names:
        for line in open(file_name):
            if random.random() < valid_fraction:
                valid_file.write(line)
            else:
                train_file.write(line)

        train_file.flush()
        valid_file.flush()

    train_file.close()
    valid_file.close()