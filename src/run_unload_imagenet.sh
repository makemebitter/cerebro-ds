#!/usr/bin/env bash
export EXP_NAME="unload-imagenet"
source runner_helper.sh

PRINT_START
psql -a -d cerebro -f unload_imagenet.sql
PRINT_END