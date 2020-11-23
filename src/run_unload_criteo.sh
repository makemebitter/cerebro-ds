#!/usr/bin/env bash
export EXP_NAME="unload-criteo"
source runner_helper.sh

PRINT_START
psql -a -d cerebro -f unload_criteo.sql
PRINT_END