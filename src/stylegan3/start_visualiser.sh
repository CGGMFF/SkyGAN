#!/bin/bash

. prepare_env.inc.sh

python visualizer.py --browse-dir ~/Downloads "$@"
