#!/bin/bash

# install torch with cuda, and requirements.txt
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip3 install -r requirements.txtpy train