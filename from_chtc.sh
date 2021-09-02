#!/bin/bash

EXPERIMENT=$(grep -A3 'experiment_name:' config/config.yaml | head -n 1 | sed 's/.*: //')
PROJECT=$(grep -A3 'project' config/config.yaml | head -n 1 | sed 's/.*: //')
cd experiments
mkdir ${PROJECT}
cd ${PROJECT}
mkdir ${EXPERIMENT}
cd ../../
scp amlucas3@submit1.chtc.wisc.edu:/home/amlucas3/loss.png ~/Documents/com/github/segmentation/experiments/${PROJECT}/${EXPERIMENT}
scp amlucas3@submit1.chtc.wisc.edu:/home/amlucas3/unet.h5 ~/Documents/com/github/segmentation/experiments/${PROJECT}/${EXPERIMENT}
scp amlucas3@submit1.chtc.wisc.edu:/home/amlucas3/commit.txt ~/Documents/com/github/segmentation/experiments/${PROJECT}/${EXPERIMENT}