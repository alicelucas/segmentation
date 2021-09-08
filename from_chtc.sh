#!/bin/bash

scp amlucas3@submit1.chtc.wisc.edu:/home/amlucas3/$1/config.yaml ~/Documents/com/github/segmentation/tmp/


EXPERIMENT=$(grep -A3 'experiment_name:' tmp/config.yaml | head -n 1 | sed 's/.*: //')
PROJECT=$(grep -A3 'project' tmp/config.yaml | head -n 1 | sed 's/.*: //')
cd experiments
mkdir -p ${PROJECT}
cd ${PROJECT}
mkdir -p ${EXPERIMENT}
cd ../../
scp amlucas3@submit1.chtc.wisc.edu:/home/amlucas3/$1/loss.png ~/Documents/com/github/segmentation/experiments/${PROJECT}/${EXPERIMENT}
scp amlucas3@submit1.chtc.wisc.edu:/home/amlucas3/$1/unet.h5 ~/Documents/com/github/segmentation/experiments/${PROJECT}/${EXPERIMENT}
scp amlucas3@submit1.chtc.wisc.edu:/home/amlucas3/$1/commit.txt ~/Documents/com/github/segmentation/experiments/${PROJECT}/${EXPERIMENT} 
mv tmp/config.yaml ~/Documents/com/github/segmentation/experiments/${PROJECT}/${EXPERIMENT}/
