# Objects' State transition for action recognition:

This repo is related to the paper state-transformation for manipulation action recognition (under single-blind review)

## Prerequisites: 
* Pytorch (v. 1.0.1)
* python (v. 3.6)
* epic-kitchens (v. 1.6.2)
* tqdm
* Numpy

## Getting started: 
* clone this repo. 
 ```
 git clone https://github.com/Nachwa/object_states.git
 ```
* Frames extracted as in (provided starter kit: https://github.com/epic-kitchens/starter-kit-action-recognition#data-pipeline-snakemake) 

## For training: 
```
python train_model.py --logname EXPNAME --batch_size 32 --nb_keyframes 5 --db_dir PATH_TO_DB_ROOT
```

