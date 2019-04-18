# Debs 2019

### File Directory Structure
----------------------------

```buildoutcfg
dataset
|   +-- train
|   +-- test
|
ssh-kd
+-- data
+-- model
+-- plugin
|   +-- encode.py
|   +-- load_model.py
|   +-- model.py
|   +-- pred.py
|   +-- seg.py
+-- utils
|   +-- data_prep.py
|   +-- viz.py
+-- train.py
+-- test.py
+-- client.py
```

The Above directory is used for the project.
- data: contains all the .npy files which are generated in data preparation stage.
- model: contains the save model after executing training of model
- plugin: contains the procedural modules  
    - load_model: module to test with the trained model
    - pred: module to predict each scene with segmentation and classification 
    - encode: module to encode data before feeding to model
    - seg: module to perform segmentation
    - model: module for the classification model
- utils: contains the utility modules
    - data_prep: module to perform data preparation

Note: Place the dataset in the `dataset` directory.  
    Train input datasets in `dataset/train`. Example: Atm, Bench, and etc.  
    Test input datasets in `dataset/test`. Example: Set1 and Set2    

### How to Run?
---------------

- Prepare the data for training the model.
    - Features: Removing ground and noise  
    - CMD: `python utils/data-prep.py`
   
- Train the model.
    - CMD: `python train.py`
    
- Test the model locally.
    - CMD: `python test.py`