# Blood vessel salience quantification and augmentation

Source code for the paper **Evaluation of Blood Vessel Segmentation Methods on Hard-to-Detect Vascular Structures** by João Pedro Parella, Matheus Viana da Silva and Cesar Henrique Comin.

The script `vessel_salience/salience.py` can be used for calculating the local vessel salience (LVS) index, low-salience recall (LSRecall) and mean LSRecall (mLSR) using the API functions `lvs`, `ls_recall` and `average_ls_recall`, respectively.

The script `vessel_salience/augmentation.py` can be used for augmenting blood vessel segments using the function `create_image`. 

Required packages are indicated in the file `requirements.txt`.

The Jupyter notebooks in the folder `notebooks` show examples for running the scripts. 