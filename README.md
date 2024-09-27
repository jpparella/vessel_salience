# Blood vessel salience quantification and augmentation

Source code for the paper **A New Approach for Evaluating and Improving the Performance of Segmentation Algorithms on Hard-to-Detect Blood Vessels** by João Pedro Parella, Matheus Viana da Silva and Cesar Henrique Comin.

The script `vessel_salience/salience.py` can be used for calculating the local vessel salience (LVS) index and the low-salience recall (LSRecall) using the API functions `lvs` and `ls_recall`.

The script `vessel_salience/augmentation.py` can be used for augmenting blood vessel segments using the function `create_image`. 

Required packages are indicated in the file `requirements.txt`.

The Jupyter notebooks in the folder `notebooks` show examples for running the scripts. They are still a work in progress but show basic functionality.