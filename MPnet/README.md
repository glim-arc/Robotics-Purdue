Install:

pip install torch==1.10.2+cu118 torchvision==0.11.3+cu118 -f https://download.pytorch.org/whl/torch_stable.html

pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116

How to run:

Create a folder called data and put all the env folders and obstacle files in the data folder.

2d test:

NMP: python mpnet test . py  −−N 100 −−NP 100 −−s 0 −−sp 4001 --epoch 100 −−env−type ’s2d’
NMP dropout: python mpnet test . py  −−N 100 −−NP 100 −−s 0 −−sp 4001 --epoch 100 −−env−type ’s2d’ --drop False
NMP lvc: python mpnet test . py  −−N 100 −−NP 100 −−s 0 −−sp 4001 --epoch 100 −−env−type ’s2d’ --lvc False

3d test:

NMP: python mpnet test . py  −−N 100 −−NP 100 −−s 0 −−sp 2001 --epoch 500 −−env−type ’s3d’
NMP dropout: python mpnet test . py  −−N 100 −−NP 100 −−s 0 −−sp 2001 --epoch 500 −−env−type ’s3d’ --drop False
NMP lvc: python mpnet test . py  −−N 100 −−NP 100 −−s 0 −−sp 2001 --epoch 500 −−env−type ’s3d’ --lvc False

Train:

python mpnet test . py

python mpnet test . py  −−N 100 −−NP 2000 --epoch 500 −−env−type ’s3d’  --batch-size 100  --learning-rate 0.01

Current setting is for the 3d model test.
Recommend to edit the bottom of the default args section before the run

Visualizer:

Recommend to edit the bottom of the default args section to the target paths to visualize.

_expert means the comparison with rrt
_5 means printing 5 paths
_3d means to visualize the 3d paths

------------------------------------------------------------------------------------------

PyTorch implementation of MPnet with a pretrained model for a 2D planner.
Code has been taken from GitHub (https://github.com/MiaoDragon/MPNet-hw) and lightly modified for the purposes of this assignment.

* mpnet_test.py:
    * for generating path plan using existing models.
    * model_path: folder of trained model (model name by default is mpnet_epoch_[number].pkl)
    * N: number of environment to test on
    * NP: number of paths to test on
    * s: start of environment index
    * sp: start of path index
    * data_path: the folder of the data
    * result_path: the folder where results are stored (each path is stored in a different folder of environment)

* visualizer.py
    * obs_file: path of the obstacle point cloud file
    * path_file: path to the stored planned path file

* mpnet_Train.py:
    * for training the model.

* To run: create results, models, and data folder, and put the data into data folder. Execute mpnet_test.py to generate plan.
* Tested with python3.8
# Gyubeum Lim
