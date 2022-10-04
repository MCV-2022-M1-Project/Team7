# Team7
## Setup 
1. Create a new Conda environment.
```
conda create -n "mcv-m1" python=3.8
conda activate mcv-m1
```
2. Install requirements:
```
pip install -r requirements.txt
```
The dataset can be found at [this](https://drive.google.com/drive/folders/1wKJYx0Dc8KpFrFfejYnSOd1nVqs2ss7z?usp=sharing) folder.

3. To obtain segmentation performance:
  a. change the test_data_directory and in the test data directory provide the ground truth in gt folder
  b. save the results of the algorithm1 in rs1 folder
  c. save the results of the algorithm2 in rs2 folder
'''

python ./src/segmentation_eval_demo.py
'''
