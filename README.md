# Team7
## Setup 
- 1. Create a new Conda environment.
```
conda create -n "mcv-m1" python=3.8
conda activate mcv-m1
```
- 2. Install requirements:
```
pip install -r requirements.txt
```
The dataset can be found at [this](https://drive.google.com/drive/folders/1wKJYx0Dc8KpFrFfejYnSOd1nVqs2ss7z?usp=sharing) folder.

### Project structure
```
config/
    masking.yaml
    ...

datasets/
    museum/
        bbdd_00000.jpg
        bbdd_00000.png
        bbdd_00000.txt
        ...
    qsd1_w1/
        00000.jpg
        ...
    qsd2_w1/
        00000.jpg
        00000.png
        ...
src/
    common/
        ...
    datasets/
        ...
    metrics/
        ...
    preprocessing/
        ...
    tasks/
        ...
    ...

tools/
    ...
```

- 3. Run the pipeline:
```
python main.py
```

- 3.1 Run the pipeline:

Use config files "config/*.yaml" in order to test a certain pipeline.
You can change there which objects to use in the pipeline by properly feeding the config file.
```
python main.py --config ./config/[TASK_HERE].yaml
```
