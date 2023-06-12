# EDVR_3Dconv
**EDVR with additional 3D conv. module for synthetic reflection removal.**  
This repo has been based on [EDVR](<https://github.com/xinntao/EDVR>) and [BasicSR](<https://github.com/XPixelGroup/BasicSR>).

# Installation
1. Requirements
- Python >= 3.7
- PyTorch >= 1.7
- NVIDIA GPU + CUDA
<br>

2. Install dependent packages

     ```bash
    cd EDVR_3Dconv
    pip install -r requirements.txt
    ```  
    <br>

3. Install BasicSR

    ```bash
    BASICSR_EXT=True python setup.py develop
    ```
    (Details at [EDVR](<https://github.com/xinntao/EDVR>) and [BasicSR Installation](<https://github.com/XPixelGroup/BasicSR/blob/master/docs/INSTALL.md>))


# Datasets

1. Prepare REDS dataset

    Download [REDS dataset](<https://seungjunnah.github.io/Datasets/reds.html>) and put them in `datasets/REDS_dataset/train_shirp`.
    <br>


2. Create synthetic reflection dataset

    ```bash
    python datasets/create_synvideo_train.py
    python datasets/create_synvideo_val.py
    python datasets/create_synvideo_test.py
    ```

# Testing

1. Configuration

    Edit ```options/test/EDVR/test_EDVR_M_RR_REDS.yml``` (Details at [Config.md](<https://github.com/XPixelGroup/BasicSR/blob/master/docs/Config.md>))
    <br>
    
    
2. Testing

    ```bash
    python basicsr/test.py -opt options/test/EDVR/test_EDVR_M_RR_REDS.yml
    ```
    (Details at [TrainTest.md](<https://github.com/XPixelGroup/BasicSR/blob/master/docs/TrainTest.md>))


# Training

1. Configuration

    Edit ```options/train/EDVR/train_EDVR_M_RR_REDS.yml``` (Details at [Config.md](<https://github.com/XPixelGroup/BasicSR/blob/master/docs/Config.md>))
    <br>


2. Training

    ```bash
    python basicsr/train.py -opt options/train/EDVR/train_EDVR_M_RR_REDS.yml
    ```
    (Details at [TrainTest.md](<https://github.com/XPixelGroup/BasicSR/blob/master/docs/TrainTest.md>))

# Results

<img src="https://github.com/stmrym/EDVR_3Dconv/assets/114562027/14b826e4-bca1-417d-a130-6cc62ba5f5e5" wddth="400">

![EDVR](https://github.com/stmrym/EDVR_3Dconv/assets/114562027/d237ed03-8383-482e-8d39-378361e0bec5)

![P3DC](https://github.com/stmrym/EDVR_3Dconv/assets/114562027/7949cea8-9b3e-4c43-bac9-3cf389fe8363)

# License

[Apache 2.0 license](<https://github.com/stmrym/EDVR_3Dconv/blob/main/LICENSE>)
