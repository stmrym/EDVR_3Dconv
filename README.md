# EDVR_3Dconv
**EDVR with additional 3D conv. module for synthetic reflection removal.**  
This repo has been based on [EDVR](<https://github.com/xinntao/EDVR>) and [BasicSR](<https://github.com/XPixelGroup/BasicSR>).

# Installation
1. Requirements
- Python >= 3.7
- PyTorch >= 1.7
- NVIDIA GPU + CUDA

2. Install dependent packages

     ```bash
    cd EDVR_3Dconv
    pip install -r requirements.txt
    ```

3. Install BasicSR

    ```bash
    BASICSR_EXT=True python setup.py develop
    ```
    (Details at [EDVR](<https://github.com/xinntao/EDVR>) and [BasicSR Installation](<https://github.com/XPixelGroup/BasicSR/blob/master/docs/INSTALL.md>))


# Datasets

1. Prepare REDS dataset

    Download [REDS dataset](<https://seungjunnah.github.io/Datasets/reds.html>) and put them in `datasets/REDS_dataset/train_shirp`.

2. Create synthetic reflection dataset

    ```bash
    python datasets/create_synvideo_train.py
    python datasets/create_synvideo_val.py
    python datasets/create_synvideo_test.py
    ```

# Testing

1. Configuration

    Edit ```options/test/EDVR/test_EDVR_M_RR_REDS.yml``` (Details at [Config.md](<https://github.com/XPixelGroup/BasicSR/blob/master/docs/Config.md>))

2. Testing

    ```bash
    python basicsr/test.py -opt options/test/EDVR/test_EDVR_M_RR_REDS.yml
    ```
    (Details at [TrainTest.md](<https://github.com/XPixelGroup/BasicSR/blob/master/docs/TrainTest.md>))


# Training

1. Configuration

    Edit ```options/train/EDVR/train_EDVR_M_RR_REDS.yml``` (Details at [Config.md](<https://github.com/XPixelGroup/BasicSR/blob/master/docs/Config.md>))

2. Training

    ```bash
    python basicsr/train.py -opt options/train/EDVR/train_EDVR_M_RR_REDS.yml
    ```
    (Details at [TrainTest.md](<https://github.com/XPixelGroup/BasicSR/blob/master/docs/TrainTest.md>))

# License

[Apache 2.0 license](<https://github.com/stmrym/EDVR_3Dconv/blob/main/LICENSE>)
