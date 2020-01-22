# VQ-VAE for Acoustic Unit Discovery and Voice Conversion

Train and evaluate models for the ZeroSpeech challenges.
Voice conversion samples can be found here.

<p align="center">
  <img width="384" height="563" alt="VQ-VAE for Acoustic Unit Discovery"
    src="https://raw.githubusercontent.com/bshall/ZeroSpeech/master/network.png">
</p>

## Quick Start

1. Ensure you have Python 3 and PyTorch 1.3 or greater.

2. Clone the repo:
    ```
    git clone https://github.com/bshall/ZeroSpeech
    cd ./ZeroSpeech
    ```

3. Install requirements:
    ```
    pip install requirements.txt

    ```
    
4. Install [NVIDIA/apex](https://github.com/NVIDIA/apex) for mixed precision training.
    
5. Download and extract ZeroSpeech2019: TTS without the T English dataset:
    ```
    wget https://download.zerospeech.com/2019/english.tgz
    tar -xvzf english.tgz
    ```
    
6. Extract train/test Mel spectrograms and preprocess audio:
    ```
    python preprocess.py --in-dir=/path/to/ZeroSpeech2019 --out-dir=datasets/ZeroSpeech2019 --split-path=datasets/ZeroSpeech2019/english/train.csv
    ```
    
7. Train the model:
    ```
    python train.py --checkpoint-dir=checkpoints/ZeroSpeech2019/english --data-dir=datasets/ZeroSpeech2019/english
    ```
    
8. Voice conversion:
    ```
    python convert.py --checkpoint=path/to/checkpoint --data-dir=datasets/ZeroSpeech2019/english --out-dir=converted/ZeroSpeech2019/english --synthesis-list=datasets/ZeroSpeech2019/english/synthesis.csv
    ```
    
9. Encode test data for evaluation:
    ```
    python encode.py --checkpoint=path/to/checkpoint --in-dir=datasets/ZeroSpeech2019/english/test --out-dir=encoded/ZeroSpeech2019/english/test
    ```
    
10. Evaluate ABX:
    ```
    python abx.py --task-type=across --task-path=path/to/abxtask --feature-dir=encoded/ZeroSpeech2019/english/test --out-dir=abx/ZeroSpeech2019/test
    ```