# PD-SAM2
## Installation
```
conda create --name pdsam2 python=3.10
conda activate pdsam2
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install requirements.txt
```

## Usage
### prepare dataset
First, download the dataset from:
- [CAMUS](https://www.creatis.insa-lyon.fr/Challenge/camus/index.html)
- [EchoNet-Dynamic](https://echonet.github.io/dynamic/index.html)
  
Then process the dataset according to `utils/preprocess_echonet.py` and `utils/preprocess_camus.py`

for example:

```
# CAMUS
python utils/preprocess_camus.py -i /data/CAMUS_public/database_nifti -o /data/CAMUS_public

# EchoNet-Dynamic
python utils/preprocess_echonet.py -i /data/EchoNet-Dynamic -o /data/EchoNet
```

### pretrain checkpoint download
[ViT-L SAM2 model](https://drive.google.com/file/d/1eDbeBrL9Cg7rOW_2I6CabB22FqibGLWS/view?usp=drive_link)

### train and test
Use `train_video.py` and `test_video.py` to train and test separately.

## Acknowledgement
The work is based on [SAM2](https://github.com/facebookresearch/sam2)
