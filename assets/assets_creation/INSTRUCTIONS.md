# Create assets using SAM3D-Objects

# 1. Install SAM3D objects on a node with sufficient compute (>32GB VRAM)
use the fork with `multidemo.py` : https://github.com/RaphaelLorenzo/sam-3d-objects/

Follow the instructions of https://github.com/facebookresearch/sam-3d-objects/blob/main/doc/setup.md

## Install on cluster (with no internet access on nodes)
To install on Jean-Zay you may use the `prepost` partition that has internet access. 
Start by following the procedure to install using mamba or conda.

Then following ideas of [this](https://github.com/lapertme2/sam-3d-objects-win/blob/main/JUST-RUN-DEMO.md) regarding the issue of downloading MoGE checkpoint :
```
wget https://huggingface.co/Ruicheng/moge-vitl/resolve/main/model.pt
mv model.pt checkpoints/hf
```
Then update config with : `vi checkpoints/hf/pipeline.yaml` change the MoGE config to : `pretrained_model_name_or_path: checkpoints/hf/model.pt`

Also do a first pass of `demo.py` to download DinoV2 checkpoints (will fail due to insufficient VRAM).

# 2. Run the demo on you image and masks
Create your inputs masks using SAM2 script :
```
python assets/assets_creation/sam2_segment.py ./assets/assets_creation/inputs/AssetsImageGlob1.jpeg
```

Move image to the node
```
scp -r ./assets/assets_creation/inputs/ jean-zay:/lustre/fswork/projects/rech/lfh/ufg41mh/Projects/github/sam-3d-objects/
```

Connect to a node : `srun ...`
Enable `conda` and `conda activate sam3d-objects`

Use the `multidemo.py` with your image and outputs.
```
python multidemo.py --image_path ./inputs/AssetsImageGlob1.jpeg --masks_path ./inputs/AssetsImageGlob1/
```

Move back results to your local computer
```
scp -r jean-zay:/lustre/fswork/projects/rech/lfh/ufg41mh/Projects/github/sam-3d-objects/output/  ./assets/assets_creation/
```