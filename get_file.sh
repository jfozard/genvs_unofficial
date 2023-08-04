

pip install gdown
gdown --fuzzy https://drive.google.com/file/d/1bThUNtIHx4xEQyffVBSf82ABDDh2HlFn/view?usp=drive_link

mv cars_train.zip data
cd data
unzip -q cars_train.zip 
cd ..

sudo apt-get update
sudo apt-get install -y git-lfs

git lfs install
git clone https://huggingface.co/JFoz/test_nvs

pip install accelerate clean-fid clip-anytorch einops jsonmerge kornia Pillow resize-right scikit-image scipy torchdiffeq torchsde torchvision tqdm wandb protobuf==3.20 wandb einops fastai protobuf==3.20 omegaconf hydra-core
