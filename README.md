
# CartoonGAN

Torch implementation for [Yang Chen, Yu-Kun Lai, Yong-Jin Liu. CartoonGAN: Generative Adversarial Networks for Photo Cartoonization](http://cg.cs.tsinghua.edu.cn/people/~Yongjin/CartoonGan-CVPR2018.pdf), CVPR2018

This code borrows from early version of [CycleGAN](https://github.com/junyanz/CycleGAN).

## Installation
The same as [CycleGAN](https://github.com/junyanz/CycleGAN)
- Install torch and dependencies from https://github.com/torch/distro
- Install torch packages `nngraph`, `class`, `display`
```bash
luarocks install nngraph
luarocks install class
luarocks install https://raw.githubusercontent.com/szym/display/master/display-scm-0.rockspec
```
- Clone this repo:
```bash
git clone https://github.com/FlyingGoblin/CartoonGAN.git
cd CartoonGAN
```

## Apply pre-trained Model
- Download the pre-trained model:
```
bash ./pretrained_models/download_model.sh
```
- Generate Shinkai style images:
```
DATA_ROOT=./datasets/<your test data> name=Shinkai model=one_direction_test phase=test loadSize=256 fineSize=256 resize_or_crop="scale_width" th test.lua
```

## Dataset
- For the photo:
All the images in our paper are from CycleGAN: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix (XXX2photo).
- For the Cartoon image:
Because all the cartoon images are from films, I don't know if I have the right to upload them. I don't know if there will be any copyright problem, I am sorry.
    - I just calculate the SSIM and PSNR between frames to remove too samiliar images.
    - The cartoon dataset should not be too small.


## Train
- Dataset:
Prepare your datasets in ./datasets/<your data>, it should have subfolders `train_A`, `train_B`, `train_B_edge`
- Train a model with init process:
```bash
DATA_ROOT=./datasets/<your data> name=<your name> th train.lua
```
- Train a model with an init model:
```bash
DATA_ROOT=./datasets/<your data> name=<your name> init_model=<path to your init model> th train.lua
```

## Test
- Dataset:
Prepare your datasets in ./datasets/<your test data>, it should have subfolders `test_A`
```bash
DATA_ROOT=./datasets/<your test data> name=<your name> model=one_direction_test phase=test loadSize=256 fineSize=256 resize_or_crop="scale_width" th test.lua
```

## BlaBla
- My email: chenyang15@mails.tsinghua.edu.cn can NOT send emails because of my graduation, and it won't be able to receive email very soon (maybe I have already missed some emails). So please contact me by cylily93@gmail.com.
- In fact I didn't mean to clean up the code before because of laziness  `_(:з」∠)_`, but after receiving several contact asking me about the reproduction, I decided to clean them up to avoid "rebuilding the wheel".
- I hope this code can speed up your work ~
- 发现我这个作者的代码的星星在GitHub里CartoonGAN相关项目里才排第5啊`_(:з」∠)_`，而且和前面的差好远`_(:з」∠)_`我唯一一个有可能有星星的项目啊`_(:з」∠)_`哭`o(╥﹏╥)o`

