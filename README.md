<h1 align="center">Optimized Stable Diffusion</h1>
<p align="center">
    <img src="https://img.shields.io/github/last-commit/mfpousa/stable-diffusion-evolution?logo=Python&logoColor=green&style=for-the-badge"/>
        <img src="https://img.shields.io/github/issues/mfpousa/stable-diffusion-evolution?logo=GitHub&style=for-the-badge"/>
                <img src="https://img.shields.io/github/stars/mfpousa/stable-diffusion-evolution?logo=GitHub&style=for-the-badge"/>
</p>

This repo contains a simplified GUI to easily run and use the stable-diffusion model locally. It includes all the VRAM optimisations developed by basujindal. 

<h1 align="center">Features</h1>

- Generate images from prompts
    - Adjust the <b>batch size</b>: choose how many images will be generated at once
- Start from an existing image and apply a prompt over it
    - Adjust the <b>strength</b>: choose how much the promp will affect the original image
- Use your <b>output images as inputs</b> at the press of a button
- Adjust the <b>quality</b>: choose how much time the model will have to perfect your image
    - 10: <b>good for quick prototyping</b>. <i>tip: generate a bunch of outputs at this quality, then copy the seed of the one you like the most and re-generate at a higher quality level</i> 
    - 20: <b>almost indistinguishable from higher values</b>
    - 50+: <b>use for your final shot</b>


<h1 align="center">Installation</h1>

1. Download the [latest weights](https://huggingface.co/CompVis) ending in <i>-original</i>. The name of the file should be similar to <b>sd-v1-4.ckpt</b>
2. Rename the file to model.ckpt
3. Place the model.ckpt file into models/ldm/stable-diffusion-v1
1. Install miniconda3
2. Open your terminal in the root of the project and run ```conda env create -f environment.yaml```
3. Bootstrap the application with ```python webui/start.py```
5. Profit!