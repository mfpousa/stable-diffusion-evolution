import numpy as np
import os
from PIL import Image
import numpy as np
from PIL import Image
from torchvision.transforms import transforms

C = 4
f = 8
Height = 512
Width = 512
n_iter = 1
ddim_eta = 0
scale = 7.5
unet_bs = 1
device = "cuda"
full_precision = False
outdir = os.path.join("output")
fallback_image = Image.fromarray(255 * np.ones((Height, Width, 3), np.uint8))

ToTensor = transforms.ToTensor()
ToPIL = transforms.ToPILImage()
