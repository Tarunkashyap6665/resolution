import os.path
import torch
# from utils import utils_model
from utils import utils_image as util
from models.network_rrdbnet import RRDBNet as net
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image



# def enhance(img_L):

#   img_L=cv2.cvtColor(img_L,cv2.COLOR_BGR2RGB)
  
#   model_name = 'BSRGAN'   # 'BSRGANx2' for scale factor 2
#   sf = 4
#   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#   model_path = os.path.join('model_zoo', model_name+'.pth')       
#   torch.cuda.empty_cache()
#   # --------------------------------
#   # define network and load model
#   # --------------------------------
#   model = net(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=sf)  # define network
#   model.load_state_dict(torch.load(model_path), strict=True)
#   model.eval()
#   for k, v in model.named_parameters():
#       v.requires_grad = False
#   model = model.to(device)
#   torch.cuda.empty_cache()
#   # --------------------------------
#   # (1) img_L
#   # --------------------------------
#   trans=transforms.ToTensor()
#   img_L=trans(img_L)

#   # img_L=torch.from_numpy(img_L)
#   # transform=transforms.ToTensor()
#   img_L=img_L.type(torch.float32)
#   img_L = img_L.to(device)
#   img_L=img_L.unsqueeze(dim=0)


#   # --------------------------------
#   # (2) inference
#   # --------------------------------
#   img_E = model(img_L)
#   return img_E

def enhance(img_path):
  # # Ensure the image has 4 channels
  # if pil_image.mode != 'RGB':
  #     pil_image = pil_image.convert('RGBA')
  #     # Split the channels
  #     r, g, b, a = pil_image.split()
  #     # Concatenate RGB channels
  #     img_L = Image.merge('RGB', (r, g, b))
  # else:
  #     img_L=pil_image  
  
  model_name = 'BSRGAN'   # 'BSRGANx2' for scale factor 2
  sf = 4
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  model_path = os.path.join('model_zoo', model_name+'.pth')       
  torch.cuda.empty_cache()
  # --------------------------------
  # define network and load model
  # --------------------------------
  model = net(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=sf)  # define network
  model.load_state_dict(torch.load(model_path), strict=True)
  model.eval()
  for k, v in model.named_parameters():
      v.requires_grad = False
  model = model.to(device)
  torch.cuda.empty_cache()
  # --------------------------------
  # (1) img_L
  # --------------------------------

  img_L = util.imread_uint(img_path, n_channels=3)
  img_L = util.uint2tensor4(img_L)

  # trans=transforms.ToTensor()
  # img_L=trans(img_L)
  # print(img_L.dtype,img_L.shape)
  img_L = img_L.to(device)
  # img_L=img_L.unsqueeze(dim=0)
  print(img_L.dtype,img_L.shape)

  # --------------------------------
  # (2) inference
  # --------------------------------
  img_E = model(img_L)


  img_E = util.tensor2uint(img_E)

  util.imsave(img_E, os.path.join(img_path))



