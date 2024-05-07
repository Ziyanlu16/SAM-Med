from models.efficientvit.efficientvim import image_encoder_efficientvim
import torch
model = image_encoder_efficientvim
x = torch.randn([1,3,1014,1024]).to("cuda")
x = model