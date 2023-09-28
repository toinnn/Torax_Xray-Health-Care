import torch

vits16 = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
# features = vits16(your_input)
print("Rodou com sucesso")