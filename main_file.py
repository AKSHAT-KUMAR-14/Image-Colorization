import argparse
from colorizers import *

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--save_prefix', type=str, default='saved', help='prefix for saving the models')
opt = parser.parse_args()

# Load colorizers
colorizer_eccv16 = eccv16(pretrained=True).eval()
colorizer_siggraph17 = siggraph17(pretrained=True).eval()

# Save the models
torch.save(colorizer_eccv16.state_dict(), f"{opt.save_prefix}_eccv16.pth")
torch.save(colorizer_siggraph17.state_dict(), f"{opt.save_prefix}_siggraph17.pth")

