import torchvision.transforms as T
import torch
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

RESIZE_SCREEN_SIZE_HEIGHT = 84
RESIZE_SCREEN_SIZE_WIDTH = 84
FRAME_NUM = 4

resize_and_grayscale = T.Compose([T.ToPILImage(),
                        T.Resize((RESIZE_SCREEN_SIZE_HEIGHT, RESIZE_SCREEN_SIZE_WIDTH), interpolation=Image.BICUBIC),
                        T.Grayscale(num_output_channels=1),
                        T.ToTensor()])

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995
TARGET_UPDATE = 10

NUM_EPISODE = 100
NUM_SIMULATION = 1

NET_PARAMETERS_BK_PATH = 'output/value_net_bk.pth'
