import torchvision.transforms as T
import torch
import secret
from PIL import Image
from agent.policy.egreedy import EgreedyOptions
from environment.cartpole import CartPole


#### Line notify ####
LINE_NOTIFY_FLG = False
LINE_NOTIFY_MSG = "実行完了\n経過時間: {}"
LINE_NOTIFY_TOKEN = secret.LINE_NOTIFY_TOKEN if LINE_NOTIFY_FLG else None

#### task ####
TASK = CartPole()

#### Simulation parameters ####
NUM_EPISODE = 1000
NUM_SIMULATION = 1

#### DQN parameters ####
BATCH_SIZE = 128
GAMMA = 0.999
TARGET_UPDATE = 10
NET_PARAMETERS_BK_PATH = 'output/value_net_bk.pth'
USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# network parameters
NW_LEARNING_RATE = 0.00025
NW_ALPHA = 0.95
NW_EPS = 0.01
# epislon-greedy paramaters
# EPS_MODE = EgreedyOptions.EPISODE
# EPS_START = 1.0
# EPS_END = 0.01
# EPS_TIMESTEPS = 950
EPS_MODE = EgreedyOptions.ACTION
EPS_START = 1.0
EPS_END = 0.01
EPS_TIMESTEPS = 30000
# Soft Update Setting
TAU = 1e-3

#### Replay Buffer parameters ####
NUM_REPLAY_BUFFER = 10000
RESIZE_SCREEN_SIZE_HEIGHT = 84
RESIZE_SCREEN_SIZE_WIDTH = 84
FRAME_NUM = 4
resize_and_grayscale = T.Compose([T.ToPILImage(),
                        T.Resize((RESIZE_SCREEN_SIZE_HEIGHT, RESIZE_SCREEN_SIZE_WIDTH), interpolation=Image.BICUBIC),
                        T.Grayscale(num_output_channels=1),
                        T.ToTensor()])