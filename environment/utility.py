import torch
import utility
import numpy as np


class EnvironmentUtility(object):

    @staticmethod
    def change_range(
        num: float, range={"maximum": 1.0, "minimum": -1.0}, target_range={"maximum": 1.0, "minimum": 0.0}):
        late = abs(range["minimum"] - num) / (range["maximum"] - range["minimum"])
        return target_range["minimum"] + (target_range["maximum"] - target_range["minimum"]) * late
    
    @staticmethod
    def tensor_change_range(
        tensor: torch.tensor, original_range={"maximum": 1.0, "minimum": -1.0}, target_range={"maximum": 1.0, "minimum": 0.0}):
        ary = tensor.to('cpu').detach().clone().numpy()
        for index in range(len(ary)):
            ary[index] = np.array(EnvironmentUtility.change_range(ary[index][0], original_range, target_range))
        return torch.from_numpy(ary)

    @staticmethod
    def num_to_round_action_number(action: float):
        element_list = [int(utility.ACTION_MINIMUM), int(utility.ACTION_MAXIMUM)]
        prob_list = [1 - action, action]
        return np.random.choice(a=element_list, size=1, p=prob_list)[0]

    @staticmethod
    def tensor_to_round_action_number(tensor: torch.tensor):
        ary = tensor.to('cpu').detach().clone().numpy()
        for index in range(len(ary)):
            ary[index] = np.array(EnvironmentUtility.num_to_round_action_number(ary[index][0]))
        return torch.from_numpy(ary)
