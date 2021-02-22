import torch
import utility
import numpy as np


class EnvironmentUtility(object):

    @staticmethod
    def num_to_round_action_number(action: float):
        element_list = [int(utility.ACTION_MINIMUM), int(utility.ACTION_MAXIMUM)]
        prob_list = [1 - action, action]
        return np.random.choice(a=element_list, size=1, p=prob_list)[0]

    @staticmethod
    def tensor_to_round_action_number(tensor: torch.tensor):
        reference_value = float((utility.ACTION_MINIMUM + utility.ACTION_MAXIMUM) / 2)
        ary = tensor.detach().clone().numpy()
        for index in range(len(ary)):
            ary[index] = np.array(EnvironmentUtility.num_to_round_action_number(ary[index][0]))
        return torch.from_numpy(ary)
