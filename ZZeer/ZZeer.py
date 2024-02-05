import torch
import numpy as np

class TrainedAgent:
    def __init__(self):
        self.model = torch.load("rl_model.zip")
        self.model.eval()

    def act(self, state):
        # Convertir l'état en tensor pytorch
        state_tensor = torch.tensor(state, dtype=torch.float32)

        # Utiliser le modèle pour prédire les actions
        with torch.no_grad():
            actions, _ = self.model(state_tensor)

        # Convertir les actions en format compréhensible par le simulateur
        parsed_actions = np.zeros((1, 8))
        parsed_actions[0, 0] = actions[0, 0]  # throttle
        parsed_actions[0, 1] = actions[0, 1]  # steer
        parsed_actions[0, 2] = actions[0, 0]  # pitch
        parsed_actions[0, 3] = actions[0, 1] * (1 - actions[0, 4])  # yaw
        parsed_actions[0, 4] = actions[0, 1] * actions[0, 4]  # roll
        parsed_actions[0, 5] = actions[0, 2]  # jump
        parsed_actions[0, 6] = actions[0, 3]  # boost
        parsed_actions[0, 7] = actions[0, 4]  # handbrake

        return parsed_actions[0]
