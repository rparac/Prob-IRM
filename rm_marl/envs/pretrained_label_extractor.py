from typing import List

from sklearn.metrics import f1_score, precision_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import gymnasium as gym
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from rm_marl.envs.wrappers import LabelExtractor

from gym_subgoal_automata.envs.officeworld.officeworld_env import OfficeWorldObject


class LabelingFunctionDataset(torch.utils.data.Dataset):
    def __init__(self, data: pd.DataFrame, propositions: List[str], num_states: int):
        self._data = data
        self.propositions = propositions
        self.num_states = num_states
        self.num_propositions = len(propositions)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        row = self._data.iloc[idx]
        obs = row["obs"]
        row_labels = row[self.propositions]
        labels = [bool(x) for x in row_labels.values]
        labels = torch.tensor(labels)
        labels = labels.float()

        obs = torch.tensor(obs)
        obs = F.one_hot(obs, num_classes=self.num_states)
        obs = obs.float()
        return obs, labels

class SimpleNN(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNN, self).__init__()

        self._lin_layer = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self._lin_layer(x)


    def inference(self, x):
        with torch.no_grad():
            # Add batch dimension
            x = x.unsqueeze(0)
            out = self._lin_layer(x)
            out = torch.sigmoid(out)
            out = out.squeeze(0)
            return out

class PretrainedLabelExtractor(LabelExtractor):
    def __init__(self, env, tb_storage_path: str, curr_id: int):
        super().__init__()

        self._curr_id = curr_id

        self._tb_storage_path = tb_storage_path

        self._num_collection_episodes = 250
        self._dataset = self._collect_training_data(env)
        self._test_dataset = self._collect_test_data(env)
        self._num_training_epochs = 2
        self._model = self._train_model(self._dataset, self._test_dataset)

    def get_labels(self, observation, info: dict):
        obs_vector = F.one_hot(torch.tensor(observation), num_classes=self._dataset.num_states)
        obs_vector = obs_vector.float()
        out = self._model.inference(obs_vector)

        ret = {}
        for prop_idx, prop_name in enumerate(self._dataset.propositions):
            ret[prop_name] = out[prop_idx].item()
        return ret

    def get_labels_without_probability(self, observation, info: dict):
        label_dict = self.get_labels(observation, info)
        ret = set()
        for label, value in label_dict.items():
            if value > 0.5:
                ret.add(label)
        return ret

    def _collect_training_data(self, env: gym.Wrapper):
        all_observations = env.unwrapped.get_observables()

        data = {
            observable: []
            for observable in all_observations
        }
        data["obs"] = []

        for i in range(self._num_collection_episodes):
            done = False
            obs, info = env.reset()
            while not done:
                obs, _reward, terminated, truncated, info = env.step(env.action_space.sample())
                done = terminated or truncated
                data["obs"].append(obs)
                labels_seen = info["observations"]
                for observable in all_observations:
                    seen = observable in labels_seen
                    data[observable].append(seen)

        df = pd.DataFrame(data)
        return LabelingFunctionDataset(df, all_observations, env.observation_space.n)

    def _collect_test_data(self, env: gym.Wrapper):
        all_observations = env.unwrapped.get_observables()

        data = {
            observable: []
            for observable in all_observations
        }
        data["obs"] = []

        num_states = env.observation_space.n 
        for obs in range(num_states):
            data["obs"].append(obs)

            agent = obs // env.unwrapped.height, obs % env.unwrapped.height

            curr_observations = set()
            for location in env.unwrapped.locations:
                if location == agent:
                    curr_observations.add(env.unwrapped.locations[location])

            if agent in env.unwrapped.coffee:
                curr_observations.add(OfficeWorldObject.COFFEE)
            if agent == env.unwrapped.mail:
                curr_observations.add(OfficeWorldObject.MAIL)

            for observable in all_observations:
                seen = observable in curr_observations
                data[observable].append(seen)

        df = pd.DataFrame(data)
        # breakpoint()
        return LabelingFunctionDataset(df, all_observations, env.observation_space.n)


    def _train_model(self, train_dataset: LabelingFunctionDataset, test_dataset: LabelingFunctionDataset):
        model = SimpleNN(train_dataset.num_states, len(train_dataset.propositions))

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        writer = SummaryWriter(log_dir=self._tb_storage_path)
        for epoch in range(self._num_training_epochs):
            # Training phase
            model.train()
            epoch_train_loss = 0.0
            
            for batch_idx, (obs, labels) in enumerate(tqdm(train_loader)):
                obs = obs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                out = model(obs)

                loss = loss_fn(out, labels)
                loss.backward()
                optimizer.step()
                
                epoch_train_loss += loss.item()
            
            # Calculate average training loss for epoch
            avg_train_loss = epoch_train_loss / len(train_dataset)
            writer.add_scalar(f'Loss/{self._curr_id}/train_epoch', avg_train_loss, epoch)
            
            # Evaluation phase
            model.eval()
            with torch.no_grad():
                test_obs, test_labels = next(iter(test_loader))
                test_obs = test_obs.to(device)
                test_labels = test_labels.to(device)

                test_out = model(test_obs)
                avg_test_loss = loss_fn(test_out, test_labels) / len(test_dataset)
                
                # Log test loss
                writer.add_scalar(f'Loss/{self._curr_id}/test', avg_test_loss.item(), epoch)
                
                # Calculate accuracy (using 0.5 threshold for binary classification)
                test_predictions = torch.sigmoid(test_out) > 0.5
                test_accuracy = (test_predictions == test_labels).float().mean()
                writer.add_scalar(f'Accuracy/{self._curr_id}/test', test_accuracy.item(), epoch)
                
                # Log per-proposition accuracy
                for prop_idx, prop_name in enumerate(test_dataset.propositions):
                    f1 = f1_score(test_labels[:, prop_idx].cpu(), test_predictions[:, prop_idx].cpu())
                    writer.add_scalar(f'F1-Score/{self._curr_id}/test_{prop_name}', f1.item(), epoch)
            
            print(f'Epoch {epoch+1}/{self._num_training_epochs} - Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss.item():.4f}, Test Acc: {test_accuracy.item():.4f}')
        model = model.cpu()
        model.eval()
        return model
