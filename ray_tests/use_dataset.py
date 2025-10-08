import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from typing import List
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
import gymnasium as gym
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image

class LabelingFunctionDataset(torch.utils.data.Dataset):
    def __init__(self, ds_file: str, propositions: List[str], num_states: int):
        self.data = pd.read_csv(ds_file)
        self._propositions = propositions
        self._num_states = num_states
        self._num_propositions = len(propositions)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        obs = row["obs"]
        row_labels = row[self._propositions]
        labels = [bool(x) for x in row_labels.values]
        labels = torch.tensor(labels)
        labels = labels.float()

        obs = torch.tensor(obs)
        obs = F.one_hot(obs, num_classes=self._num_states)
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
            return out


env_config = {
    "name": "gym_subgoal_automata:OfficeWorldDeliverCoffee-v0",
    "render_mode": "rgb_array",
    "seed": 0,
    "max_episode_length": 300,
    "num_random_seeds": None,
}
env = gym.make(env_config["name"], render_mode=env_config["render_mode"],
                params={"generation": "random", "environment_seed": env_config["seed"],
                        "hide_state_variables": True, "num_plants": 1})

num_states = env.observation_space.n
propositions = env.unwrapped.get_observables()
dataset_file = "/home/rp218/projects/rm-marl/training_dataset.csv"

labeling_function_dataset = LabelingFunctionDataset(dataset_file, propositions, num_states)
train_size = int(0.8 * len(labeling_function_dataset))
test_size = len(labeling_function_dataset) - train_size
train_set, test_set = torch.utils.data.random_split(labeling_function_dataset, [train_size, test_size])

train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=len(test_set), shuffle=False)

model = SimpleNN(num_states, len(propositions))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

loss_fn = nn.BCEWithLogitsLoss()

num_epochs = 2

# Initialize TensorBoard writer
writer = SummaryWriter(log_dir='runs/labeling_function_training')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

for epoch in range(num_epochs):
    # Training phase
    model.train()
    epoch_train_loss = 0.0
    num_train_batches = 0
    
    for batch_idx, (obs, labels) in enumerate(tqdm(train_loader)):
        obs = obs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        out = model(obs)

        loss = loss_fn(out, labels)
        loss.backward()
        optimizer.step()
        
        epoch_train_loss += loss.item()
        num_train_batches += 1
        
        # Log batch loss
        global_step = epoch * len(train_loader) + batch_idx
        writer.add_scalar('Loss/train_batch', loss.item(), global_step)
    
    # Calculate average training loss for epoch
    avg_train_loss = epoch_train_loss / num_train_batches
    writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)
    
    # Evaluation phase
    model.eval()
    with torch.no_grad():
        test_obs, test_labels = next(iter(test_loader))
        test_obs = test_obs.to(device)
        test_labels = test_labels.to(device)

        test_out = model(test_obs)
        test_loss = loss_fn(test_out, test_labels)
        
        # Log test loss
        writer.add_scalar('Loss/test', test_loss.item(), epoch)
        
        # Calculate accuracy (using 0.5 threshold for binary classification)
        test_predictions = torch.sigmoid(test_out) > 0.5
        test_accuracy = (test_predictions == test_labels).float().mean()
        writer.add_scalar('Accuracy/test', test_accuracy.item(), epoch)
        
        # Log per-proposition accuracy
        for prop_idx, prop_name in enumerate(propositions):
            prop_accuracy = (test_predictions[:, prop_idx] == test_labels[:, prop_idx]).float().mean()
            writer.add_scalar(f'Accuracy/test_{prop_name}', prop_accuracy.item(), epoch)
    
    print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Test Loss: {test_loss.item():.4f}, Test Acc: {test_accuracy.item():.4f}')

print(f'\nTraining complete! Computing confusion matrices...')

# Compute confusion matrix for each proposition on test set
print("\n" + "="*50)
print("CONFUSION MATRIX RESULTS")
print("="*50)

model.eval()
with torch.no_grad():
    test_obs, test_labels = next(iter(test_loader))
    test_obs = test_obs.to(device)
    test_labels = test_labels.to(device)
    
    test_out = model(test_obs)
    test_predictions = torch.sigmoid(test_out) > 0.5
    
    # Convert to numpy for sklearn
    test_predictions_np = test_predictions.cpu().numpy()
    test_labels_np = test_labels.cpu().numpy()
    
    # Compute confusion matrix for each proposition
    for prop_idx, prop_name in enumerate(propositions):
        cm = confusion_matrix(test_labels_np[:, prop_idx], test_predictions_np[:, prop_idx])
        
        print(f"\nProposition: {prop_name}")
        print("-" * 40)
        print("Confusion Matrix:")
        print(f"                Predicted")
        print(f"                 0      1")
        print(f"Actual    0   {cm[0,0]:5d}  {cm[0,1]:5d}")
        print(f"          1   {cm[1,0]:5d}  {cm[1,1]:5d}")
        
        # Calculate metrics
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\nMetrics:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        
        # Log metrics to TensorBoard
        writer.add_scalar(f'ConfusionMatrix/{prop_name}/Accuracy', accuracy, num_epochs)
        writer.add_scalar(f'ConfusionMatrix/{prop_name}/Precision', precision, num_epochs)
        writer.add_scalar(f'ConfusionMatrix/{prop_name}/Recall', recall, num_epochs)
        writer.add_scalar(f'ConfusionMatrix/{prop_name}/F1-Score', f1, num_epochs)
        
        # Log confusion matrix values to TensorBoard
        writer.add_scalar(f'ConfusionMatrix/{prop_name}/TrueNegatives', tn, num_epochs)
        writer.add_scalar(f'ConfusionMatrix/{prop_name}/FalsePositives', fp, num_epochs)
        writer.add_scalar(f'ConfusionMatrix/{prop_name}/FalseNegatives', fn, num_epochs)
        writer.add_scalar(f'ConfusionMatrix/{prop_name}/TruePositives', tp, num_epochs)
        
        # Create confusion matrix visualization and log as image
        fig, ax = plt.subplots(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['False', 'True'])
        disp.plot(ax=ax, cmap='Blues', values_format='d')
        ax.set_title(f'Confusion Matrix: {prop_name}\nAccuracy: {accuracy:.4f}, F1: {f1:.4f}')
        
        # Convert plot to image and log to TensorBoard
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        image = Image.open(buf)
        image_array = np.array(image)
        
        # TensorBoard expects (C, H, W) format
        image_array = np.transpose(image_array[:, :, :3], (2, 0, 1))
        writer.add_image(f'ConfusionMatrix/{prop_name}', image_array, num_epochs)
        
        plt.close(fig)
        buf.close()

print("\n" + "="*50)

# Close the writer
writer.close()

print(f'\nTensorBoard logs saved to: runs/labeling_function_training')
print('To view logs, run: tensorboard --logdir=runs/labeling_function_training')
print(f'\nSample data point: {labeling_function_dataset[0]}')