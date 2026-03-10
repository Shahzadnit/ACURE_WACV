import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from Model_backbone_varient import SpO2Model
from Loss import  SpO2Loss
from dataset import train_dataset, test_dataset
from utils import *
from torchvision import transforms
import torch
from torch.utils.data import Dataset
import numpy as np
import os
import torchvision.transforms.functional as TF
import random

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # For deterministic behavior (might affect performance)
    torch.backends.cudnn.deterministic = True
set_seed(1)

class RandomHorizontalFlip3D:
    """Randomly flip the video horizontally."""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, video):
        if np.random.rand() < self.p:
            return video.flip(-1)  # Assuming the last dimension is width
        return video

class RandomCrop3D:
    """Randomly crop the video to a given size."""
    def __init__(self, output_size):
        self.output_size = output_size  # (C, D, H, W)

    def __call__(self, video):
        _, D, H, W = video.shape
        c, d, h, w = self.output_size

        if d > D or h > H or w > W:
            raise ValueError("Crop size should be smaller than the video size.")

        # Random start indices
        d_start = np.random.randint(0, D - d + 1)
        h_start = np.random.randint(0, H - h + 1)
        w_start = np.random.randint(0, W - w + 1)

        return video[:, d_start:d_start + d, h_start:h_start + h, w_start:w_start + w]



# Example transformation pipeline
# transform = transforms.Compose([
#     RandomHorizontalFlip3D(p=0.5),
#     RandomCrop3D(output_size=(12, 300, 32, 32)),
#     # Add more transformations as needed
#     # For normalization, you might need a custom Normalize3D
# ])

# If you need to normalize, define a custom Normalize3D
class Normalize3D:
    """Normalize 3D tensors with mean and std."""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        # Assuming tensor shape is (C, D, H, W)
        for c in range(tensor.shape[0]):
            tensor[c] = (tensor[c] - self.mean[c]) / self.std[c]
        return tensor
    
class DynamicMinMaxScale:
    """
    Scales values in a 3D/4D tensor dynamically to [0,1] 
    based on the tensor's min and max.
    
    Shape assumptions: (C, D, H, W) or (C, H, W) etc.
    """
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        min_val = tensor.min()
        max_val = tensor.max()
        range_val = max_val - min_val

        # Avoid division by zero if all values are the same
        if range_val == 0:
            return torch.zeros_like(tensor)
        
        return (tensor - min_val) / range_val


# Example usage:
normalize = Normalize3D(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
dynamic_scale = DynamicMinMaxScale()
transform = transforms.Compose([
        dynamic_scale,
        normalize
    ])


# 10. K-Fold Cross-Validation with Resuming Checkpoints
def train_model_kfold(map_files, map_dir, n_splits=5, num_epochs=50, batch_size=20, device='cpu', base_plot_save_path='plots', base_model_save_path='models', resume=False, checkpoint_dir=None, fps=None,temp_dim=300):
    # Extract unique subject IDs and group files by subject
    subject_to_files = {}
    for f in map_files:
        subject_id = f.split('-')[0]  # Get subject ID (e.g., '01' from '01-02.npz')
        if subject_id not in subject_to_files:
            subject_to_files[subject_id] = []
        subject_to_files[subject_id].append(f)
    
    # Get unique subject IDs
    subject_ids = sorted(list(subject_to_files.keys()))
    print(f"Unique subjects: {subject_ids}")

    # Perform K-fold on subject IDs
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    logger = setup_logging(base_model_save_path)
    fold_mae_scores = []
    fold = 1

    # Determine the starting fold if resuming
    start_fold = 1
    if resume:
        for fold_idx in range(1, n_splits + 1):
            checkpoint_path = os.path.join(checkpoint_dir, f'fold_{fold_idx}_checkpoint.pth')
            if os.path.exists(checkpoint_path):
                start_fold = fold_idx
            else:
                break  # Resume from this fold onwards

    for fold, (train_indices, test_indices) in enumerate(kfold.split(subject_ids), 1):
        if fold < start_fold:
            continue  # Skip folds before the resume point
        print(f"Starting Fold {fold}/{n_splits}")
        
        # Inner: split remaining subjects into train/val (K-1 folds -> 1 val fold)
        inner_kfold = KFold(n_splits=n_splits - 1, shuffle=True, random_state=fold)
        inner_splits = list(inner_kfold.split(train_indices))
        inner_pick = (fold - 1) % (n_splits - 1)  # rotate which inner fold is used as val
        inner_train_idx, inner_val_idx = inner_splits[inner_pick]

        train_subjects = [subject_ids[i] for i in train_indices[inner_train_idx]]
        val_subjects = [subject_ids[i] for i in train_indices[inner_val_idx]]

        # Get subject IDs for this fold
        # train_subjects = [subject_ids[i] for i in train_indices]
        test_subjects = [subject_ids[i] for i in test_indices]

        # Get corresponding video files
        train_map_files = []
        for subject in train_subjects:
            train_map_files.extend(subject_to_files[subject])
        test_map_files = []
        for subject in test_subjects:
            test_map_files.extend(subject_to_files[subject])
        
        val_map_files = []
        for subject in val_subjects:
            val_map_files.extend(subject_to_files[subject])

        print(f"Fold {fold}: {len(train_subjects)} subjects for training, {len(val_subjects)} subjects for Val, {len(test_subjects)} subjects for testing")
        print(f"Train subjects: {train_subjects},Val subjects: {val_subjects}, Test subjects: {test_subjects}")
        print(f"Train files: {len(train_map_files)},Val files: {len(val_map_files)}, Test files: {len(test_map_files)}")

        plot_save_path = os.path.join(base_plot_save_path, f'fold_{fold}')
        model_save_path = os.path.join(base_model_save_path, f'fold_{fold}')


        # # dataset old
        train_data = train_dataset(map_files=map_files,map_dir=map_dir,chunk_size=temp_dim,transform=transform)
        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        eval_data = test_dataset(map_files=test_map_files,map_dir=map_dir,chunk_size=temp_dim,transform=transform)
        eval_dataloader = DataLoader(eval_data, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
        
        val_data = test_dataset(map_files=val_map_files,map_dir=map_dir,chunk_size=temp_dim,transform=transform)
        val_dataloader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)


        model = SpO2Model(input_channel=3, output_dim=temp_dim,temporal_kind="LTC", temporal_steps=100,backbone_kind="RESNET3D18")

        criterion = SpO2Loss() 
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        best_mae = train_model(train_dataloader, eval_dataloader,val_dataloader, model, criterion, optimizer, num_epochs=num_epochs, device=device, plot_save_path=plot_save_path, model_save_path=model_save_path, logger=logger, checkpoint_dir=checkpoint_dir, resume=resume, fold=fold,fps=fps)
        fold_mae_scores.append(best_mae)

    average_mae, std_mae = np.mean(fold_mae_scores), np.std(fold_mae_scores)
    print(f"Average MAE across all folds: {average_mae:.2f}, Standard Deviation: {std_mae:.2f}")
    logger.info(f"Average MAE across all folds: {average_mae:.2f}, Standard Deviation: {std_mae:.2f}")

# 11. Main script execution
if __name__ == "__main__":
    map_dir = '/media/sda1_acces/Code_SPO2/ACURE_WACV/Dataset/Pure_data_video'
    base_plot_save_path = '/media/sda1_acces/Code_SPO2/ACURE_WACV/results/PURE_res/PURE_Plots_eval'
    base_model_save_path = '/media/sda1_acces/Code_SPO2/ACURE_WACV/results/PURE_res/PURE_weight'
    checkpoint_dir = '/media/sda1_acces/Code_SPO2/ACURE_WACV/results/PURE_res/PURE_checkpoints'
    fps = 30   # pure_fps:30, BH_rPPG_fps:15

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    map_files = sorted([f for f in os.listdir(map_dir) if f.endswith('.npz')])

    train_model_kfold(map_files, map_dir, n_splits=5, num_epochs=60, batch_size=20, device=device, 
                      base_plot_save_path=base_plot_save_path,
                       base_model_save_path=base_model_save_path,
                         resume=False, checkpoint_dir=checkpoint_dir,fps=fps,temp_dim=300)
