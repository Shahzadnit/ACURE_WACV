import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import KFold
from Model_backbone_varient import SpO2Model
from dataset import  test_dataset
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import random
from utils import setup_logging



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
# normalize = Normalize3D(mean=mean, std=std)
dynamic_scale = DynamicMinMaxScale()
transform = transforms.Compose([
        dynamic_scale,
        normalize
    ])



def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)
    return model


def plot_spo2_values(predicted, original, epoch, save_path, filename_prefix="fold", frame_rate=15):
    os.makedirs(save_path, exist_ok=True)
    # Sample one value per second
    sampled_indices = np.arange(0, len(predicted), frame_rate)  # Every frame_rate steps
    sampled_time = sampled_indices / frame_rate
    sampled_predicted = np.array(predicted)[sampled_indices]
    sampled_original = np.array(original)[sampled_indices]

    plt.figure(figsize=(20, 10))
    plt.plot(sampled_time, sampled_predicted, label='Predicted SpO₂')
    plt.plot(sampled_time, sampled_original, label='Original SpO₂')
    plt.xlabel('Time (s)')
    plt.ylabel('SpO₂')
    plt.title(f'Sampled SpO₂ Prediction vs Original - Epoch {epoch}')
    plt.legend()
    plot_filename = f"{filename_prefix}_spo2_epoch_{epoch}_sampled.pdf"
    plt.savefig(os.path.join(save_path, plot_filename))
    plt.close()



import torch
import torch.nn as nn
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

def evaluate_model(dataloader, model, device='cpu', fold=1, plot_save_path='plots', fps=None):
    model.eval()
    model.to(device)
    total_samples = 0
    total_loss = 0.0
    predicted_values = []
    original_values = []
    
    with torch.no_grad():
        for idx, (st_map, labels) in enumerate(dataloader):
            st_map, labels = st_map.to(device), labels.to(device)
            outputs, dc_output, ac_output = model(st_map)
            loss = nn.MSELoss()(outputs, labels)
            total_loss += loss.item() * outputs.size(0)  # Multiply by the batch size to accumulate correctly
            total_samples += outputs.size(0)
            predicted_values.extend(outputs.cpu().numpy().flatten())
            original_values.extend(labels.cpu().numpy().flatten())
    
    # Plot predicted vs original values
    plot_spo2_values(predicted_values, original_values, fold, plot_save_path, frame_rate=fps)    
    
    # Calculate evaluation metrics
    mae = total_loss / total_samples
    rmse = mean_squared_error(original_values, predicted_values, squared=False)  # Root Mean Squared Error
    corrcoef, _ = pearsonr(original_values, predicted_values)  # Pearson Correlation Coefficient
    
    # # Print metrics
    print(f"Test MAE (Mean Absolute Error): {mae:.3f}")
    print(f"Test RMSE (Root Mean Squared Error): {rmse:.3f}")
    print(f"Test Pearson Correlation Coefficient: {corrcoef:.3f}")
    
    return mae, rmse, corrcoef


# 10. K-Fold Cross-Validation with Resuming Checkpoints
def test_model_kfold(map_files, map_dir, n_splits=5, device='cpu', base_plot_save_path='plots', weight_path=None,fps=None,temp_dim=300):
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
    fold_mae_scores, fold = [], 1
    fold_rmse_scores = []
    fold_corr_scores = []

    # Determine the starting fold if resuming
    start_fold = 1

    for fold, (train_indices, test_indices) in enumerate(kfold.split(subject_ids), 1):
        print(f"Starting Fold {fold}/{n_splits}")
        # Get subject IDs for this fold
        train_subjects = [subject_ids[i] for i in train_indices]
        test_subjects = [subject_ids[i] for i in test_indices]

        # Get corresponding video files
        train_map_files = []
        for subject in train_subjects:
            train_map_files.extend(subject_to_files[subject])
        test_map_files = []
        for subject in test_subjects:
            test_map_files.extend(subject_to_files[subject])
        
        test_map_files_1 = []
        for file in test_map_files:
            if '-2' in file:
            # if '-04' in file or '-05' in file:
            # if '-01' in file:
                test_map_files_1.append(file)
        test_map_files = np.array(test_map_files_1)

        print(f"Fold {fold}: {len(train_subjects)} subjects for training, {len(test_subjects)} subjects for testing")
        print(f"Train subjects: {train_subjects}, Test subjects: {test_subjects}")
        print(f"Train files: {len(train_map_files)}, Test files: {len(test_map_files)}")

        plot_save_path = os.path.join(base_plot_save_path, f'fold_{fold}')

        eval_data = test_dataset(map_files=test_map_files,map_dir=map_dir,chunk_size=temp_dim,transform=transform)
        eval_dataloader = DataLoader(eval_data, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

        model = SpO2Model(input_channel=3, output_dim=temp_dim,temporal_kind="LTC", temporal_steps=100,backbone_kind="RESNET3D18")

        checkpoint_path = os.path.join(weight_path, f'fold_{fold}/best_model.pth')
        model = load_checkpoint(model, checkpoint_path)

        best_mae, rmse, corrcoef = evaluate_model(eval_dataloader, model, device=device, fold=fold, plot_save_path=plot_save_path,fps=fps)

        # best_mae = train_model(train_dataloader, eval_dataloader, model, criterion, optimizer, num_epochs=num_epochs, device=device, plot_save_path=plot_save_path, model_save_path=model_save_path, logger=logger, checkpoint_dir=checkpoint_dir, resume=resume, fold=fold)
        fold_mae_scores.append(best_mae)
        fold_rmse_scores.append(rmse)
        fold_corr_scores.append(corrcoef)

    average_mae, std_mae = np.mean(fold_mae_scores), np.std(fold_mae_scores)
    average_rmse, std_rmse = np.mean(fold_rmse_scores), np.std(fold_rmse_scores)
    average_corr, std_corr = np.mean(fold_corr_scores), np.std(fold_corr_scores)
    print(f"Average MAE across all folds: {average_mae:.2f}, Standard Deviation: {std_mae:.2f}")
    logger.info(f"Average MAE across all folds: {average_mae:.2f}, Standard Deviation: {std_mae:.2f}")
    print(f"Average RMSE across all folds: {average_rmse:.2f}, Standard Deviation: {std_rmse:.2f}")
    logger.info(f"Average RMSE across all folds: {average_rmse:.2f}, Standard Deviation: {std_rmse:.2f}")
    print(f"Average MAE Corr all folds: {average_corr:.2f}, Standard Deviation: {std_corr:.2f}")
    logger.info(f"Average Corr across all folds: {average_corr:.2f}, Standard Deviation: {std_corr:.2f}")

# 11. Main script execution
if __name__ == "__main__":

    map_dir = map_dir = '/media/sdb_access/SPO2_light_weight/Code_from_cluster/FUSE/Dataset/Pure_data_video'
    base_plot_save_path = '/media/sda1_acces/Code_SPO2/ACURE_WACV/results/PURE_res/PURE_test_plots'
    base_model_save_path = '/media/sda1_acces/Code_SPO2/ACURE_WACV/results/PURE_res/PURE_weight'
    fps = 30   # pure_fps:30, BH_rPPG_fps:15



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    map_files = sorted([f for f in os.listdir(map_dir) if f.endswith('.npz')])

    test_model_kfold(map_files, map_dir, n_splits=5, device=device, base_plot_save_path=base_plot_save_path, weight_path=base_model_save_path, fps=fps,temp_dim=300)
