import os
import numpy as np
import torch
from torch.utils.data import Dataset
import json
from scipy.signal import butter, filtfilt
import cv2

from torchvision import transforms
from PIL import Image



# 3. Data Synchronization and DataLoader
def get_video_frame_rate(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return frame_rate

def resample(input_signal, target_length):
    input_signal = np.asarray(input_signal)
    return np.asarray(np.interp(np.linspace(1, input_signal.shape[0], target_length), np.linspace(1, input_signal.shape[0], input_signal.shape[0]), input_signal))

def extract_synchronized_spo2(json_path, number_of_frame):
    with open(json_path, 'r') as f:
        data = json.load(f)
    spo2_values = [item['Value']['o2saturation'] for item in data['/FullPackage']]
    return resample(spo2_values, number_of_frame)

def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def extract_dc_ac_components(spatio_temporal_map, fs=30):
    b_lp, a_lp = butter_lowpass(cutoff=0.3, fs=fs, order=5)
    dc_component = filtfilt(b_lp, a_lp, spatio_temporal_map, axis=3)
    # b_bp, a_bp = butter_bandpass(lowcut=0.75, highcut=2.5, fs=fs, order=5)
    b_bp, a_bp = butter_bandpass(lowcut=0.75, highcut=4, fs=fs, order=5)
    ac_component = filtfilt(b_bp, a_bp, spatio_temporal_map, axis=3)
    return dc_component, ac_component



# 4. Chunk-based DataLoader for training
class train_dataset(Dataset):
    def __init__(self, map_files, map_dir, chunk_size=300, transform=None):
        self.map_files = map_files
        self.map_dir = map_dir
        self.chunk_size = chunk_size
        self.transform = transform
        
        # Precompute metadata for all chunks for efficient access
        self.chunk_metadata = []  # Store tuples of (map_file, start_frame)
        for map_file in self.map_files:
            file_path = os.path.join(self.map_dir, map_file)
            file = np.load(file_path)
            spatio_temporal_map = file['video']
            number_of_frames = spatio_temporal_map.shape[0]
            
            for start_frame in range(0, number_of_frames - chunk_size + 1, chunk_size):
                self.chunk_metadata.append((map_file, start_frame))
    
    def __len__(self):
        return len(self.chunk_metadata)
    
    def __getitem__(self, idx):
        map_file, start_frame = self.chunk_metadata[idx]
        file_path = os.path.join(self.map_dir, map_file)
        file = np.load(file_path)
        
        spatio_temporal_map = file['video']
        synchronized_spo2 = file['wave']
        fps = file['fps']
        
        end_frame = start_frame + self.chunk_size
        spatio_temporal_map_chunk = spatio_temporal_map[start_frame:end_frame, :, :, 0:3]
        
        # Extract DC and AC components
        dc_component, ac_component = extract_dc_ac_components(
            spatio_temporal_map_chunk.transpose(3, 1, 2, 0), fs=fps
        )
        dc_component = dc_component.transpose(0, 3, 1, 2)
        ac_component = ac_component.transpose(0, 3, 1, 2)
        spatio_temporal_map_chunk = spatio_temporal_map_chunk.transpose(3, 0, 1, 2)
        
        # Convert to tensors
        dc_tensor = torch.tensor(dc_component.copy(), dtype=torch.float32)
        ac_tensor = torch.tensor(ac_component.copy(), dtype=torch.float32)
        st_tensor = torch.tensor(spatio_temporal_map_chunk.copy(), dtype=torch.float32)
        spo2_tensor = torch.tensor(synchronized_spo2[start_frame:end_frame], dtype=torch.float32)
        
        # Apply transformations if any
        if self.transform:
            st_tensor = self.transform(st_tensor)
            dc_tensor = self.transform(dc_tensor)
            ac_tensor = self.transform(ac_tensor)
            # Note: You might not want to transform spo2_tensor
         
        return st_tensor, dc_tensor, ac_tensor, spo2_tensor


class test_dataset(Dataset):
    def __init__(self, map_files, map_dir, chunk_size=300, transform=None):
        """
        Args:
            map_files (list): List of map file names.
            map_dir (str): Directory where map files are stored.
            chunk_size (int, optional): Number of frames per chunk. Default is 300.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.map_files = map_files
        self.map_dir = map_dir
        self.chunk_size = chunk_size
        self.transform = transform
        
        # Precompute metadata for all chunks for efficient access
        self.chunk_metadata = []  # Store tuples of (map_file, start_frame)
        for map_file in self.map_files:
            file_path = os.path.join(self.map_dir, map_file)
            file = np.load(file_path)
            spatio_temporal_map = file['video']
            number_of_frames = spatio_temporal_map.shape[0]
            
            for start_frame in range(0, number_of_frames - chunk_size + 1, chunk_size):
                self.chunk_metadata.append((map_file, start_frame))
    
    def __len__(self):
        return len(self.chunk_metadata)
    
    def __getitem__(self, idx):
        map_file, start_frame = self.chunk_metadata[idx]
        file_path = os.path.join(self.map_dir, map_file)
        file = np.load(file_path)
        
        spatio_temporal_map = file['video']
        synchronized_spo2 = file['wave']
        fps = file['fps']
        
        end_frame = start_frame + self.chunk_size
        spatio_temporal_map_chunk = spatio_temporal_map[start_frame:end_frame, :, :, 0:3]
        
        spatio_temporal_map_chunk = spatio_temporal_map_chunk.transpose(3, 0, 1, 2)
        
        # Convert to tensors
        st_tensor = torch.tensor(spatio_temporal_map_chunk.copy(), dtype=torch.float32)
        spo2_tensor = torch.tensor(synchronized_spo2[start_frame:end_frame], dtype=torch.float32)
        
        # Apply transformations if any
        if self.transform:
            st_tensor = self.transform(st_tensor)
            # Typically, you do not transform spo2_tensor as it's the target
            
        return st_tensor, spo2_tensor
    





class test_single_video(Dataset):
    def __init__(self, map_file, map_dir, chunk_size=300, transform=None):
        """
        Args:
            map_files (list): List of map file names.
            map_dir (str): Directory where map files are stored.
            chunk_size (int, optional): Number of frames per chunk. Default is 300.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.map_files = map_file
        self.map_dir = map_dir
        self.chunk_size = chunk_size
        self.transform = transform
        
        # Precompute metadata for all chunks for efficient access
        self.chunk_metadata = []  # Store tuples of (map_file, start_frame)
        # for map_file in self.map_files:
        file_path = map_file
        file = np.load(file_path)
        spatio_temporal_map = file['video']
        number_of_frames = spatio_temporal_map.shape[0]
        
        for start_frame in range(0, number_of_frames - chunk_size + 1, chunk_size):
            self.chunk_metadata.append((map_file, start_frame))
    
    def __len__(self):
        return len(self.chunk_metadata)
    
    def __getitem__(self, idx):
        map_file, start_frame = self.chunk_metadata[idx]
        file_path = os.path.join(self.map_dir, map_file)
        file = np.load(file_path)
        
        spatio_temporal_map = file['video']
        synchronized_spo2 = file['wave']
        fps = file['fps']
        
        end_frame = start_frame + self.chunk_size
        spatio_temporal_map_chunk = spatio_temporal_map[start_frame:end_frame, :, :, 0:3]
        
        spatio_temporal_map_chunk = spatio_temporal_map_chunk.transpose(3, 0, 1, 2)
        
        # Convert to tensors
        st_tensor = torch.tensor(spatio_temporal_map_chunk.copy(), dtype=torch.float32)
        spo2_tensor = torch.tensor(synchronized_spo2[start_frame:end_frame], dtype=torch.float32)
        
        # Apply transformations if any
        if self.transform:
            st_tensor = self.transform(st_tensor)
            # Typically, you do not transform spo2_tensor as it's the target
            
        return st_tensor, spo2_tensor


