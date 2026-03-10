import os
import numpy as np
import cv2
import json
from utils_1 import mediapipe_landmark_video
from utils_1 import *

def read_ground_truth(json_path):
    with open(json_path, 'r') as infile:
        gt_data = json.load(infile)
    ## Read video timestamps
    video_t = []
    for sample in gt_data['/Image']:
        video_t.append(sample['Timestamp'])
    ## Read oximeter data
    wave_t = []
    wave = []
    for sample in gt_data['/FullPackage']:
        wave_t.append(sample['Timestamp'])
        wave.append(sample['Value']['o2saturation'])
    video_t = np.array(video_t)*1e-9
    wave_t = np.array(wave_t)*1e-9
    wave = np.array(wave)
    wave = np.interp(video_t, wave_t, wave)
    return video_t, wave

def resample(input_signal, target_length):
    input_signal = np.asarray(input_signal)
    return np.asarray(np.interp(np.linspace(1, input_signal.shape[0], target_length), np.linspace(1, input_signal.shape[0], input_signal.shape[0]), input_signal))

def extract_synchronized_spo2(json_path, number_of_frame):
    with open(json_path, 'r') as f:
        data = json.load(f)
    spo2_values = [item['Value']['o2saturation'] for item in data['/FullPackage']]
    rppg_values = [item['Value']['waveform'] for item in data['/FullPackage']]
    rppg_values = resample(rppg_values, number_of_frame)
    return resample(spo2_values, number_of_frame), rppg_values


def construct_spatio_temporal_map(video_path):
    landmarks = mediapipe_landmark_video(video_path)
    video, successful = make_video_array(video_path, landmarks)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    output_video = []
    for rgb_frame in video:
        rgb_frame = cv2.resize(rgb_frame,(32,32))
        # rgb_frame = cv2.resize(rgb_frame,(64,64))
        hsv_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2HSV)
        hsl_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2HLS)
        lab_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2Lab)
        color_space_frame = np.concatenate([rgb_frame, hsv_frame, hsl_frame, lab_frame],2)
        output_video.append(color_space_frame)
    # output_video = np.array(output_video)
    output_video = np.stack(output_video,axis=0)
    return output_video


def save_spatio_temporal_map_as_image(st_map, image_path, colormap=cv2.COLORMAP_JET):
    st_map_min = st_map.min()
    st_map_max = st_map.max()
    st_map_normalized = 255 * (st_map - st_map_min) / (st_map_max - st_map_min)
    st_map_normalized = st_map_normalized.astype(np.uint8)

    # Apply the colormap to each channel
    st_map_colored = [cv2.applyColorMap(st_map_channel, colormap) for st_map_channel in st_map_normalized]
    
    # Concatenate all colored channels along the width axis
    st_map_image = np.concatenate(st_map_colored, axis=1)
    
    # Save the final image
    cv2.imwrite(image_path, st_map_image)




def preprocess_and_save_spatio_temporal_maps(video_dir,json_dir, output_dir, image_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    
    for video_file in video_files:
        video_path = os.path.join(video_dir, video_file)
        json_path = os.path.join(json_dir, video_file.replace('mp4','json'))
        spatio_temporal_map = construct_spatio_temporal_map(video_path)
        num_of_frame = spatio_temporal_map.shape[0]
        wave, rppg_wave = extract_synchronized_spo2(json_path, num_of_frame)
        map_filename = os.path.splitext(video_file)[0] + '.npz'
        map_path = os.path.join(output_dir, map_filename)
        # np.save(map_path, spatio_temporal_map)
        if np.any(wave < 90) or np.any(wave > 100):
            print(video_path, 'is noisy GT')
            continue
        np.savez_compressed(map_path, video = spatio_temporal_map, wave=wave,rppg_wave=rppg_wave, fps=30)   
        image_filename = os.path.splitext(video_file)[0] + '.png'
        image_path = os.path.join(image_dir, image_filename)
        # save_spatio_temporal_map_as_image(spatio_temporal_map, image_path)
        
        print(f"Saved spatio-temporal map and image for {video_file}")

# Example usage
if __name__ == "__main__":
    video_dir = '/media/sda1_acces/Code_SPO2/SPO2_work_new/Datasets/PURE_video_sample'
    output_dir = '/media/sdb_access/SPO2_light_weight/Dataset/Pure_data_video_with_rppg'
    image_dir = '/media/sdb_access/SPO2_light_weight/New_data_process/PURE_ST_maps_img'
    json_dir = '/media/sda1_acces/Code_SPO2/SPO2_work_new/Datasets/PURE_JSON'
    preprocess_and_save_spatio_temporal_maps(video_dir,json_dir, output_dir, image_dir)


