import numpy as np
import argparse
import os
import cv2
import csv
import time
from retinaface.pre_trained_models import get_model
from utils_1 import mediapipe_landmark_video
from utils_1 import *
import torch

def print_time(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return "%d:%02d:%02d" % (hour, minutes, seconds)



def ls(x='.'):
    return sorted(os.listdir(x))

def join(*x):
    return os.path.join(*x)


def resample_ppg(input_signal, target_length):
    """Samples a PPG sequence into specific length."""
    input_signal = np.asarray(input_signal)
    return np.asarray(np.interp(np.linspace(1, input_signal.shape[0], target_length), np.linspace(1, input_signal.shape[0], input_signal.shape[0]), input_signal))


def get_orignal_wave(path):
    wave = []
    with open(path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        for j,row in enumerate(csv_reader):
            if j==0:
                continue
            else:
                wave.append(float(row[0]))  # Assuming timestamps are in the first column
    return wave

def construct_spatio_temporal_map(video_path):
    landmarks = mediapipe_landmark_video(video_path)
    video, successful = make_video_array(video_path, landmarks)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    output_video = []
    for rgb_frame in video:
        rgb_frame = cv2.resize(rgb_frame,(32,32))
        hsv_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2HSV)
        hsl_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2HLS)
        lab_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2Lab)
        color_space_frame = np.concatenate([rgb_frame, hsv_frame, hsl_frame, lab_frame],2)
        output_video.append(color_space_frame)
    # output_video = np.array(output_video)
    output_video = np.stack(output_video,axis=0)
    return output_video, fps


def main(args):

    input_root = args.input
    output_root = args.output
    if not os.path.isdir(output_root):
        os.makedirs(output_root)
    

    subjects = ls(input_root)
    for subject in subjects:
        sub = subject.split('_')[0]
        ses = subject.split('_')[1]
        session = sub+'-'+ses
        output_path = join(output_root, f'{session}.npz')
        tic = time.time()
        subject_dir = join(input_root, subject)
        gt_path = subject_dir+'/sensor.csv'
        video_path = subject_dir+'/'+subject+'.avi'
        wave = get_orignal_wave(gt_path)
        spatio_temporal_map, fps = construct_spatio_temporal_map(video_path)
        num_of_frame = spatio_temporal_map.shape[0]

        wave = resample_ppg(wave, num_of_frame)
        wave = wave[0:num_of_frame]
        if np.any(wave < 90) or np.any(wave > 100):
            print(video_path, 'is noisy GT')
            continue

        
        np.savez_compressed(output_path, video = spatio_temporal_map, wave=wave, fps=fps)
       
        print('Time taken by: ', print_time(time.time()-tic))





if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='/media/sda1_acces/rPPG_datasets/Pub_BH-rPPG_FULL_compack',
                        help='Path to the original PURE dataset directory.')
    parser.add_argument('--output', default='/media/sda1_acces/Code_SPO2/SPO2_work_new/SPO2_code_v3/Dataset/Bh_rPPG_dataset',
                        help='Path to the preprocessed output dataset directory with cropped faces.')
    args = parser.parse_args()
    main(args)

