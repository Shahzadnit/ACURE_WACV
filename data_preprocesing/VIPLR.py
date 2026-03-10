import numpy as np
import argparse
import os
import cv2
import csv
import time
import traceback
from tqdm import tqdm

from utils_1 import mediapipe_landmark_video
from utils_1 import *  # make_video_array, etc.


def print_time(seconds):
    seconds = int(seconds)
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return "%d:%02d:%02d" % (hour, minutes, seconds)


def join(*x):
    return os.path.join(*x)


def resample_ppg(input_signal, target_length):
    """Samples a PPG/SpO2 waveform into specific length."""
    input_signal = np.asarray(input_signal, dtype=np.float32)
    if input_signal.shape[0] == 0:
        return input_signal
    x_old = np.linspace(1, input_signal.shape[0], input_signal.shape[0])
    x_new = np.linspace(1, input_signal.shape[0], target_length)
    return np.asarray(np.interp(x_new, x_old, input_signal), dtype=np.float32)


def get_orignal_wave(path):
    wave = []
    with open(path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        for j, row in enumerate(csv_reader):
            if j == 0:
                continue
            # robust: skip empty rows
            if len(row) == 0:
                continue
            wave.append(float(row[0]))
    return wave


def construct_spatio_temporal_map(video_path):
    landmarks = mediapipe_landmark_video(video_path)
    video, successful = make_video_array(video_path, landmarks)

    if not successful or video is None or len(video) == 0:
        raise RuntimeError(f"make_video_array failed or returned empty video for: {video_path}")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    output_video = []
    for rgb_frame in video:
        rgb_frame = cv2.resize(rgb_frame, (32, 32))
        output_video.append(rgb_frame)

    output_video = np.stack(output_video, axis=0)
    return output_video, fps


def safe_append_csv(csv_path, row):
    """
    Append a row to a CSV file (creates file + header if not exists).
    Row is a dict.
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def load_processed_set(processed_csv_path):
    """
    Load processed sessions (output filenames) from processed_log.csv.
    This makes resume robust even if npz files exist but you changed naming later.
    """
    processed = set()
    if not os.path.isfile(processed_csv_path):
        return processed

    with open(processed_csv_path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # we store output_path; session can be derived but easiest is output_path
            if "output_path" in r:
                processed.add(r["output_path"])
    return processed


def build_session_id(video_path):
    # original logic preserved
    parts = video_path.split("/")
    sub = parts[-4].replace("p", "")
    ses = parts[-3].replace("v", "")
    source = parts[-2].replace("source", "")
    session = sub + "-" + ses + "-" + source
    return session


def main(args):
    input_root = args.input
    output_root = args.output
    os.makedirs(output_root, exist_ok=True)

    # logs live next to output folder (you can change)
    logs_dir = join(output_root, "_logs")
    os.makedirs(logs_dir, exist_ok=True)
    processed_csv = join(logs_dir, "processed_log.csv")
    failed_csv = join(logs_dir, "failed_log.csv")

    processed_set = load_processed_set(processed_csv)

    # collect videos
    video_paths = []
    for root, dirs, files in os.walk(input_root):
        for filename in files:
            if filename.lower().endswith(".avi"):
                full_path = os.path.join(root, filename)

                # requirement (4): skip source4
                if "source4" not in full_path: # dont use infrared 
                    video_paths.append(full_path)

    video_paths = sorted(video_paths)
    print(f"Number of videos found (excluding source4): {len(video_paths)}")

    tic = time.time()

    # tqdm with ETA + average time/video
    pbar = tqdm(video_paths, desc="Processing videos", unit="video", dynamic_ncols=True)
    for idx, video_path in enumerate(pbar):
        start_one = time.time()

        # wrap each video in try/except so pipeline never stops
        try:
            session = build_session_id(video_path)
            output_path = join(output_root, f"{session}.npz")

            # RESUME RULE:
            # skip if output exists OR if it is recorded as processed
            if os.path.isfile(output_path) or output_path in processed_set:
                pbar.set_postfix_str("skipped(resume)")
                continue

            gt_path = video_path.replace("video.avi", "gt_SpO2.csv")
            if not os.path.isfile(gt_path):
                raise FileNotFoundError(f"GT file not found: {gt_path}")

            wave = get_orignal_wave(gt_path)
            spatio_temporal_map, fps = construct_spatio_temporal_map(video_path)
            num_of_frame = spatio_temporal_map.shape[0]

            wave = resample_ppg(wave, num_of_frame)
            wave = wave[:num_of_frame]
            if np.any(wave < 90) or np.any(wave > 100):
                print(video_path, 'is noisy GT')
                continue

            np.savez_compressed(output_path, video=spatio_temporal_map, wave=wave, fps=fps)

            # log success
            safe_append_csv(processed_csv, {
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "video_path": video_path,
                "output_path": output_path,
                "num_frames": int(num_of_frame),
                "fps": float(fps) if fps is not None else -1.0
            })
            processed_set.add(output_path)

            # update tqdm postfix with ETA-ish info
            dt = time.time() - start_one
            avg = (time.time() - tic) / max(1, (idx + 1))
            pbar.set_postfix_str(f"ok | {dt:.2f}s/video | avg {avg:.2f}s")

        except Exception as e:
            # log failure + stack trace
            err = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            safe_append_csv(failed_csv, {
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "video_path": video_path,
                "error": str(e),
                "traceback": err
            })
            pbar.set_postfix_str("FAILED (logged)")
            continue

    print("Done.")
    print("Total time taken:", print_time(time.time() - tic))
    print("Success log:", processed_csv)
    print("Failure log:", failed_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="/media/sdb_access/VIPLR_1_data_set/VIPLR_zip_data/",
        help="Path to the original VIPLR dataset directory."
    )
    parser.add_argument(
        "--output",
        default="/media/sdb_access/SPO2_light_weight/Dataset/VIPLR_data_video_check_only",
        help="Path to the preprocessed output dataset directory."
    )
    args = parser.parse_args()
    main(args)