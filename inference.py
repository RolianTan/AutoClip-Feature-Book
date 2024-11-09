import argparse
import torch
import os
from tqdm import tqdm
import numpy as np
from model import AutoClipModel
from transformers import CLIPTokenizer, CLIPModel, CLIPVisionModel
from moviepy.editor import concatenate_videoclips, VideoFileClip, AudioFileClip
from concurrent.futures import ThreadPoolExecutor
import json
import cv2

def get_args():
    parser = argparse.ArgumentParser(description='Inference Configs')
    parser.add_argument('--exp_dir', type=str, default='', help='exp_dir')
    parser.add_argument('--pretrained', type=str, default='', help='the pretrained .pt weight file for inference')
    parser.add_argument('--data_dir', type=str, default='', help='directory containing video and json files')
    parser.add_argument('--video_name', type=str, default='', help='the inference video file name')
    parser.add_argument('--json_name', type=str, default='', help='the json file with captions and intervals')
    parser.add_argument('--reference_video', type=str, help='reference video to extract BGM from')
    parser.add_argument('--device', type=str, default='cuda', help='device to run the inference process')
    parser.add_argument('--fps', type=int, default=2, help='the inference fps')
    parser.add_argument('--size', type=int, default=224, help='frame size (e.g., 224 for 224x224)')
    parser.add_argument('--n_frames', type=int, default=8, help='number of frames to sample from each interval')
    parser.add_argument('--out_mash_video', type=str, help='mash video name')
    parser.add_argument('--t', type=float, default=0.07, help='temperature for projection')
    parser.add_argument('--projection_dim', type=int, default=512, help='dimension of the projection space')
    return parser.parse_args()

def intervals_overlap(start1, end1, start2, end2):
    return max(start1, start2) < min(end1, end2)

def extract_video_frames(video_path, frame_size, target_fps):
    video = cv2.VideoCapture(video_path)
    frames = []
    original_fps = video.get(cv2.CAP_PROP_FPS)
    extract_gap = int(original_fps / target_fps)
    step = 0

    while video.isOpened():
        stat, frame = video.read()
        if not stat:
            break
        if step % extract_gap == 0:
            frame = cv2.resize(frame, (frame_size, frame_size))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        step += 1

    video.release()
    return np.array(frames)

def sample_frames(frames, n_frames):
    if len(frames) >= n_frames:
        sampled_indices = np.linspace(0, len(frames) - 1, n_frames, dtype=int)
        sampled_frames = [frames[i] for i in sampled_indices]
    else:
        sampled_frames = frames + [frames[-1]] * (n_frames - len(frames))
    return sampled_frames

def extract_segment(i, seg_frames, all_video_frames, n_frames):
    interval_frames = all_video_frames[(i * seg_frames):((i + 1) * seg_frames)]
    return sample_frames(interval_frames, n_frames)

def main():
    config = get_args()

    # 1. Load the model and the pre-trained weights
    vision_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(config.device)
    vision_input_dim = vision_model.config.hidden_size
    text_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").text_model.to(config.device)
    text_input_dim = text_model.config.hidden_size
    model = AutoClipModel(vision_model, vision_input_dim, text_model, text_input_dim, config.projection_dim, config.t).to(config.device)
    model.load_state_dict(torch.load(os.path.join(config.exp_dir, config.pretrained)))
    model.eval()

    # 2. Load the JSON file with interval times and captions
    json_path = os.path.join(config.data_dir, config.json_name)
    with open(json_path, 'r') as f:
        intervals_data = json.load(f)

    # 3. Load and process the raw video
    video_path = os.path.join(config.data_dir, config.video_name)
    raw_video = VideoFileClip(video_path)
    total_time = raw_video.duration
    all_video_frames = extract_video_frames(video_path, config.size, config.fps)  # Shape: (total_frames, c, h, w)

    # 4. Extract the BGM from the reference video
    reference_video_path = config.reference_video
    reference_video = VideoFileClip(reference_video_path)
    bgm_audio = reference_video.audio
    bgm_duration = bgm_audio.duration

    mashup = []
    used_video_clips = []

    # Set up tokenizer for text processing
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    # 5. Process each interval and find the best video segment
    for item in tqdm(intervals_data, desc='Processing intervals'):
        start_time = item['start_time']
        end_time = item['end_time']
        caption = item['caption']

        t = end_time - start_time
        n = int(total_time // t)
        seg_frames = int(t * config.fps)

        # Tokenize the caption
        text_input = tokenizer(caption, return_tensors='pt', padding='max_length', truncation=True, max_length=77).to(config.device)

        # Extract segments in parallel
        with ThreadPoolExecutor(max_workers=12) as executor:
            futures = [executor.submit(extract_segment, i, seg_frames, all_video_frames, config.n_frames) for i in range(n)]
            video_seg_frames = [future.result() for future in futures]

        # Stack all frames
        video_seg_frames = np.array(video_seg_frames)
        video_seg_frames = torch.from_numpy(video_seg_frames).permute(0, 1, 4, 2, 3).to(config.device)

        with torch.no_grad():
            video_features = model.vision_model(video_seg_frames)
            video_features = video_features / video_features.norm(dim=-1, keepdim=True)

            text_features = model.text_model(text_input)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            logit_scale = model.logit_scale.exp()
            similarity = logit_scale * text_features @ video_features.t()
            sorted_idx = torch.argsort(similarity, dim=1, descending=True).squeeze()

        fit = False
        for idx in sorted_idx:
            idx = idx.item()
            candidate_start = round(idx * t, 2)
            candidate_end = min(round((idx + 1) * t, 2), total_time)
            if not any(intervals_overlap(candidate_start, candidate_end, used_start, used_end) for used_start, used_end in used_video_clips):
                used_video_clips.append((candidate_start, candidate_end))
                fit = True
                break

        if not fit:
            idx = sorted_idx[0].item()
            candidate_start = idx * t
            candidate_end = min((idx + 1) * t, total_time)
            used_video_clips.append((candidate_start, candidate_end))

        mashup.append(raw_video.subclip(candidate_start, candidate_end))

    # 6. Generate the final mashup video
    generate_final_video(mashup, bgm_audio, bgm_duration, os.path.join(config.data_dir, config.out_mash_video))

def generate_final_video(selected_clips, bgm_audio, bgm_duration, output_file):
    final_video = concatenate_videoclips(selected_clips, method="compose")

    # Crop the final video to match the BGM duration
    if final_video.duration > bgm_duration:
        final_video = final_video.subclip(0, bgm_duration)

    # Set the extracted BGM as the audio track for the final video
    final_video = final_video.set_audio(bgm_audio)

    # Write the final video file
    final_video.write_videofile(output_file, audio_codec="aac", threads=12)

if __name__ == '__main__':
    main()
