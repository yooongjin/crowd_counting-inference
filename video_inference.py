import torch
import decord
import json
import argparse
import cv2
import os

import torch.nn.functional as F
import numpy as np
# from mmcv import Config, DictAction
# from lib.models.build_counter import Baseline_Counter
# from lib.utils.points_from_den import local_maximum_points

MEAN=[0.485, 0.456, 0.406]
STD=[0.229, 0.224, 0.225]

def argument_parser():
    parser = argparse.ArgumentParser(description="Video Inference Script")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the video file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model file")
    parser.add_argument("--output_path", type=str, default='output', help="Path to save the output results")
    parser.add_argument("--sample_rate", type=int, default=5, help="Sample rate for frames")
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size for processing frames")

    args = parser.parse_args()
    
    return args

def local_maximum_points(density_map, gaussian_maximum,radius=8.,patch_size=128, den_scale=1., threshold=0.15):
    _,_,h,w = density_map.shape
    # kernel = torch.ones(3,3)/9.
    # kernel =kernel.unsqueeze(0).unsqueeze(0).cuda()
    # weight = nn.Parameter(data=kernel, requires_grad=False)
    # density_map = F.conv2d(density_map, weight, stride=1, padding=1)


    # import pdb
    # pdb.set_trace()
    if h % patch_size != 0:
        pad_dims = (0, 0, 0, patch_size - h % patch_size)
        h = (h // patch_size + 1) * patch_size
        density_map = F.pad(density_map, pad_dims, "constant")


    if w % patch_size != 0:
        pad_dims = (0, patch_size - w % patch_size, 0, 0)
        w = (w // patch_size + 1) * patch_size
        density_map = F.pad(density_map, pad_dims, "constant")


    local_max = F.max_pool2d(density_map, (patch_size, patch_size), stride=patch_size)
    local_max = local_max*threshold
    local_max[local_max<threshold*gaussian_maximum] = threshold*gaussian_maximum
    local_max[local_max>0.3*gaussian_maximum] = 0.3*gaussian_maximum

    local_max = F.interpolate(local_max, scale_factor=patch_size)

    keep = F.max_pool2d(density_map, (3, 3), stride=1, padding=1)
    # keep = F.interpolate(keep, scale_factor=2)
    keep = (keep == density_map).float()
    density_map = keep * density_map

    density_map[density_map < local_max] = 0
    density_map[density_map > 0] = 1
    count = int(torch.sum(density_map).item())

    points = torch.nonzero(density_map)[:,[0,1,3,2]].float() # b,c,h,w->b,c,w,h
    rois = torch.zeros((points.size(0), 5)).float().to(density_map)
    rois[:, 0] = points[:, 0]
    rois[:, 1] = torch.clamp(points[:, 2] - radius, min=0)
    rois[:, 2] = torch.clamp(points[:, 3] - radius, min=0)
    rois[:, 3] = torch.clamp(points[:, 2] + radius, max=w)
    rois[:, 4] = torch.clamp(points[:, 3] + radius, max=h)

    pre_data = {'num': count, 'points': points[:,2:].cpu().numpy()*den_scale, 'rois': rois.cpu().numpy()}
    return pre_data

def input_transform(frames):
    frames = frames[:, :, :, [2, 1, 0]] 
    frames = frames / 255.0
    frames -= torch.tensor(MEAN, dtype=torch.float32).view(1, 1, 1, 3)
    frames /= torch.tensor(STD, dtype=torch.float32).view(1, 1, 1, 3)
    return frames

def check_frames(frames, divisor):
    h, w = frames.shape[1:3]
    if h % divisor != 0:
        real_h = h + (divisor - h % divisor)
    else:
        real_h = h
    if w % divisor != 0:
        real_w = w + (divisor - w % divisor)
    else:
        real_w = w
    
    padded_frames = torch.zeros((frames.shape[0], real_h, real_w, frames.shape[3]), dtype=torch.float32)
    padded_frames[:, :h, :w, :] = torch.tensor(frames, dtype=torch.float32)
    
    return padded_frames, h, w

def normalize(x):
    min_val = x.min()
    max_val = x.max()
    if max_val - min_val == 0:
        return x
    return (x - min_val) / (max_val - min_val)
    

if __name__ == "__main__":
    args = argument_parser()
    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(os.path.join(args.output_path, 'density_map'), exist_ok=True)
    os.makedirs(os.path.join(args.output_path, 'frames'), exist_ok=True)
    
    model = torch.jit.load(args.checkpoint)
    model.eval()
    model = model.cuda()
    print("Model loaded successfully.")
    
    video_reader = decord.VideoReader(args.video_path, ctx=decord.cpu(0))
    num_frames = len(video_reader)
    print(f"Video loaded: {args.video_path}")
    print(f"Total frames in video: {num_frames}")
    
    decord_frame_indices = list(range(0, num_frames, args.sample_rate))
    frames = video_reader.get_batch(decord_frame_indices).asnumpy()
    for i, frame in enumerate(frames):
        frame_filename = os.path.join(args.output_path, 'frames', f"frame_{i+1}.png")
        cv2.imwrite(frame_filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    print(f"Number of frames sampled: {len(frames)}")
    print(f"Index of sampled frames: {decord_frame_indices}")

    frames = torch.tensor(frames[:, :, :, [2, 1, 0]])
    # frames, h, w = check_frames(frames, 32)
    # frames = input_transform(frames)
    # frames = frames.permute(0, 3, 1, 2) 
    # print(f"Frames shape after transformation: {frames.shape}")
    
    outputs = []
    print("Starting inference...")
    with torch.no_grad():
        for i in range(0, len(frames), args.batch_size):
            batch_frames = frames[i:i + args.batch_size]
            batch_frames = batch_frames.cuda()
            if batch_frames.shape[0] == 0:
                continue
            
            print(f"Processing batch {i // args.batch_size + 1}, frames {i} to {i + args.batch_size - 1}")
            output = model(batch_frames)
            output = output.cpu().numpy()
            outputs.append(output)
            
    outputs = np.concatenate(outputs, axis=0)
    print(f"Inference completed. Results shape: {outputs.shape}")

    print(f"Saving results to {args.output_path}")    
    results = {"video_id" : os.path.basename(args.video_path),
              "frames" : []}
    
    for i, output in enumerate(outputs):
        norm_output = normalize(output).squeeze()        
        heatmap = cv2.applyColorMap((norm_output * 255).astype(np.uint8), cv2.COLORMAP_JET)
        output_density_filename = os.path.join(args.output_path, 'density_map', f"frame_d_{i+1}.png")
        cv2.imwrite(output_density_filename, heatmap)
        
        pred_points = local_maximum_points(torch.tensor(output[np.newaxis, ...]),model.gaussian_maximum.clone().cpu(), patch_size=32,threshold=0.2)
        results["frames"].append({
            "frame_no": i+1,
            "crowd_count": pred_points['num'],
            "location": pred_points['points'].tolist(),
            "heatmap": output_density_filename
        })
        
    results_filename = os.path.join(args.output_path, "results.json")
    with open(results_filename, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {results_filename}")