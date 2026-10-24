import gc
import os
from argparse import ArgumentParser
import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from mmcv import Config
from mmdet.datasets import build_dataset
from tqdm import tqdm
import utils.data.const_data

## see implementation details from https://github.com/KaiChen1998/GeoDiffusion

WORK_DIR = './construction_project/GeoDiffusion/work_dirs'
DEFAULT_CKPTS = [
    'real_synth1', 
    'real_synth2', 
    'real_synth3', 
    'real_synth4', 
    'real_synth5', 
    'real_synth6', 
    'real_synth7', 
    'real_synth8', 
    'real_synth9', 
    'real_synth10', 
    'real_synth11', 
    'real_synth12',
]

def parse_args():
    p = ArgumentParser()
    p.add_argument('--ckpts', nargs='+', default=DEFAULT_CKPTS)
    p.add_argument('--work-dir', default=WORK_DIR)
    p.add_argument('--config', default='./config.py')
    p.add_argument('--split', choices=['train', 'val'], default='val')
    p.add_argument('--prompt-version', default='v1')
    p.add_argument('--num-bucket-per-side', nargs=2, type=int, default=[512, 512])
    p.add_argument('--nsamples', type=int, default=1)
    p.add_argument('--cfg-scale', type=float, default=5)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--dpmsolver', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--overwrite', action='store_true')
    return p.parse_args()


def find_step(pipeline, type_name):
    for step in pipeline:
        if step['type'] == type_name:
            return step
    raise KeyError(f'{type_name} not found in pipeline')

def build_split(args):
    cfg = Config.fromfile(args.config)
    split_cfg = cfg.data[args.split]
    split_cfg.update(dict(prompt_version=args.prompt_version, num_bucket_per_side=args.num_bucket_per_side))
    find_step(split_cfg.pipeline, 'RandomFlip')['flip_ratio'] = 0.0
    width, height = find_step(split_cfg.pipeline, 'Resize')['img_scale']
    return build_dataset(split_cfg), int(width), int(height)


def load_pipe(ckpt_dir, use_dpmsolver):
    pipe = StableDiffusionPipeline.from_pretrained(
        ckpt_dir, torch_dtype=torch.float16,
        safety_checker=None, requires_safety_checker=False)
    if use_dpmsolver:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    return pipe.to('cuda')


def main():
    args = parse_args()
    dataset, width, height = build_split(args)
    print(f'Image resolution: {width} x {height}, #samples: {len(dataset)}')
    prompts = [dataset[i]['text'] for i in tqdm(range(len(dataset)), desc='prompts')]
    infos = dataset.data_infos

    for ckpt in args.ckpts:
        pipe = load_pipe(os.path.join(args.work_dir, ckpt, 'checkpoint', 'final'), args.dpmsolver)
        out_root = os.path.join(ckpt, 'output_img', f'{args.cfg_scale:g}')

        for i, prompt in enumerate(tqdm(prompts, desc=ckpt)):
            info = infos[i]
            stem = os.path.splitext(info['file_name'])[0]
            out_paths = [os.path.join(out_root, f'{stem}_{k}.jpg') for k in range(args.nsamples)]
            if not args.overwrite and all(os.path.exists(p) for p in out_paths):
                continue

            gen = torch.Generator('cuda').manual_seed(args.seed + i)
            images = pipe([prompt] * args.nsamples,
                        guidance_scale=args.cfg_scale,
                        num_inference_steps=args.steps,
                        height=height, width=width,
                        generator=gen).images

            os.makedirs(os.path.dirname(out_paths[0]), exist_ok=True)
            for image, path in zip(images, out_paths):
                image.resize((info['width'], info['height'])).save(path)

        del pipe
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
