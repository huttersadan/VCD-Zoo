import argparse
import torch
import os
import json
import sys
import os
from PIL import Image
import math
from vcd_utils.vcd_add_noise import add_diffusion_noise
from vcd_utils.vcd_sample import evolve_vcd_sampling_llava
import matplotlib.pyplot as plt
from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers import GenerationConfig
import tqdm
from accelerate import Accelerator
from accelerate.utils import gather_object
evolve_vcd_sampling_llava()
accelerator = Accelerator()
UNIFIED_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(UNIFIED_ROOT)
OUTPUT_ROOT = os.path.join(UNIFIED_ROOT, "outputs")
SAMPLE_LIMIT = int(os.environ.get("VCD_SAMPLE_LIMIT", "0") or 0)

def limit_samples(items):
    return items[:SAMPLE_LIMIT] if SAMPLE_LIMIT > 0 else items

# parser
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("--max_new_tokens",type = int,default=128)
parser.add_argument("--cd_alpha",type = float,default=1.0)
parser.add_argument("--cd_beta",type = float,default=0.1)
parser.add_argument("--image_folder",type = str,default="/data/dtt/projects/SPAC/coco_dataset/image")
parser.add_argument("--batch_size",type = int,default=8)
parser.add_argument('--original', action='store_true')
parser.add_argument("--use_avisc", type=bool, default=False)
parser.add_argument("--layer_gamma", type=float, default=0.5)
parser.add_argument("--masking_scheme", type=str, default="zeros")
parser.add_argument("--lamb", type=int, default=100)
args = parser.parse_args()

# model loading
model_path = "/data/dtt/pretrain_model_or_weight/llava-1.5-7b-hf"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoProcessor.from_pretrained(model_path)
model = LlavaForConditionalGeneration.from_pretrained(model_path,device_map={"": accelerator.process_index},torch_dtype=torch.bfloat16) # 
accelerator.wait_for_everyone()

# prompt
prompt = "USER: <image>\nPlease provide a detailed description of the image in 3 to 5 complete sentences. Mention the main objects, scene, actions, and important visual details.\nASSISTANT:"

# batch inference
generation_config = GenerationConfig(
        num_beams = 1,
        max_new_tokens = args.max_new_tokens,
        do_sample = True,
    )
batch_size = args.batch_size
image_paths_all = os.listdir(args.image_folder)
image_paths_all = limit_samples(image_paths_all)

with accelerator.split_between_processes(image_paths_all) as single_gpu_image_paths:
    results=[]
    # single_gpu
    for single_image_path in tqdm.tqdm(single_gpu_image_paths):
        image_id = int((single_image_path.split('/')[-1])[-10:-4])
        image_full_path = os.path.join(args.image_folder, single_image_path)
        raw_image = Image.open(image_full_path).convert("RGB")
        inputs = processor(images=raw_image, text = prompt, return_tensors="pt").to(device, torch.bfloat16)
        
        # VCD process
        image_tensor = inputs['pixel_values'][0]
        images_cd = add_diffusion_noise(image_tensor, 500)
        images_cd = None if args.original else images_cd.unsqueeze(0)
        # inference
        model.eval()
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs['input_ids'],    
                pixel_values=inputs['pixel_values'],
                attention_mask=inputs['attention_mask'],
                images_cd=images_cd,
                cd_beta=args.cd_beta, 
                cd_alpha=args.cd_alpha, 
                generation_config=generation_config,
                use_avisc=args.use_avisc,
                layer_gamma=args.layer_gamma,
                masking_scheme=args.masking_scheme,
                lamb=args.lamb,
            )
        output_texts = processor.batch_decode(
            outputs,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        output_text = output_texts.split("ASSISTANT:")[-1]
        results.append({'image_id': image_id, "caption": output_text})
        torch.cuda.empty_cache()

results_gathered = gather_object(results)

if accelerator.is_main_process:
    if args.original:
        type_method = "original"
    elif args.use_avisc:
        type_method = "AVISC"
    else:
        type_method = "VCD"

    output_dir = os.path.join(OUTPUT_ROOT, "chair_output", "llava", type_method)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'captions.jsonl'), 'w') as file:
        for inst in results_gathered:
            json.dump(inst, file)
            file.write('\n')
