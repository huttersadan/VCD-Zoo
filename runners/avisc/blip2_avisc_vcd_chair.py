import argparse
import torch
import os
import json
import os
from torchvision import transforms
from PIL import Image
from vcd_utils.vcd_add_noise import add_diffusion_noise
from vcd_utils.vcd_blip2_generate import evolve_vcd_sampling_blip2
evolve_vcd_sampling_blip2()
import matplotlib.pyplot as plt
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration

from vcd_utils.vcd_add_noise import add_diffusion_noise
import tqdm

from accelerate import Accelerator
from accelerate.utils import gather_object

accelerator = Accelerator()
UNIFIED_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = os.path.dirname(UNIFIED_ROOT)
OUTPUT_ROOT = os.path.join(UNIFIED_ROOT, "outputs")
SAMPLE_LIMIT = int(os.environ.get("VCD_SAMPLE_LIMIT", "0") or 0)

def limit_samples(items):
    return items[:SAMPLE_LIMIT] if SAMPLE_LIMIT > 0 else items

# parser
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("--max_length",type = int,default=178)
parser.add_argument("--cd_alpha",type = float,default=1.0)
parser.add_argument("--cd_beta",type = float,default=0.1)
parser.add_argument("--image_folder",type = str,default="/data/dtt/projects/SPAC/coco_dataset/image")
parser.add_argument("--batch_size",type = int,default=8)
parser.add_argument('--original', action='store_true')
parser.add_argument("--use_avisc", type=bool, default=False)
parser.add_argument("--layer_gamma", type=float, default=0.5)
parser.add_argument("--masking_scheme", type=str, default="zeros")
parser.add_argument("--lamb", type=int, default=15)
args = parser.parse_args()


# blip2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "/data/dtt/pretrain_model_or_weight/instructblip-vicuna-7b"
model = InstructBlipForConditionalGeneration.from_pretrained(model_path,device_map = {"": accelerator.process_index},torch_dtype=torch.bfloat16)
processor = InstructBlipProcessor.from_pretrained(model_path)
accelerator.wait_for_everyone()


# inference
prompt = "\nUSER: Please provide a detailed description of the image in 3 to 5 complete sentences. Mention the main objects, scene, actions, and important visual details.\nASSISTANT:"

batch_size = args.batch_size
image_paths_all = os.listdir(args.image_folder)
image_paths_all = limit_samples(image_paths_all)
with accelerator.split_between_processes(image_paths_all) as single_gpu_image_paths:
    results=[]
    for single_image_path in tqdm.tqdm(single_gpu_image_paths):
        image_id = int((single_image_path.split('/')[-1])[-10:-4])
        image_full_path = os.path.join(args.image_folder, single_image_path)
        raw_image = Image.open(image_full_path).convert("RGB")
        inputs = processor(images=raw_image, text = prompt, return_tensors="pt").to(device, torch.bfloat16)
        image_tensor = inputs['pixel_values']
        # VCD
        #image_tensor = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
        image_tensor_cd = add_diffusion_noise(image_tensor[0], 500)
        
        image_tensor_cd = None  if args.original  else image_tensor_cd.unsqueeze(0)
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                do_sample=False,
                num_beams=1,
                max_length=args.max_length,
                min_length=1,
                pixel_values_cd = image_tensor_cd,
                attention_mask_cd = inputs['attention_mask'],
                cd_beta = args.cd_beta, 
                cd_alpha = args.cd_alpha, 
                
                use_avisc=args.use_avisc,
                layer_gamma=args.layer_gamma,
                masking_scheme=args.masking_scheme,
                lamb=args.lamb,
                )
        
        generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        print(generated_text)
        results.append({"image_id":image_id,"caption":generated_text})
        torch.cuda.empty_cache()
results_gathered = gather_object(results)
if accelerator.is_main_process:
    print('\n\n\n')
    print(results_gathered)
    print('\n\n\n')
    # Write to File
    if args.original:
        type_method = "original"
    elif args.use_avisc:
        type_method = "AVISC"
    else:
        type_method = "VCD"

    output_dir = os.path.join(OUTPUT_ROOT, "chair_output", "blip2", type_method)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'captions.jsonl'),'w') as file:
        for inst in results_gathered:
            json.dump(inst,file)
            file.write('\n')

#CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch  blip2_new.py  --use_avisc True
