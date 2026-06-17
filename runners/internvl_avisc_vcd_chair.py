import argparse
import torch
import os
import json
import sys
import os
from PIL import Image
import math
from vcd_utils.vcd_add_noise import add_diffusion_noise

import matplotlib.pyplot as plt
from internvl.internvl_utils import split_model,load_image,get_model_input
from transformers import AutoModel, AutoTokenizer
from transformers import GenerationConfig
import tqdm
from accelerate import Accelerator
from accelerate.utils import gather_object

accelerator = Accelerator()
UNIFIED_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(UNIFIED_ROOT)
OUTPUT_ROOT = os.path.join(UNIFIED_ROOT, "outputs")
SAMPLE_LIMIT = int(os.environ.get("VCD_SAMPLE_LIMIT", "0") or 0)

def limit_samples(items):
    return items[:SAMPLE_LIMIT] if SAMPLE_LIMIT > 0 else items

# parser
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("--cd_alpha",type = float,default=1.0)
parser.add_argument("--cd_beta",type = float,default=0.1)
parser.add_argument("--internvl_model_path",type = str,default="/data/dtt/pretrain_model_or_weight/InternVL2-2B")

parser.add_argument("--image_folder",type = str,default="/data/dtt/projects/SPAC/coco_dataset/image")
parser.add_argument("--batch_size",type = int,default=8)
parser.add_argument('--original', action='store_true')
parser.add_argument("--use_avisc", type=bool, default=False)
parser.add_argument("--layer_gamma", type=float, default=0.5)
parser.add_argument("--masking_scheme", type=str, default="zeros")
parser.add_argument("--lamb", type=int, default=100)
# llava config
parser.add_argument("--max_new_tokens",type = int,default=128)

# blip2 config
parser.add_argument("--max_length",type = int,default=256)
# shared config
parser.add_argument('--num_beams', type=int, default=1)
parser.add_argument('--do_sample', action='store_true')


parser.add_argument('--specific_name', type=str, default='2B')  
args = parser.parse_args()

# model loading
model_path = args.internvl_model_path
if args.use_avisc:
    from vcd_utils.vcd_sample import evolve_vcd_sampling_llava
    evolve_vcd_sampling_llava()
    from vcd_utils.vcd_sample import internvl_forward
else:
    from vcd_utils.vcd_sample import evolve_vcd_sampling_llava_true
    evolve_vcd_sampling_llava_true()
    
#device_map = split_model('InternVL2-2B')
model = AutoModel.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    device_map={"": accelerator.process_index},
    ).eval()
if args.original ==False:
    model.language_model.forward = internvl_forward    
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
IMG_START_TOKEN='<img>'
IMG_END_TOKEN='</img>'
IMG_CONTEXT_TOKEN='<IMG_CONTEXT>'
img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
model.img_context_token_id = img_context_token_id
if args.use_avisc:

    generation_config = dict(
        max_new_tokens=args.max_new_tokens, 
        do_sample=True,
        num_beams=1,
        )
else:
    generation_config = dict(
        max_new_tokens=args.max_new_tokens, 
        do_sample=False,
        num_beams=args.num_beams,
        )
accelerator.wait_for_everyone()

# prompt
prompt = "<image>\nPlease provide a detailed description of the image in 3 to 5 complete sentences. Mention the main objects, scene, actions, and important visual details."
# batch inference

batch_size = args.batch_size
image_paths_all = os.listdir(args.image_folder)[:500]
image_paths_all = limit_samples(image_paths_all)

with accelerator.split_between_processes(image_paths_all) as single_gpu_image_paths:
    results=[]
    # single_gpu
    for single_image_path in tqdm.tqdm(single_gpu_image_paths):
        image_id = int((single_image_path.split('/')[-1])[-10:-4])
        image_full_path = os.path.join(args.image_folder, single_image_path)
        #raw_image = Image.open(image_full_path).convert("RGB")
        #inputs = processor(images=raw_image, text = prompt, return_tensors="pt").to(device, torch.bfloat16)
        pixel_values = load_image(image_full_path,max_num=12).to(torch.bfloat16).cuda()
        model_inputs, eos_token_id = get_model_input(pixel_values,prompt,model,tokenizer)
        input_ids = model_inputs['input_ids'].cuda()
        attention_mask = model_inputs['attention_mask'].cuda()
        generation_config['eos_token_id'] = eos_token_id
        # VCD process
        image_tensor = pixel_values
        images_cd = add_diffusion_noise(image_tensor, 500)
        images_cd = None if args.original else images_cd
        #print('pixel_values.shape:{}'.format(pixel_values.shape))
        #print('input_ids.shape:{}'.format(input_ids.shape))
        # inference
        model.eval()
        if args.original:
            with torch.no_grad():
                generation_config['do_sample'] = True
                generation_config['num_beams'] = 3
                outputs = model.generate(
                    input_ids=input_ids,    
                    pixel_values=pixel_values,
                    attention_mask=attention_mask,
                    **generation_config,
                )
        else:
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,    
                    pixel_values=pixel_values,
                    attention_mask=attention_mask,
                    images_cd=images_cd,
                    cd_beta=args.cd_beta, 
                    cd_alpha=args.cd_alpha, 
                    
                    use_avisc=args.use_avisc,
                    layer_gamma=args.layer_gamma,
                    masking_scheme=args.masking_scheme,
                    lamb=args.lamb,
                    **generation_config,
                )
        
        # output text
        output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        print(output_text)
        results.append({'image_id':image_id,"caption":output_text})
        torch.cuda.empty_cache()
results_gathered=gather_object(results)

if accelerator.is_main_process:
    print('\n\n\n')
    print(results_gathered)
    print('\n\n\n')
    
    if args.original:
        type_method = "original"
    elif args.use_avisc:
        type_method = "AVISC"
    else:
        type_method = "VCD"
    specific_name = args.specific_name  
    output_dir = os.path.join(OUTPUT_ROOT, "chair_output", "internvl", type_method)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'captions.jsonl'),'w') as file:
        for inst in results_gathered:
            json.dump(inst,file)
            file.write('\n')
# # write to file

#CUDA_VISIBLE_DEVICES=2,3,4,5 accelerate launch --config_file accelerate_config/default_config.yaml internvl_avisc_vcd_chair.py --use_avisc True 

#CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --config_file default_config.yaml internvl_avisc_vcd_chair.py --use_avisc False --original
