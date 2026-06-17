import argparse
import torch
import os
import json
import sys
import os

from PIL import Image
import math
from vcd_utils.vcd_add_noise import add_diffusion_noise
from vcd_utils.vcd_sample import evolve_vcd_sampling_llava_true,evolve_vcd_sampling_true

import matplotlib.pyplot as plt
from transformers import LlavaForConditionalGeneration,AutoProcessor
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from transformers import AutoModel, AutoTokenizer
from internvl.internvl_utils import load_image, get_model_input

import tqdm
from transformers import GenerationConfig

from accelerate import Accelerator
from accelerate.utils import gather_object
accelerator = Accelerator()
UNIFIED_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = os.path.dirname(UNIFIED_ROOT)
OUTPUT_ROOT = os.path.join(UNIFIED_ROOT, "outputs")
SAMPLE_LIMIT = int(os.environ.get("VCD_SAMPLE_LIMIT", "0") or 0)

def limit_samples(items):
    return items[:SAMPLE_LIMIT] if SAMPLE_LIMIT > 0 else items

def recorder(out, pred_list):
    NEG_WORDS = ["No", "not", "no", "NO"]
    for line in out:

        line = line.replace('.', '')
        line = line.replace(',', '')
        words = line.split(' ')
        if any(word in NEG_WORDS for word in words) or any(word.endswith("n't") for word in words):
            pred_list.append(0)
        else:
            pred_list.append(1)
    
    return pred_list

def yes_no_from_text(text):
    return "yes" if recorder([text], [])[0] == 1 else "no"


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("--max_new_tokens",type = int,default=32)
parser.add_argument("--max_length",type = int,default=178)

parser.add_argument("--cd_alpha",type = float,default=1)
parser.add_argument("--cd_beta",type = float,default=0.1)
parser.add_argument("--image_folder",type = str,default="/data/dtt/projects/SPAC/coco_dataset/image")
parser.add_argument("--batch_size",type = int,default=8)
parser.add_argument('--original', action='store_true')
parser.add_argument("--use_avisc", type=bool, default=False)
parser.add_argument("--layer_gamma", type=float, default=0.5)
parser.add_argument("--masking_scheme", type=str, default="zeros")
parser.add_argument("--lamb", type=int, default=100)
parser.add_argument("--model_name", type=str, default='blip2')
parser.add_argument('--mme_name',type=str,default='existence')
parser.add_argument("--internvl_model_path",type = str,default="/data/dtt/pretrain_model_or_weight/InternVL2-2B")
args = parser.parse_args()
args.mme_path = "/data/dtt/dataset/MME_Benchmark_release_version/" + args.mme_name

#disable_torch_init()
#from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration

# model loading
if args.model_name == 'blip2':
    # blip2
    print('InstructBlip loading')   
    evolve_vcd_sampling_true()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "/data/dtt/pretrain_model_or_weight/instructblip-vicuna-7b"
    model = InstructBlipForConditionalGeneration.from_pretrained(model_path,device_map = {"": accelerator.process_index},torch_dtype=torch.bfloat16)
    processor = InstructBlipProcessor.from_pretrained(model_path)

else:
    # llava / internvl
    print('Llava loading')
    evolve_vcd_sampling_llava_true()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model_name == 'llava':
        model_path = "/data/dtt/pretrain_model_or_weight/llava-1.5-7b-hf"
        processor = AutoProcessor.from_pretrained(model_path)
        model = LlavaForConditionalGeneration.from_pretrained(model_path,device_map={"": accelerator.process_index},torch_dtype=torch.bfloat16) # 
        generation_config = GenerationConfig(
            num_beams = 1,
            max_new_tokens = args.max_new_tokens,
            do_sample = False,
        )
    elif args.model_name == 'internvl':
        print('InternVL loading')
        model_path = args.internvl_model_path
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            load_in_4bit=True,
            device_map={"": accelerator.process_index},
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        img_context_token_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
        model.img_context_token_id = img_context_token_id
        generation_config = dict(
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            num_beams=1,
        )
    else:
        raise ValueError("unknown model_name: {}".format(args.model_name))
accelerator.wait_for_everyone()

# batch inference
batch_size = args.batch_size
image_paths_all = os.listdir(args.image_folder)


#Input processing
output_text_list = []
text_path = []
picture_path = []
for single_path in os.listdir(args.mme_path):
    if single_path[-4:] == ".txt":
        text_path.append(args.mme_path +'/' + single_path)
    else:
        picture_path.append(args.mme_path +'/' + single_path)
text_path_ls = text_path
picture_path_ls = picture_path
data_question_ls = []
data_image_ls = []
data_label_ls = []
for single_picture_path in picture_path_ls:
    temp_txt_path = single_picture_path[:-4] + '.txt'
    with open(temp_txt_path,'r') as file:
        for line in file.readlines():
            temp_question, temp_label = line.split('\t')
            data_question_ls.append(temp_question)
            data_label_ls.append(temp_label)
            data_image_ls.append(single_picture_path)
length_dataset = len(data_label_ls)
for i in range(len(data_label_ls)):
    if data_label_ls[i] == 'No':
        data_label_ls[i] = 0
    else:
        data_label_ls[i] = 1

rs_ls = [{'question':data_question_ls[idx],'image_path':data_image_ls[idx],'label':data_label_ls[idx]} for idx in range(length_dataset)]
rs_ls = limit_samples(rs_ls)

print(data_label_ls)
with accelerator.split_between_processes(rs_ls) as single_gpu_question_ls:

    write_to_file = []
    response_records = []
    idx = 0
    for inst in tqdm.tqdm(single_gpu_question_ls):
        idx +=1
        
        # load image
        image_path = inst['image_path']
        image_id = int((image_path.split('/')[-1])[-10:-4])
        raw_image = Image.open(image_path)
        raw_image = raw_image.convert("RGB")
        question = inst['question']

        # inference
        with torch.inference_mode():
            if args.model_name == 'llava':
                qu = "USER: <image>\n{} Answer yes or no only.\nASSISTANT:".format(question)
                inputs = processor(images=raw_image, text = qu, return_tensors="pt").to(device, torch.bfloat16)
                image_tensor = inputs['pixel_values'][0]
                image_tensor_cd = add_diffusion_noise(image_tensor, 500)
                images_cd = None if args.original else image_tensor_cd.unsqueeze(0)
                outputs = model.generate(
                    input_ids=inputs['input_ids'],    
                    pixel_values=inputs['pixel_values'],
                    attention_mask=inputs['attention_mask'],
                    images_cd= images_cd , 
                    generation_config=generation_config,
                    cd_beta = args.cd_beta, 
                    cd_alpha = args.cd_alpha, 
                    use_cache = True,
                    # use_avisc=args.use_avisc,
                    # layer_gamma=args.layer_gamma,
                    # masking_scheme=args.masking_scheme,
                    # lamb=args.lamb,
                    # input_ids_length = input_ids_length
                )
                output_texts = processor.batch_decode(
                    outputs, 
                    skip_special_tokens=True, 
                    clean_up_tokenization_spaces=False
                )[0].strip()
                output_text = output_texts.split("ASSISTANT:")[-1]
            elif args.model_name == 'blip2':
                qu = "USER: <image>\n{} Answer yes or no only.\nASSISTANT:".format(question)
                inputs = processor(images=raw_image, text = qu, return_tensors="pt").to(device, torch.bfloat16)
                image_tensor = inputs['pixel_values'][0]
                image_tensor_cd = add_diffusion_noise(image_tensor, 500)
                images_cd = None if args.original else image_tensor_cd.unsqueeze(0)
                output_texts = model.generate(
                    **inputs,
                    do_sample=False,
                    num_beams=1,
                    max_length=args.max_length,
                    min_length=1,
                    pixel_values_cd = images_cd,
                    attention_mask_cd = inputs['attention_mask'],
                    cd_beta = args.cd_beta, 
                    cd_alpha = args.cd_alpha, 
                    # use_avisc=args.use_avisc,
                    # layer_gamma=args.layer_gamma,
                    # masking_scheme=args.masking_scheme,
                    # lamb=args.lamb,
                    #model_name='blip'
                    )
                output_text  = processor.batch_decode(output_texts, skip_special_tokens=True)[0].strip()
            elif args.model_name == 'internvl':
                pixel_values = load_image(image_path,max_num=12).to(torch.bfloat16).cuda()
                internvl_qu = "<image>\n{} Answer yes or no only.".format(question)
                model_inputs, eos_token_id = get_model_input(pixel_values,internvl_qu,model,tokenizer)
                generation_config['eos_token_id'] = eos_token_id
                images_cd = add_diffusion_noise(pixel_values, 500)
                images_cd = None if args.original else images_cd
                outputs = model.generate(
                    input_ids=model_inputs['input_ids'].cuda(),
                    pixel_values=pixel_values,
                    attention_mask=model_inputs['attention_mask'].cuda(),
                    images_cd=images_cd,
                    cd_beta=args.cd_beta,
                    cd_alpha=args.cd_alpha,
                    **generation_config,
                )
                output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        
        
        # Label list
        label = inst['label']
        #print('label:{}'.format(label))

        # write to file
        Image_Name = image_path.split('/')[-1]
        Question = question
        Ground_Truth_Answer = 'Yes' if label == 1 else "No"
        raw_output_text = output_text
        Your_Response = yes_no_from_text(raw_output_text)
        wait_to_write = Image_Name + "\t" + Question + "\t" + Ground_Truth_Answer + "\t" + Your_Response + "\n"
        write_to_file.append(wait_to_write)
        response_records.append({
            "benchmark": "mme",
            "method": "original" if args.original else "VCD",
            "model": args.model_name,
            "mme_name": args.mme_name,
            "image_name": Image_Name,
            "image_path": image_path,
            "question": Question,
            "label": label,
            "label_text": Ground_Truth_Answer,
            "response": Your_Response,
            "raw_response": raw_output_text,
        })
        
        # if idx % 10 == 0:
        #     print(output_text)
        
        # cuda_empty
        torch.cuda.empty_cache()

# gather main process
write_to_file_gather =  gather_object(write_to_file)
response_records_gather = gather_object(response_records)

if accelerator.is_main_process:
    type_method = "original" if args.original else "VCD"
    output_dir = os.path.join(OUTPUT_ROOT, "mme_output", args.model_name, type_method)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "{}.txt".format(args.mme_name)),'w') as file:
        for inst in write_to_file_gather:
            file.write(inst)
    response_dir = os.path.join(output_dir, "responses")
    os.makedirs(response_dir, exist_ok=True)
    with open(os.path.join(response_dir, "{}.jsonl".format(args.mme_name)),'w') as file:
        for inst in response_records_gather:
            json.dump(inst, file, ensure_ascii=False)
            file.write('\n')
