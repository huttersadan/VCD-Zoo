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
from transformers import LlavaForConditionalGeneration,AutoProcessor
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration

import tqdm
from transformers import GenerationConfig
from internvl.internvl_utils import split_model,load_image,get_model_input
from transformers import AutoModel, AutoTokenizer
from accelerate import Accelerator
from accelerate.utils import gather_object
accelerator = Accelerator()
#from transformers import AutoProcessor, AutoModelForCausalLM
UNIFIED_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(UNIFIED_ROOT)
OUTPUT_ROOT = os.path.join(UNIFIED_ROOT, "outputs")
POPE_ROOT = os.environ.get("POPE_ROOT", os.path.join(UNIFIED_ROOT, "pope_dataset"))
SAMPLE_LIMIT = int(os.environ.get("VCD_SAMPLE_LIMIT", "0") or 0)

def limit_samples(items):
    return items[:SAMPLE_LIMIT] if SAMPLE_LIMIT > 0 else items

def pope_path(*parts):
    return os.path.join(POPE_ROOT, *parts)

POPE_PATH_coco = {
    "random": pope_path("POPE", "coco", "coco_pope_random.json"),
    "popular": pope_path("POPE", "coco", "coco_pope_popular.json"),
    "adversarial": pope_path("POPE", "coco", "coco_pope_adversarial.json"),
}
POPE_PATH_aokvqa = {
    "random": pope_path("POPE", "aokvqa", "aokvqa_pope_random.json"),
    "popular": pope_path("POPE", "aokvqa", "aokvqa_pope_popular.json"),
    "adversarial": pope_path("POPE", "aokvqa", "aokvqa_pope_adversarial.json"),
}

POPE_PATH_gqa = {
    "random": pope_path("POPE", "gqa", "gqa_pope_random.json"),
    "popular": pope_path("POPE", "gqa", "gqa_pope_popular.json"),
    "adversarial": pope_path("POPE", "gqa", "gqa_pope_adversarial.json"),
}

POPE_PATH_ls = {
    'coco':POPE_PATH_coco,
    'aokvqa':POPE_PATH_aokvqa,
    'gqa':POPE_PATH_gqa
}



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
parser.add_argument("--max_length",type = int,default=128)
parser.add_argument('--do_sample', action='store_true')
parser.add_argument('--num_beams', type=int, default=1)
parser.add_argument("--cd_alpha",type = float,default=1.0)
parser.add_argument("--cd_beta",type = float,default=0.1)
parser.add_argument("--type_question",type=str,default='popular')
parser.add_argument('--type_dataset',type = str,default='coco')
parser.add_argument("--batch_size",type = int,default=8)
parser.add_argument('--original', action='store_true')
parser.add_argument("--use_avisc", type=bool, default=False)
parser.add_argument("--layer_gamma", type=float, default=0.5)
parser.add_argument("--masking_scheme", type=str, default="zeros")
parser.add_argument("--lamb", type=int, default=100)
parser.add_argument("--model_name", type=str, default='internvl')
parser.add_argument("--internvl_model_path",type = str,default="/data/dtt/pretrain_model_or_weight/InternVL2-2B")
parser.add_argument('--specific_name', type=str, default=None)  
args = parser.parse_args()
if accelerator.is_main_process:
    print("args.original:{}".format(args.original))
    print("args.model_name:{}".format(args.model_name))
    print('args.use_avisc:{}'.format(args.use_avisc))
#disable_torch_init()
#from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
if args.use_avisc:
    from vcd_utils.vcd_sample import evolve_vcd_sampling_llava
    evolve_vcd_sampling_llava()
    from vcd_utils.vcd_sample import internvl_forward
else:
    from vcd_utils.vcd_sample import evolve_vcd_sampling_llava_true
    evolve_vcd_sampling_llava_true()
# model loading
if args.model_name == 'blip2':
    # blip2
    print('InstructBlip loading')   
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "/data/dtt/pretrain_model_or_weight/instructblip-vicuna-7b"
    model = InstructBlipForConditionalGeneration.from_pretrained(model_path,device_map = {"": accelerator.process_index},torch_dtype=torch.bfloat16)
    processor = InstructBlipProcessor.from_pretrained(model_path)

elif args.model_name == 'llava':
    # llava
    print('Llava loading')
    
    model_path = "/data/dtt/pretrain_model_or_weight/llava-1.5-7b-hf"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(model_path)
    model = LlavaForConditionalGeneration.from_pretrained(model_path,device_map={"": accelerator.process_index},torch_dtype=torch.bfloat16) # 
    generation_config = GenerationConfig(
        num_beams = 1,
        max_new_tokens = args.max_new_tokens,
        do_sample = False,
    )
elif args.model_name == 'internvl':
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
    IMG_START_TOKEN='<img>'
    IMG_END_TOKEN='</img>'
    IMG_CONTEXT_TOKEN='<IMG_CONTEXT>'
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    model.img_context_token_id = img_context_token_id
    if args.use_avisc:
        model.language_model.forward = internvl_forward    
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

# batch inference
batch_size = args.batch_size

# question_ls 
question_ls = []
POPE_path = POPE_PATH_ls[args.type_dataset]
pope_path = POPE_path[args.type_question]
with open(pope_path,'r') as file:
    for line in file.readlines():
        question_ls.append(json.loads(line))
question_ls = limit_samples(question_ls)
pred_list,label_list = [],[]
root_image_path = "/data/dtt/dataset/MSCOCO/val2014/" if args.type_dataset != 'gqa' else "/data/dtt/dataset/gqa/"

with accelerator.split_between_processes(question_ls) as single_gpu_question_ls:
    pred_list, label_list = [], []
    response_records=[]
    print("{}. {}".format(args.type_dataset,args.type_question))
    idx = 0
    for inst in tqdm.tqdm(single_gpu_question_ls):
        idx +=1
        image_path = root_image_path + inst['image']
        image_id = int((image_path.split('/')[-1])[-10:-4])
        raw_image = Image.open(image_path)
        raw_image = raw_image.convert("RGB")
        


        label = inst['label']
        if label=='yes':
            label = 1
        else:
            label = 0
        label_list = label_list + [label]


        with torch.inference_mode():
            if args.model_name == 'llava':
                qu = inst['text']
                # prompt = "USER: <image>\nPlease describe the details in the picture.\nASSISTANT:"
                qu = "USER: <image>\n{} Please answer yes or no only.\nASSISTANT:".format(qu)
                inputs = processor(images=raw_image, text = qu, return_tensors="pt").to(device, torch.bfloat16)
                image_tensor = inputs['pixel_values'][0]
                image_tensor_cd = add_diffusion_noise(image_tensor, 500)
                images_cd = None if args.original else image_tensor_cd.unsqueeze(0)

                input_ids_length = inputs['input_ids'].shape[-1]
                outputs = model.generate(
                    input_ids=inputs['input_ids'],    
                    pixel_values=inputs['pixel_values'],
                    attention_mask=inputs['attention_mask'],
                    images_cd= images_cd , 
                    generation_config=generation_config,
                    cd_beta = args.cd_beta, 
                    cd_alpha = args.cd_alpha, 
                    use_cache = True,
                    use_avisc=args.use_avisc,
                    layer_gamma=args.layer_gamma,
                    masking_scheme=args.masking_scheme,
                    lamb=args.lamb,
                    input_ids_length = input_ids_length
                )
                output_texts = processor.batch_decode(
                    outputs, 
                    skip_special_tokens=True, 
                    clean_up_tokenization_spaces=False
                )[0].strip()
                output_text = output_texts.split("ASSISTANT:")[-1]
            elif args.model_name == 'blip2':
                qu = inst['text']
                # prompt = "USER: <image>\nPlease describe the details in the picture.\nASSISTANT:"
                qu = "USER: <image>\n{} Please answer yes or no only.\nASSISTANT:".format(qu)
                inputs = processor(images=raw_image, text = qu, return_tensors="pt").to(device, torch.bfloat16)
                image_tensor = inputs['pixel_values'][0]
                image_tensor_cd = add_diffusion_noise(image_tensor, 750)
                images_cd = None if args.original else image_tensor_cd.unsqueeze(0)

                input_ids_length = inputs['input_ids'].shape[-1]
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
                    use_avisc=args.use_avisc,
                    layer_gamma=args.layer_gamma,
                    masking_scheme=args.masking_scheme,
                    lamb=args.lamb,
                    model_name='blip'
                    )
                output_text  = processor.batch_decode(output_texts, skip_special_tokens=True)[0].strip()
            elif args.model_name == 'internvl':
                pixel_values = load_image(image_path,max_num=12).to(torch.bfloat16).cuda()
                internvl_qu = "<image>\n{} Please answer yes or no only.\n".format(inst['text'])
                model_inputs, eos_token_id = get_model_input(pixel_values,internvl_qu,model,tokenizer)
                input_ids = model_inputs['input_ids'].cuda()
                attention_mask = model_inputs['attention_mask'].cuda()
                generation_config['eos_token_id'] = eos_token_id
                # VCD process
                image_tensor = pixel_values
                images_cd = add_diffusion_noise(image_tensor, 500)
                images_cd = None if args.original else images_cd
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
                output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        raw_output_text = output_text
        output_text = yes_no_from_text(raw_output_text)
        single_pred = 1 if output_text == "yes" else 0
        pred_list.append(single_pred)
        response_records.append({
            "benchmark": "pope",
            "method": "original" if args.original else ("AVISC" if args.use_avisc else "VCD"),
            "model": args.model_name,
            "type_dataset": args.type_dataset,
            "type_question": args.type_question,
            "image": inst.get("image"),
            "image_id": image_id,
            "question": inst.get("text"),
            "label": label,
            "label_text": inst.get("label"),
            "prediction": single_pred,
            "response": output_text,
            "raw_response": raw_output_text,
        })
        # if idx % 10 == 0:
        #     print(output_text)
        torch.cuda.empty_cache()
label_list = gather_object(label_list) 
pred_list = gather_object(pred_list)
response_records = gather_object(response_records)


        
if accelerator.is_main_process:
    

    # Evaluation metric
    pos = 1
    neg = 0
    yes_ratio = pred_list.count(1) / len(pred_list)
    # unknown_ratio = pred_list.count(2) / len(pred_list)
    count = 20
    TP, TN, FP, FN = 0, 0, 0, 0
    rs_list = []
    for pred, label in zip(pred_list, label_list):
        if pred == pos and label == pos:
            TP += 1
        elif pred == pos and label == neg:
            FP += 1
        elif pred == neg and label == neg:
            TN += 1
        elif pred == neg and label == pos:
            FN += 1

    print('TP\tFP\tTN\tFN\t')
    print('{}\t{}\t{}\t{}'.format(TP, FP, TN, FN))

    precision = float(TP) / float(TP + FP)
    recall = float(TP) / float(TP + FN)
    f1 = 2*precision*recall / (precision + recall)
    acc = (TP + TN) / (TP + TN + FP + FN)
    acc_txt = 'Accuracy: {}'.format(acc)
    precision_txt = 'Precision: {}'.format(precision)
    recall_txt = 'Recall: {}'.format(recall)
    F1_txt = 'F1 score: {}'.format(f1)
    yes_ratio = 'Yes ratio: {}'.format(yes_ratio)
    print('Accuracy: {}'.format(acc))
    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall))
    print('F1 score: {}'.format(f1))
    print('Yes ratio: {}'.format(yes_ratio))

    if args.original:
        type_method = "original"
    elif args.use_avisc:
        type_method = "AVISC"
    else:
        type_method = "VCD"
    subset_name = "{}_{}".format(args.type_dataset, args.type_question)
    output_dir = os.path.join(OUTPUT_ROOT, "pope_output", args.model_name, type_method, subset_name)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "results.txt")
    with open(output_path,'w') as file:
        file.write(acc_txt + '\n')
        file.write(precision_txt+'\n')
        file.write(recall_txt+'\n')
        file.write(F1_txt+'\n')
        file.write(yes_ratio+'\n')
    response_path = os.path.join(output_dir, "responses.jsonl")
    with open(response_path, 'w') as file:
        for inst in response_records:
            json.dump(inst, file, ensure_ascii=False)
            file.write('\n')
