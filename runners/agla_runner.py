import argparse
import json
import os
import sys
from pathlib import Path

import torch
import tqdm
from PIL import Image
from torchvision import transforms
from transformers import (
    AutoModel,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    InstructBlipForConditionalGeneration,
    InstructBlipProcessor,
    LlavaForConditionalGeneration,
    set_seed,
)

UNIFIED_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(UNIFIED_ROOT))

from agla_utils.augmentation import augmentation
from internvl.internvl_utils import get_model_input, load_image, load_image_no_path
from lavis.models import load_model_and_preprocess


OUTPUT_ROOT = UNIFIED_ROOT / "outputs"
POPE_ROOT = Path(os.environ.get("POPE_ROOT", UNIFIED_ROOT / "pope_dataset"))
SAMPLE_LIMIT = int(os.environ.get("VCD_SAMPLE_LIMIT", "0") or 0)

LLAVA_MODEL_PATH = "/data/dtt/pretrain_model_or_weight/llava-1.5-7b-hf"
BLIP2_MODEL_PATH = "/data/dtt/pretrain_model_or_weight/instructblip-vicuna-7b"
COCO_IMAGE_ROOT = "/data/dtt/dataset/MSCOCO/val2014"
GQA_IMAGE_ROOT = "/data/dtt/dataset/gqa"
MME_ROOT = "/data/dtt/dataset/MME_Benchmark_release_version"

POPE_PATHS = {
    "coco": {
        "random": POPE_ROOT / "POPE" / "coco" / "coco_pope_random.json",
        "popular": POPE_ROOT / "POPE" / "coco" / "coco_pope_popular.json",
        "adversarial": POPE_ROOT / "POPE" / "coco" / "coco_pope_adversarial.json",
    },
    "aokvqa": {
        "random": POPE_ROOT / "POPE" / "aokvqa" / "aokvqa_pope_random.json",
        "popular": POPE_ROOT / "POPE" / "aokvqa" / "aokvqa_pope_popular.json",
        "adversarial": POPE_ROOT / "POPE" / "aokvqa" / "aokvqa_pope_adversarial.json",
    },
    "gqa": {
        "random": POPE_ROOT / "POPE" / "gqa" / "gqa_pope_random.json",
        "popular": POPE_ROOT / "POPE" / "gqa" / "gqa_pope_popular.json",
        "adversarial": POPE_ROOT / "POPE" / "gqa" / "gqa_pope_adversarial.json",
    },
}


def limit_samples(items):
    return items[:SAMPLE_LIMIT] if SAMPLE_LIMIT > 0 else items


def yes_no_from_text(text):
    words = text.replace(".", " ").replace(",", " ").split()
    neg_words = {"No", "no", "NO", "not"}
    return "no" if any(word in neg_words or word.endswith("n't") for word in words) else "yes"


def safe_div(num, denom):
    return float(num) / float(denom) if denom else 0.0


def write_pope_metrics(output_dir, pred_list, label_list):
    pos, neg = 1, 0
    tp = fp = tn = fn = 0
    for pred, label in zip(pred_list, label_list):
        if pred == pos and label == pos:
            tp += 1
        elif pred == pos and label == neg:
            fp += 1
        elif pred == neg and label == neg:
            tn += 1
        elif pred == neg and label == pos:
            fn += 1

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    acc = safe_div(tp + tn, tp + tn + fp + fn)
    yes_ratio = safe_div(pred_list.count(1), len(pred_list))

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "results.txt").open("w") as file:
        file.write(f"Accuracy: {acc}\n")
        file.write(f"Precision: {precision}\n")
        file.write(f"Recall: {recall}\n")
        file.write(f"F1 score: {f1}\n")
        file.write(f"Yes ratio: {yes_ratio}\n")


def load_agla_aux(device):
    model_itm, vis_processors, text_processors = load_model_and_preprocess(
        "blip_image_text_matching", "large", device=device, is_eval=True
    )
    return model_itm, vis_processors, text_processors, transforms.Compose([transforms.ToTensor()])


def build_augmented_image(raw_image, prompt, device, aux):
    model_itm, vis_processors, text_processors, loader = aux
    tensor_image = loader(raw_image.resize((384, 384)))
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    question = text_processors["eval"](prompt)
    tokenized_text = model_itm.tokenizer(
        question, padding="longest", truncation=True, return_tensors="pt"
    ).to(device)
    return augmentation(image, question, tensor_image, model_itm, tokenized_text, raw_image)


def load_model(model_name, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_name == "llava":
        from agla_utils.llava_sample import evolve_agla_sampling

        evolve_agla_sampling()
        processor = AutoProcessor.from_pretrained(LLAVA_MODEL_PATH)
        model = LlavaForConditionalGeneration.from_pretrained(
            LLAVA_MODEL_PATH, device_map="auto", torch_dtype=torch.bfloat16
        ).eval()
        generation_config = GenerationConfig(
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
        )
        return model, processor, device, generation_config

    if model_name == "blip2":
        from agla_utils.blip2_sample import evolve_agla_sampling

        evolve_agla_sampling()
        processor = InstructBlipProcessor.from_pretrained(BLIP2_MODEL_PATH)
        model = InstructBlipForConditionalGeneration.from_pretrained(
            BLIP2_MODEL_PATH, device_map="auto", torch_dtype=torch.bfloat16
        ).eval()
        return model, processor, device, None

    if model_name == "internvl":
        from agla_utils.llava_sample import evolve_agla_sampling

        evolve_agla_sampling()
        model = AutoModel.from_pretrained(
            args.internvl_model_path,
            torch_dtype=torch.bfloat16,
            load_in_4bit=True,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="auto",
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(
            args.internvl_model_path, trust_remote_code=True, use_fast=False
        )
        img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        model.img_context_token_id = img_context_token_id
        generation_config = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.do_sample,
            "num_beams": args.num_beams,
        }
        return model, tokenizer, device, generation_config

    raise ValueError(f"unknown model: {model_name}")


def generate_llava(model, processor, device, generation_config, raw_image, prompt, aux, args):
    inputs = processor(images=raw_image, text=prompt, return_tensors="pt").to(device, torch.bfloat16)
    augmented_image = build_augmented_image(raw_image, prompt, device, aux)
    images_cd = processor(images=augmented_image, text=prompt, return_tensors="pt").to(
        device, torch.float16
    )["pixel_values"]
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            attention_mask=inputs["attention_mask"],
            images_cd=images_cd.half(),
            cd_alpha=args.cd_alpha,
            cd_beta=args.cd_beta,
            use_cache=True,
            generation_config=generation_config,
        )
    decoded = processor.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return decoded.split("ASSISTANT:")[-1].strip()


def generate_blip2(model, processor, device, raw_image, prompt, aux, args):
    inputs = processor(images=raw_image, text=prompt, return_tensors="pt").to(device, torch.bfloat16)
    augmented_image = build_augmented_image(raw_image, prompt, device, aux)
    images_cd = processor(images=augmented_image, text=prompt, return_tensors="pt").to(
        device, torch.float16
    )["pixel_values"]
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=False,
            num_beams=args.num_beams,
            max_length=args.max_length,
            min_length=1,
            pixel_values_cd=images_cd.half(),
            attention_mask_cd=inputs["attention_mask"],
            cd_alpha=args.cd_alpha,
            cd_beta=args.cd_beta,
        )
    return processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()


def generate_internvl(model, tokenizer, device, generation_config, raw_image, image_path, prompt, aux, args):
    pixel_values = load_image(str(image_path), max_num=12).to(torch.bfloat16).cuda()
    model_inputs, eos_token_id = get_model_input(pixel_values, prompt, model, tokenizer)
    augmented_image = build_augmented_image(raw_image, prompt, device, aux)
    images_cd = load_image_no_path(augmented_image, max_num=12).to(torch.bfloat16).cuda()
    run_config = dict(generation_config)
    run_config["eos_token_id"] = eos_token_id
    with torch.no_grad():
        outputs = model.generate(
            input_ids=model_inputs["input_ids"].cuda(),
            pixel_values=pixel_values,
            attention_mask=model_inputs["attention_mask"].cuda(),
            images_cd=images_cd,
            cd_alpha=args.cd_alpha,
            cd_beta=args.cd_beta,
            **run_config,
        )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()


def generate_response(model_pack, model_name, raw_image, image_path, prompt, aux, args):
    model, processor_or_tokenizer, device, generation_config = model_pack
    if model_name == "llava":
        return generate_llava(model, processor_or_tokenizer, device, generation_config, raw_image, prompt, aux, args)
    if model_name == "blip2":
        return generate_blip2(model, processor_or_tokenizer, device, raw_image, prompt, aux, args)
    return generate_internvl(model, processor_or_tokenizer, device, generation_config, raw_image, image_path, prompt, aux, args)


def run_chair(args):
    model_pack = load_model(args.model_name, args)
    aux = load_agla_aux(model_pack[2])
    prompt = {
        "llava": "USER: <image>\nPlease provide a detailed description of the image in 3 to 5 complete sentences. Mention the main objects, scene, actions, and important visual details.\nASSISTANT:",
        "blip2": "\nUSER: Please provide a detailed description of the image in 3 to 5 complete sentences. Mention the main objects, scene, actions, and important visual details.\nASSISTANT:",
        "internvl": "<image>\nPlease provide a detailed description of the image in 3 to 5 complete sentences. Mention the main objects, scene, actions, and important visual details.",
    }[args.model_name]
    image_paths = limit_samples(sorted(os.listdir(args.image_folder)))
    results = []
    for image_name in tqdm.tqdm(image_paths):
        image_id = int(image_name[-10:-4])
        image_path = Path(args.image_folder) / image_name
        raw_image = Image.open(image_path).convert("RGB")
        caption = generate_response(model_pack, args.model_name, raw_image, image_path, prompt, aux, args)
        results.append({"image_id": image_id, "caption": caption})
        torch.cuda.empty_cache()

    output_dir = OUTPUT_ROOT / "chair_output" / args.model_name / "AGLA"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "captions.jsonl").open("w") as file:
        for item in results:
            json.dump(item, file, ensure_ascii=False)
            file.write("\n")


def run_pope(args):
    model_pack = load_model(args.model_name, args)
    aux = load_agla_aux(model_pack[2])
    question_path = POPE_PATHS[args.type_dataset][args.type_question]
    with question_path.open() as file:
        questions = limit_samples([json.loads(line) for line in file])

    root_image_path = Path(GQA_IMAGE_ROOT if args.type_dataset == "gqa" else COCO_IMAGE_ROOT)
    pred_list, label_list, responses = [], [], []
    for inst in tqdm.tqdm(questions):
        image_path = root_image_path / inst["image"]
        image_id = int(str(image_path)[-10:-4])
        raw_image = Image.open(image_path).convert("RGB")
        prompt = {
            "llava": f"USER: <image>\n{inst['text']} Please answer yes or no only.\nASSISTANT:",
            "blip2": f"USER: {inst['text']} Please answer yes or no only.\nASSISTANT:",
            "internvl": f"<image>\n{inst['text']} Please answer yes or no only.",
        }[args.model_name]
        raw_response = generate_response(model_pack, args.model_name, raw_image, image_path, prompt, aux, args)
        response = yes_no_from_text(raw_response)
        pred = 1 if response == "yes" else 0
        label = 1 if inst["label"] == "yes" else 0
        pred_list.append(pred)
        label_list.append(label)
        responses.append(
            {
                "benchmark": "pope",
                "method": "AGLA",
                "model": args.model_name,
                "type_dataset": args.type_dataset,
                "type_question": args.type_question,
                "image": inst.get("image"),
                "image_id": image_id,
                "question": inst.get("text"),
                "label": label,
                "label_text": inst.get("label"),
                "prediction": pred,
                "response": response,
                "raw_response": raw_response,
            }
        )
        torch.cuda.empty_cache()

    output_dir = OUTPUT_ROOT / "pope_output" / args.model_name / "AGLA" / f"{args.type_dataset}_{args.type_question}"
    write_pope_metrics(output_dir, pred_list, label_list)
    with (output_dir / "responses.jsonl").open("w") as file:
        for item in responses:
            json.dump(item, file, ensure_ascii=False)
            file.write("\n")


def read_mme_items(mme_name):
    items = []
    mme_path = Path(MME_ROOT) / mme_name
    for image_path in sorted(mme_path.iterdir()):
        if image_path.suffix.lower() == ".txt":
            continue
        text_path = image_path.with_suffix(".txt")
        if not text_path.exists():
            continue
        with text_path.open() as file:
            for line in file:
                question, label = line.rstrip("\n").split("\t")
                items.append({"question": question, "image_path": image_path, "label": label})
    return limit_samples(items)


def run_mme(args):
    if args.model_name == "internvl":
        raise SystemExit("MME + internvl is not wired for AGLA in the original AGLA scripts.")
    model_pack = load_model(args.model_name, args)
    aux = load_agla_aux(model_pack[2])
    rows, responses = [], []
    for inst in tqdm.tqdm(read_mme_items(args.mme_name)):
        image_path = inst["image_path"]
        raw_image = Image.open(image_path).convert("RGB")
        prompt = {
            "llava": f"USER: <image>\n{inst['question']} Answer yes or no only.\nASSISTANT:",
            "blip2": f"USER: {inst['question']} Answer yes or no only.\nASSISTANT:",
        }[args.model_name]
        raw_response = generate_response(model_pack, args.model_name, raw_image, image_path, prompt, aux, args)
        response = yes_no_from_text(raw_response)
        ground_truth = "Yes" if inst["label"] == "Yes" else "No"
        rows.append(f"{image_path.name}\t{inst['question']}\t{ground_truth}\t{response}\n")
        responses.append(
            {
                "benchmark": "mme",
                "method": "AGLA",
                "model": args.model_name,
                "category": args.mme_name,
                "image": image_path.name,
                "question": inst["question"],
                "label": ground_truth.lower(),
                "response": response,
                "raw_response": raw_response,
            }
        )
        torch.cuda.empty_cache()

    output_dir = OUTPUT_ROOT / "mme_output" / args.model_name / "AGLA"
    response_dir = output_dir / "responses"
    response_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"{args.mme_name}.txt").open("w") as file:
        file.writelines(rows)
    with (response_dir / f"{args.mme_name}.jsonl").open("w") as file:
        for item in responses:
            json.dump(item, file, ensure_ascii=False)
            file.write("\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=["chair", "pope", "mme"], required=True)
    parser.add_argument("--model_name", choices=["llava", "blip2", "internvl"], required=True)
    parser.add_argument("--type_dataset", choices=["coco", "aokvqa", "gqa"], default="coco")
    parser.add_argument("--type_question", choices=["random", "popular", "adversarial"], default="popular")
    parser.add_argument("--mme_name", default="existence")
    parser.add_argument("--image_folder", default="/data/dtt/projects/SPAC/coco_dataset/image")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=178)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--cd_alpha", type=float, default=1.0)
    parser.add_argument("--cd_beta", type=float, default=0.1)
    parser.add_argument("--internvl_model_path", default="/data/dtt/pretrain_model_or_weight/InternVL2-2B")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.benchmark == "chair":
        run_chair(args)
    elif args.benchmark == "pope":
        run_pope(args)
    else:
        run_mme(args)


if __name__ == "__main__":
    main()
