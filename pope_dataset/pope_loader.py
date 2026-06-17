import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import skimage.io as io
import PIL.Image


class POPEDataSet(Dataset):
    def __init__(self, pope_path, data_path,processor,vqg_model,vqg_tokenizer,clip_model,preprocess_clip,device,prefix_length,generate_beam):
        self.pope_path = pope_path
        self.data_path = data_path
        self.processor = processor
        self.vqg_model = vqg_model
        self.vqg_tokenizer = vqg_tokenizer
        self.clip_model = clip_model
        self.preprocess_clip = preprocess_clip
        self.device = device
        self.prefix_length = prefix_length
        self.generate_beam = generate_beam
        #self.trans = trans

        image_list, query_list, label_list,question_list = [], [], [],[]
        
        for q in open(pope_path, 'r'):
            line = json.loads(q)
            image_list.append(line['image'])
            query_list.append(line['text'])
            label_list.append(line['label'])
            question_list.append(line['question_id'])

        for i in range(len(label_list)):
            if label_list[i] == 'no':
                label_list[i] = 0
            else:
                label_list[i] = 1
        assert len(image_list) == len(query_list)
        assert len(image_list) == len(label_list)

        self.image_list = image_list
        self.query_list = query_list
        self.label_list = label_list
        self.question_list = question_list
    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):
        image_path = os.path.join(self.data_path, self.image_list[index])
        raw_image = Image.open(image_path).convert("RGB")
        image = io.imread(image_path)
        pil_image = PIL.Image.fromarray(image)
        image = self.preprocess_clip(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            prefix = self.clip_model.encode_image(image).to(self.device, dtype=torch.float32)
            prefix_embed = self.vqg_model.clip_project(prefix).reshape(1, self.prefix_length, -1)
            generated_text_prefix = self.generate_beam(self.vqg_model, self.vqg_tokenizer, embed=prefix_embed)[0]
        #print(generated_text_prefix)
        #image = self.trans(raw_image)
        query = self.query_list[index]
        label = self.label_list[index]
        teacher_qu = "<image>\nUSER:{}\n, Note that the existing question:'{}'. ASSISTANT:".format(query,generated_text_prefix)
        student_qu = "<image>\nUSER:{}\n,ASSISTANT:".format(query)
        teacher_inputs = self.processor(images = raw_image,text = teacher_qu,return_tensors='pt')
        student_inputs = self.processor(images = raw_image,text = student_qu,return_tensors='pt')
        #label_dict = {'label':label}
        return {"teacher_inputs":teacher_inputs,'label':label,'student_inputs':student_inputs,'question_id':self.question_list[index]}
    

