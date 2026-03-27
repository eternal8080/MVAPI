import os, json, hashlib
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

import math
import base64
from PIL import Image
import io

from datasets import load_dataset, concatenate_datasets

def get_data(dataset, **kwargs):
    if dataset == 'custom':
        return CustomDataset(**kwargs)
    elif dataset == 'mathvision':
        return MathVision(**kwargs)
    elif dataset == 'Geometry3k':
        return Geometry3K(**kwargs)
    elif dataset == 'GeoQA':
        return GeoQA(**kwargs)
    elif dataset == 'mathvista':
        return MathVista(**kwargs)
    elif dataset == 'VCBench':
        return VCBench(**kwargs)
    elif dataset == 'MMSI':
        return MMSI(**kwargs)
    elif dataset == 'ReMI':
        return ReMI(**kwargs)
    elif dataset == 'mmvet':
        return MMVET(**kwargs)
    elif dataset == 'textVQA':
        return textVQA(**kwargs)
    elif dataset == 'VisWiz':
        return VisWiz(**kwargs)
    elif dataset == 'MMMU':
        return MMMU(**kwargs)
    elif dataset == 'LLaVA_Wild':
        return LLaVA_Wild(**kwargs)
    elif dataset == 'MME':
        return MME(**kwargs)
    else:
        raise ValueError('Invalid dataset')

def Disposable_Dataloader(dataset, batch_size, drange, **kwargs):
    for i in range(drange[0], min(len(dataset),drange[1]), batch_size):
        st = i
        ed = i + batch_size
        keys, image_path_or_pil_images, questions, image_paths = [] , [] , [], []
        for j in range(st, ed):
            if j >= min(len(dataset),drange[1]):
                break
            key, image_path_or_pil_image, question, image_path = dataset.get_item(j)
            keys.append(key)
            image_path_or_pil_images.append(image_path_or_pil_image)
            questions.append(question)
            image_paths.append(image_path)
        yield keys, image_path_or_pil_images, questions, image_paths


MMMU_CAT_SHORT2LONG = {
    'acc': 'Accounting',
    'agri': 'Agriculture',
    'arch': 'Architecture_and_Engineering',
    'art': 'Art',
    'art_theory': 'Art_Theory',
    'bas_med': 'Basic_Medical_Science',
    'bio': 'Biology',
    'chem': 'Chemistry',
    'cli_med': 'Clinical_Medicine',
    'cs': 'Computer_Science',
    'design': 'Design',
    'diag_med': 'Diagnostics_and_Laboratory_Medicine',
    'econ': 'Economics',
    'elec': 'Electronics',
    'ep': 'Energy_and_Power',
    'fin': 'Finance',
    'geo': 'Geography',
    'his': 'History',
    'liter': 'Literature',
    'manage': 'Manage',
    'mark': 'Marketing',
    'mate': 'Materials',
    'math': 'Math',
    'mech': 'Mechanical_Engineering',
    'music': 'Music',
    'phar': 'Pharmacy',
    'phys': 'Physics',
    'psy': 'Psychology',
    'pub_health': 'Public_Health',
    'socio': 'Sociology'
}

MME_SUB_1 = ['scene', 'posters', 'celebrity']
MME_SUB_2 = ['count', 'code_reasoning', 'existence', 'OCR', 'color', 'commonsense_reasoning', 'numerical_calculation', 'position', 'text_translation']


def isliststr(s):
    return (s[0] == '[') and (s[-1] == ']')

def istype(s, type):
    if isinstance(s, type):
        return True
    try:
        return isinstance(eval(s), type)
    except Exception as _:
        return False

def decode_base64_to_image(base64_string, target_size=-1):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    if image.mode in ('RGBA', 'P'):
        image = image.convert('RGB')
    if target_size > 0:
        image.thumbnail((target_size, target_size))
    return image

class MMVET:
    def __init__(self, **kwargs):
        file_path = os.path.join("../../dataset/mm-vet","mm-vet.json")
        with open(file_path, 'r') as file:
            self.json = json.load(file)

        self.formated = kwargs.get('formated', False)

    def formated_question(self, question):
        if self.formated:
            question = question
        return question

    def get_item(self, index = 0):
        key = f"v1_{index}"
        item = self.json[key]
        image_path = os.path.join("../../dataset/mm-vet","images",item["imagename"])
        question = self.formated_question(item["question"])
        return key, image_path, question

    def __len__(self):
        return len(self.json)

class textVQA:
    def __init__(self, **kwargs):
        file_path = os.path.join("../../dataset/TextVQA/TextVQA_0.5.1_val.json")
        with open(file_path, 'r') as file:
            self.json = json.load(file)
            self.data = self.json['data']
        
        self.formated = kwargs.get('formated', False)

    def formated_question(self, question):
        if self.formated:
            question = question + " Answer the question using a single word or phrase."
        return question

    def get_item(self, index = 0):
        key = f"{index}"
        item = self.data[index]
        image_path = os.path.join("../../dataset/TextVQA/train_images",item["image_id"]+".jpg")
        question = self.formated_question(item["question"])
        return key, image_path, question

    def __len__(self):
        return len(self.data)
        
class VisWiz:
    def __init__(self, **kwargs):
        file_path = os.path.join("../../dataset/VisWiz/val.json")
        with open(file_path, 'r') as file:
            self.json = json.load(file)

        self.formated = kwargs.get('formated', False)

    def formated_question(self, question):
        if self.formated:
            question = question + " When the provided information is insufficient, respond with 'Unanswerable'. Answer the question using a single word or phrase."
        return question

    def get_item(self, index = 0):
        key = f"{index}"
        item = self.json[index]
        image_path = os.path.join("../../dataset/VisWiz/val",item["image"])
        question = item["question"]
        question = self.formated_question(question)
        return key, image_path, question

    def __len__(self):
        return len(self.json)

class MMMU:
    def __init__(self, **kwargs):
        data_path = "../../dataset/MMMU/MMMU"
        split = "validation"
        # run for each subject
        sub_dataset_list = []
        for subject in MMMU_CAT_SHORT2LONG.values():
            sub_dataset = load_dataset(data_path, subject, split=split)
            sub_dataset_list.append(sub_dataset)

        # merge all dataset
        self.dataset = concatenate_datasets(sub_dataset_list)

        self.formated = kwargs.get('formated', False)

    def get_question(self, question, options):
        return question

    def formated_question(self, sample):
        if self.formated:
            question = sample['question']
            options = eval(sample['options'])
            example = ""
            if sample['question_type'] == 'multiple-choice':
                start_chr = 'A'
                prediction_range = []
                index2ans = {}
                for option in options:
                    prediction_range.append(start_chr)
                    example += f"({start_chr}) {option}\n"
                    index2ans[start_chr] = option
                    start_chr = chr(ord(start_chr) + 1)
                empty_prompt_sample_structure = "{}\n\n{}\n\nAnswer with the option's letter from the given choices directly."
                empty_prompt = empty_prompt_sample_structure.format(question, example)
            else:
                empty_prompt_sample_structure = "{}\n\nAnswer the question using a single word or phrase."
                empty_prompt = empty_prompt_sample_structure.format(question)
            return empty_prompt
        else:
            return sample['question']

    def get_item(self, index = 0):
        key = f"{index}"
        item = self.dataset[index]
        question = self.formated_question(item)
        pil_image = item["image_1"]
        return key, pil_image, question

    def __len__(self):
        return len(self.dataset)

class LLaVA_Wild:
    def __init__(self, **kwargs):
        file_path = os.path.join("../../dataset/LLaVA_Wild/llava-bench-in-the-wild/questions.jsonl")
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        self.data = data

        self.formated = kwargs.get('formated', False)

    def get_item(self, index = 0):
        key = f"{index}"
        item = self.data[index]
        image_path = os.path.join("../../dataset/LLaVA_Wild/llava-bench-in-the-wild","images",item["image"])
        question = item["text"]
        question = self.formated_question(question)
        return key, image_path, question

    def formated_question(self, question):
        if self.formated:
            question = question
        return question

    def __len__(self):
        return len(self.data)

class MME:
    def __init__(self, **kwargs):
        base_path = "../../dataset/MME/MME_Benchmark_release_version"
        self.data = []
        for sub in MME_SUB_1:
            sub_folder = os.path.join(base_path, sub)
            question_folder = os.path.join(sub_folder, "questions_answers_YN")
            image_folder = os.path.join(sub_folder, "images")
            for question_file in sorted(os.listdir(question_folder)):
                question_file_path = os.path.join(question_folder, question_file)
                image_name = question_file.split('.')[0] + ".jpg"
                image_path = os.path.join(image_folder, image_name)
                if not os.path.exists(os.path.join(base_path, "images", image_name)):
                    continue
                with open(question_file_path, 'r') as file:
                    for l in file.readlines():
                        question, answer = l.strip().split('\t')
                        self.data.append([
                            image_path,
                            question,
                            answer,
                            sub
                        ])
        for sub in MME_SUB_2:
            sub_folder = os.path.join(base_path, sub)
            files = os.listdir(sub_folder)
            file_names = sorted(list(set([f.split('.')[0] for f in files])))
            for file_name in file_names:
                question_file = file_name + ".txt"
                image_name = file_name + ".jpg"
                question_file_path = os.path.join(sub_folder, question_file)
                image_path = os.path.join(sub_folder, image_name)
                if not os.path.exists(question_file_path) or not os.path.exists(image_path):
                    continue
                with open(question_file_path, 'r') as file:
                    for l in file.readlines():
                        question, answer = l.strip().split('\t')
                        self.data.append([
                            image_path,
                            question,
                            answer,
                            sub
                        ])

        self.formated = kwargs.get('formated', False)
            
    def formated_question(self, question):
        if self.formated:
            question = question + " Answer the question using a single word or phrase."
        return question

    def get_item(self, index = 0):
        key = f"{index}"
        item = self.data[index]
        image_path = item[0]
        question = item[1]
        question = self.formated_question(question)
        return key, image_path, question

    def __len__(self):
        return len(self.data)
        
class CustomDataset:
    def __init__(self, **kwargs):
        # 读取 JSON 数据
        file_path = kwargs.get('file_path', "your/path/to/multi-step_en.json")
        with open(file_path, 'r') as file:
            self.data = json.load(file)

        # 是否需要格式化问题
        self.formated = kwargs.get('formated', False)

    def formated_question(self, question):
        if self.formated:
            # 可在此进行问题格式化，若不需要格式化，可直接返回
            question = question
        return question

    def get_item(self, index=0):
        item = self.data[index]

        # 使用 translate_question 作为问题
        question = item["concise_question"] if "concise_question" in item else item["translate_question"]
        # 使用 input_image 作为图像路径
        image_paths = item["input_image"]
        # 加载图片
        images = [Image.open('your/path/to/multiimage_dataset/' + image_path) for image_path in image_paths]

        # 如果需要图像预处理
        if self.formated:
            images = [self.transform(image) for image in images]

        # 使用 problem_id 作为 key
        key = f"problem_{item['problem_id']}"

        image_paths = ["your/path/to/multiimage_dataset/" + path for path in item["input_image"]]
        return key, images, question, image_paths

    def __len__(self):
        return len(self.data)


from PIL import Image
import json
import os

class Geometry3K:
    def __init__(self, **kwargs):
        file_path = kwargs.get('file_path', "your/path/to/PGPS9K/Geometry3K/test.json")
        image_folder = kwargs.get('image_folder', "your/path/to/PGPS9K/Diagram")
        self.formated = kwargs.get('formated', False)

        with open(file_path, 'r') as f:
            self.data = json.load(f)

        self.image_folder = image_folder
        self.keys = list(self.data.keys())

    def formated_question(self, question):
        if self.formated:
            # 可在此插入格式化模版
            question = f"{question}"
        return question

    def get_item(self, index=0):
        prob_id = self.keys[index]
        item = self.data[prob_id]

        question = item["text"]
        parsing_sem = " ".join(seq + "." for seq in item.get("parsing_sem_seqs", []))
        parsing_stru = " ".join(seq + "." for seq in item.get("parsing_stru_seqs", []))

        full_question = f"{self.formated_question(question)}\nThis is the structural information of the image:\n{parsing_stru}\nThis is the semantic information:\n{parsing_sem}"

        image_file = os.path.join(self.image_folder, item["diagram"])
        image = Image.open(image_file).convert("RGB")

        key = prob_id  # e.g., "prob_1"

        return key, [image], question, image_file

    def __len__(self):
        return len(self.data)



# import json
# from PIL import Image

# class GeoQA:
#     def __init__(self, **kwargs):
#         file_path = kwargs.get('file_path', "your/path/to/geoqa/test_concise.jsonl")
#         with open(file_path, 'r') as file:
#             self.data = json.load(file)

#         # 是否需要格式化问题
#         self.formated = kwargs.get('formated', False)

#     def formated_question(self, question):
#         if self.formated:
#             # 可在此进行问题格式化，若不需要格式化，可直接返回
#             question = question
#         return question

#     def get_item(self, index=0):
#         item = self.data[str(index)]  # 使用字符串索引，因为你的数据以字符串形式存储

#         # 使用 subject 作为问题
#         question = item["subject"]
#         # 引入结构信息
#         parsing_sem_seqs = item["manual_program"]
#         # 将列表转换为字符串
#         parsing_sem_seqs = " ".join([seq + "." for seq in parsing_sem_seqs])
#         parsing_sem_seqs = question + " " + parsing_sem_seqs
#         # 图像路径基于问题id构造
#         image_id = item["id"]
#         # 加载图片
#         images = [Image.open(image_path)]

#         # 如果需要图像预处理
#         if self.formated:
#             images = [self.transform(image) for image in images]

#         # 使用 id 作为 key
#         key = f"problem_{item['id']}"

#         return key, images, question, image_path

#     def __len__(self):
#         return len(self.data)


import json
from PIL import Image

class MathVision:
    def __init__(self, **kwargs):
        # 数据路径
        file_path = kwargs.get('file_path', 'your/path/to/Mathvision/output-testmini.json')
        with open(file_path, 'r') as f:
            self.data = json.load(f)

        # 图像根路径
        self.image_root = kwargs.get('image_root', 'your/path/to/Mathvision')

        # 是否格式化问题
        self.formated = kwargs.get('formated', False)

    def formated_question(self, question, options):
        if self.formated and options and ''.join(options) != 'ABCDE':
            assert len(options) == 5, f"Invalid options for question: {question}"
            options_text = "\n" + "\n".join([
                f"({chr(65+i)}) {opt}" for i, opt in enumerate(options)
            ])
            question += options_text
        return question

    def get_item(self, index=0):
        item = self.data[index]  # MathVision 数据是 list 格式
        qa_id = item["id"]
        question = item["question"]
        options = item.get("options", [])
        answer = item.get("answer")

        # 拼接选项（如果需要格式化）
        question = self.formated_question(question, options)

        # 构造图像路径
        image_path = os.path.join(self.image_root, item["image"])
        images = [Image.open(image_path)]

        # 如果有图像预处理逻辑
        if self.formated:
            images = [self.transform(image) for image in images]  # 如果你有 self.transform 函数

        key = f"problem_{qa_id}"

        return key, images, question, image_path

    def __len__(self):
        return len(self.data)



class MathVista:
    def __init__(self, **kwargs):
        file_path = kwargs.get('file_path', "your/path/to/mathvista/testmini.json")
        with open(file_path, 'r', encoding='utf-8') as file:
            self.data = json.load(file)  # 注意 MathVista 是 JSON 而不是 JSONL

        self.image_folder = kwargs.get('image_folder', "your/path/to/mathvista")
        self.formated = kwargs.get('formated', False)

    def formated_question(self, question, choices):
        if choices:
            question = f"{question}"
        if self.formated:
            # 可添加 prompt 模板
            question = f"{question}"
        return question

    def get_item(self, index=0):
        key_list = list(self.data.keys())
        item = self.data[key_list[index]]

        question_id = item["pid"]
        question = item["question"]
        choices = item.get("choices")
        question = self.formated_question(question, choices)

        image_path = f"{self.image_folder}/{item['image']}"
        image = Image.open(image_path).convert("RGB")
        images = [image]

        if self.formated:
            images = [self.transform(img) for img in images]

        key = f"problem_{question_id}"

        return key, images, question, image_path

    def __len__(self):
        return len(self.data)
    

class VCBench:
    def __init__(self, **kwargs):
        # 读取 JSON 数据
        file_path = kwargs.get('file_path', "your/path/to/VCBench/VCBench_with_answer_concise_under9images.json")
        with open(file_path, 'r') as file:
            self.data = json.load(file)

        # 是否需要格式化问题
        self.formated = kwargs.get('formated', False)

    def formated_question(self, question):
        if self.formated:
            # 可在此进行问题格式化，若不需要格式化，可直接返回
            question = question
        return question

    def get_item(self, index=0):
        item = self.data[index]

        # 使用 translate_question 作为问题
        question = item["question_english"]
        # 使用 input_image 作为图像路径
        image_paths = item["images_question"]
        # 加载图片
        images = [Image.open('your/path/to/VCBench/' + image_path) for image_path in image_paths]

        # 如果需要图像预处理
        if self.formated:
            images = [self.transform(image) for image in images]

        # 使用 problem_id 作为 key
        key = f"problem_{item['id']}"

        image_paths = ["your/path/to/VCBench/" + path for path in item["images_question"]]
        return key, images, question, image_paths

    def __len__(self):
        return len(self.data)
    

class MMSI:
    def __init__(self, **kwargs):
        # 读取 JSON 数据
        file_path = kwargs.get('file_path', "your/path/to/MMSI/MMSI_consise.json")
        with open(file_path, 'r') as file:
            self.data = json.load(file)

        # 是否需要格式化问题
        self.formated = kwargs.get('formated', False)

    def formated_question(self, question):
        if self.formated:
            # 可在此进行问题格式化，若不需要格式化，可直接返回
            question = question
        return question

    def get_item(self, index=0):
        item = self.data[index]

        # 使用 translate_question 作为问题
        question = item["question"]
        # 使用 input_image 作为图像路径
        image_paths = item["images"]
        # 加载图片
        images = [Image.open(image_path) for image_path in image_paths]

        # 如果需要图像预处理
        if self.formated:
            images = [self.transform(image) for image in images]

        # 使用 problem_id 作为 key
        key = f"problem_{item['id']}"

        image_paths = [path for path in item["images"]]        # 加载图片
        return key, images, question, image_paths

    def __len__(self):
        return len(self.data)
    
    
class ReMI:
    def __init__(self, **kwargs):
        # 读取 JSON 数据
        file_path = kwargs.get('file_path', "your/path/to/ReMI/test.json")
        with open(file_path, 'r') as file:
            self.data = json.load(file)

        # 是否需要格式化问题
        self.formated = kwargs.get('formated', False)

    def formated_question(self, question):
        if self.formated:
            # 可在此进行问题格式化，若不需要格式化，可直接返回
            question = question
        return question

    def get_item(self, index=0):
        item = self.data[index]

        # 使用 translate_question 作为问题
        question = item["question"]
        # 使用 input_image 作为图像路径
        image_paths = item["image_paths"]
        # 加载图片
        images = [Image.open(image_path) for image_path in image_paths]

        # 如果需要图像预处理
        if self.formated:
            images = [self.transform(image) for image in images]

        # 使用 problem_id 作为 key
        key = f"problem_{item['qa_id']}"

        image_paths = [path for path in image_paths]        # 加载图片
        return key, images, question, image_paths

    def __len__(self):
        return len(self.data)
