"""
script for generating questions for KMIR dataset.
"""
import torch
import json
import os
import json
import pandas as pd

from transformers import AutoModelWithLMHead
from transformers import AutoTokenizer
from transformers import Trainer, TrainingArguments
from transformers import PreTrainedTokenizerBase
from transformers import Trainer

from datasets import Dataset
from typing import Optional
from dataclasses import dataclass
from tqdm import tqdm
from torch.utils.data import DataLoader

VERBOSE = True

def vprint(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)

def load_dataset(data_dir):
    data = []
    with open(data_dir, 'r') as ifp:
        for line in ifp.readlines():
            try:
                data.append(eval(line.strip().replace('null', 'None')))
            except:
                print(line)
                raise
    
    vprint('datum count: ', len(data))
    
    return data

class PreprocessFunction:

    def __init__(self, tokenizer, input_ids_max_length):
        self.tokenizer = tokenizer 
        self.input_ids_max_length = input_ids_max_length

    def __call__(self, example):
        # Tokenize the texts
        context = example['query'].replace('[MASK]', example['answer'][0])
        input_text = "answer: %s  context: %s </s>" % (example['answer'][0], context)
        result = self.tokenizer(
            input_text, 
            padding=True, 
            max_length=self.input_ids_max_length,
            truncation=True
        )
        return result


@dataclass
class CustomDataCollator:

    tokenizer: PreTrainedTokenizerBase
    padding: bool = True
    input_ids_max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    history_pad_token_id: int = 0

    def __call__(self, features):
        
        # get padding side
        padding_side = self.tokenizer.padding_side
        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.input_ids_max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt"
        )
        return features
    
class CustomTrainer(Trainer):
    
    def set_batch_size(self, ibatch_size):
        """
        Really don't want to deal with HF args.
        This thing has forced data parrallel by default
            gives specified bsize * num_gpu_avail
        """
        self.ibatch_size = ibatch_size
        return self
    
    # stub trainer for getting non-shuffled dataloaders
    # override get_train_dataloader to avoid shuffling
    def get_train_dataloader(self) -> DataLoader:
           
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        train_dataset = self._remove_unused_columns(train_dataset, description="training")

        return DataLoader(
            train_dataset,
            batch_size=self.ibatch_size,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

def has_tensor(obj) -> bool:
    """
    Given a possibly complex data structure,
    check if it has any torch.Tensors in it.
    Credit: AllenNLP
    """
    if isinstance(obj, torch.Tensor):
        return True
    elif isinstance(obj, dict):
        return any(has_tensor(value) for value in obj.values())
    elif isinstance(obj, (list, tuple)):
        return any(has_tensor(item) for item in obj)
    else:
        return False

def move_to_device(obj, cuda_device):
    """
    Given a structure (possibly) containing Tensors on the CPU,
    move all the Tensors to the specified GPU (or do nothing, if they should be on the CPU).
    Credit: AllenNLP
    """

    if cuda_device == torch.device("cpu") or not has_tensor(obj):
        return obj
    elif isinstance(obj, torch.Tensor):
        return obj.to(cuda_device)
    elif isinstance(obj, dict):
        return {key: move_to_device(value, cuda_device) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [move_to_device(item, cuda_device) for item in obj]
    elif isinstance(obj, tuple) and hasattr(obj, "_fields"):
        # This is the best way to detect a NamedTuple, it turns out.
        return obj.__class__(*(move_to_device(item, cuda_device) for item in obj))
    elif isinstance(obj, tuple):
        return tuple(move_to_device(item, cuda_device) for item in obj)
    else:
        return obj

def get_questions(model_input, model, tokenizer):
    model.eval()
    with torch.no_grad():
        output = model.generate(
            **move_to_device(model_input, torch.device('cuda')),
            max_length=64
        )

    generated = tokenizer.batch_decode(output)
    clean_sent = lambda s:s.replace('question: ', '').replace('<pad>', '').replace('<\s>', '').replace('</s>', '').strip()
    generated = [clean_sent(sent) for sent in generated]
    return generated

def main(args):
    
    vprint('loading model')
    tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")
    model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")

    
    vprint('preparing data')
    raw_data = load_dataset(args.input_file)
    data = Dataset.from_pandas(pd.DataFrame(raw_data))
    data = data.map(PreprocessFunction(tokenizer, 64))
    # a hack to get nicely formatted dataloader
    dataloader = CustomTrainer(
        model=model, 
        train_dataset=data,
        args = TrainingArguments(output_dir = '.unused'),
        data_collator=CustomDataCollator(
                tokenizer=tokenizer,
                input_ids_max_length=64
            )
    ).set_batch_size(args.batch_size).get_train_dataloader()
    
    vprint('generating questions')
    model = model.to(torch.device('cuda'))
    all_questions = []
    for datum in tqdm(dataloader, total=len(dataloader)):
        questions = get_questions(datum, model, tokenizer)
        all_questions += questions
        
    for generated_question, datum in zip(all_questions, raw_data):
        datum['question'] = generated_question 
        
    vprint('5 samples of prepared data')
    vprint(raw_data[:5])
    
    vprint('saving')
    with open(args.output_file, 'w') as ofp:
        for datum in raw_data:
            ofp.write(json.dumps(datum)+'\n')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Gen Questions')
    parser.add_argument('--input_file', type=str, default='na',
                        help='location of the data corpus')
    parser.add_argument('--output_file', type=str, default='na',
                        help='location of the data corpus')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='inference batch size')
    args = parser.parse_args()
    main(args)

