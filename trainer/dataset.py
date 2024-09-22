import torch
from torch.utils.data import Dataset, Subset
import os
import hashlib
from deepspeed.accelerator import get_accelerator
import random
import numpy as np


# Utility to load QA data from .q and .a files
def load_qa_data(data_dir):
    qa_pairs = []
    question_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.py')])
    for question_file in question_files:
        question_path = os.path.join(data_dir, question_file)
        answer_path = question_path.replace('.py', '.cu')
        
        if os.path.exists(answer_path):
            with open(question_path, 'r', encoding='utf-8') as qf, open(answer_path, 'r', encoding='utf-8') as af:
                question = qf.read().strip()
                answer = af.read().strip()
                qa_pairs.append((question, answer))
    return qa_pairs


class SpecificDataset:

    def __init__(self, output_path, seed, local_rank, dataset_name):
        self.output_path = output_path
        self.seed = seed
        self.local_rank = local_rank
        self.dataset_name = dataset_name
        self.dataset_name_clean = os.path.basename(dataset_name).replace('/', '_')
        
        # Load QA data instead of using HuggingFace datasets
        if os.path.exists(dataset_name):
            self.qa_pairs = load_qa_data(dataset_name)
        else:
            raise ValueError(f"Dataset not found at {dataset_name}")
    
    def get_train_data(self):
        # 80% for training
        return self.qa_pairs[:int(0.8 * len(self.qa_pairs))]

    def get_eval_data(self):
        # 20% for evaluation
        return self.qa_pairs[int(0.8 * len(self.qa_pairs)):]

    def get_question(self, sample):
        return sample[0]

    def get_answer(self, sample):
        return sample[1]

    def get_prompt_and_answer(self, sample):
        question, answer = sample
        return f"Q: {question}\nA: {answer}"


def get_shuffle_idx(seed, size):
    np_rng = np.random.RandomState(seed=seed)
    dtype_ = np.uint32
    if size >= (np.iinfo(np.uint32).max - 1):
        dtype_ = np.int64
    shuffle_idx = np.arange(start=0, stop=size, step=1, dtype=dtype_)
    np_rng.shuffle(shuffle_idx)
    return shuffle_idx


def get_raw_dataset_split_index(local_rank,
                                output_path,
                                dataset_name,
                                seed,
                                split_name,
                                data_split,
                                split_index,
                                data_size,
                                rebuild=False):
    index_file_name = f"{output_path}/{dataset_name}_seed{seed}_{split_name}_{data_split}_{split_index}.npy"
    if rebuild or (not os.path.isfile(index_file_name)):
        splits = [float(s) for s in data_split.split(',')]
        splits_sum = sum(splits)
        splits = [split / splits_sum for split in splits]
        splits_index = [0]
        for index, split in enumerate(splits):
            splits_index.append(splits_index[index] +
                                int(round(split * float(data_size))))
        diff = splits_index[-1] - data_size
        for index in range(1, len(splits_index)):
            splits_index[index] -= diff
        assert splits_index[-1] == data_size

        shuffle_idx = get_shuffle_idx(seed, data_size)
        for split_i in range(len(splits)):
            shuffle_idx_split_file_name = f"{output_path}/{dataset_name}_seed{seed}_{split_name}_{data_split}_{split_i}.npy"
            shuffle_idx_split = shuffle_idx[splits_index[split_i]:splits_index[split_i + 1]]
            np.save(shuffle_idx_split_file_name, shuffle_idx_split, allow_pickle=True)
    index = np.load(index_file_name, allow_pickle=True)
    return index.tolist()


class PromptDataset(Dataset):

    def __init__(self, prompt_dataset, chosen_dataset, pad_token_id, train_phase) -> None:
        super().__init__()
        self.prompt_dataset = prompt_dataset
        self.chosen_dataset = chosen_dataset
        self.pad_token_id = pad_token_id
        self.train_phase = train_phase

    def __len__(self):
        return len(self.chosen_dataset)

    def __getitem__(self, idx):
        return {
            "input_ids": self.chosen_dataset[idx]["input_ids"],
            "attention_mask": self.chosen_dataset[idx]["attention_mask"],
            "labels": self.chosen_dataset[idx]["input_ids"]
        }


def create_dataset_split(current_dataset, raw_dataset, train_phase, tokenizer,
                         end_of_conversation_token, max_seq_len):
    prompt_dataset = []
    chosen_dataset = []
  
    for i, sample in enumerate(current_dataset):
        # Tokenize the question-answer format
        qa_sentence = raw_dataset.get_prompt_and_answer(sample)
        qa_sentence += end_of_conversation_token
        qa_token = tokenizer(qa_sentence,
                             max_length=max_seq_len,
                             padding="max_length",
                             truncation=True,
                             return_tensors="pt")
        qa_token["input_ids"] = qa_token["input_ids"].squeeze(0)
        qa_token["attention_mask"] = qa_token["attention_mask"].squeeze(0)
        chosen_dataset.append(qa_token)
    
    print(f'Creating dataset {raw_dataset.dataset_name_clean} for {train_phase=} size={len(chosen_dataset)}')

    return PromptDataset(prompt_dataset, chosen_dataset, tokenizer.pad_token_id, train_phase)


def create_dataset(local_rank, dataset_name, data_split, output_path, train_phase,
                   seed, tokenizer, end_of_conversation_token, max_seq_len, rebuild):
    raw_dataset = SpecificDataset(output_path, seed, local_rank, dataset_name)

    train_dataset = raw_dataset.get_train_data()
    train_index = get_raw_dataset_split_index(local_rank, output_path,
                                              raw_dataset.dataset_name_clean,
                                              seed, "train", data_split,
                                              train_phase - 1,
                                              len(train_dataset), rebuild)
    train_dataset = Subset(train_dataset, train_index)
    train_dataset = create_dataset_split(train_dataset, raw_dataset,
                                         train_phase, tokenizer,
                                         end_of_conversation_token,
                                         max_seq_len)

    eval_dataset = raw_dataset.get_eval_data()
    eval_index = get_raw_dataset_split_index(local_rank, output_path,
                                             raw_dataset.dataset_name_clean,
                                             seed, "eval", data_split,
                                             train_phase - 1,
                                             len(eval_dataset), rebuild)
    eval_dataset = Subset(eval_dataset, eval_index)
    eval_dataset = create_dataset_split(eval_dataset, raw_dataset, train_phase,
                                        tokenizer, end_of_conversation_token,
                                        max_seq_len)
    return train_dataset, eval_dataset


def create_prompt_dataset(local_rank,
                          data_path,
                          data_split,
                          output_path,
                          train_phase,
                          seed,
                          tokenizer,
                          max_seq_len,
                          end_of_conversation_token="<|endoftext|>",
                          sft_only_data_path=[],
                          rebuild=False):
    """
    Creates the prompt dataset in QA format.
    """
    print(f"Called Prompt Gen for QA format")
    os.makedirs(output_path, exist_ok=True)
    fname = "_".join(data_path)
    sft_cache_key = "_".join(sft_only_data_path)
    tokenizer_name = tokenizer.init_kwargs["name_or_path"].replace("/", "_")
    fname = f"{fname}_split{data_split}_phase{train_phase}_seed{seed}_tokenizer{tokenizer_name}_seqlen{max_seq_len}_sft{sft_cache_key}"
    fname = "_".join(fname.split("/"))
    fname = hashlib.sha256(fname.encode()).hexdigest()
    train_fname = f"{output_path}/traindata_{fname}.pt"
    eval_fname = f"{output_path}/evaldata_{fname}.pt"

    cache_found = os.path.isfile(train_fname) and os.path.isfile(eval_fname)
    buf_create_cache = torch.ByteTensor([not cache_found]).to(
        get_accelerator().current_device_name())
    torch.distributed.all_reduce(buf_create_cache)

    if local_rank <= 0 and rebuild:
        print(f'Creating prompt dataset {data_path}, {rebuild=}')
   
        train_dataset, eval_dataset = create_dataset(
            local_rank,
            data_path[0],
            data_split,
            output_path,
            train_phase,
            seed,
            tokenizer,
            end_of_conversation_token,
            max_seq_len,
            rebuild=rebuild)
     
        torch.save(train_dataset, train_fname)
        torch.save(eval_dataset, eval_fname)
    else:
        if local_rank <= 0:
            print("Not rebuilding!!")
            
    torch.distributed.barrier()
    return torch.load(train_fname), torch.load(eval_fname)
