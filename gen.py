import torch
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

from run_generation import sample_sequence
from yt_encoder import YTEncoder
from transformers import GPT2LMHeadModel
import threading
import argparse
import regex as re

from os import environ
device = environ.get('DEVICE', 'cuda:0')

flavor_id = device + environ.get('INSTANCE', ':0')
import logging

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

model_path = '/content/gpt2/s_checkpoint-1900000/checkpoint-1903981'

tokenizer = YTEncoder.from_pretrained(model_path)

model = GPT2LMHeadModel.from_pretrained(model_path)
model.to(device)
model.eval()

# poetry_model = GPT2LMHeadModel.from_pretrained(model_path)
# poetry_model.to(device)
# poetry_model.eval()

# from apex import amp
# [model, poetry_model] = amp.initialize([model, poetry_model], opt_level='O2')

def get_sample(model, prompt, length:int, num_samples:int, allow_linebreak:bool):
    logger.info(prompt)
   
    filter_n = tokenizer.encode('\n')[-1:]
    filter_single = [1] + tokenizer.encode('[')[-1:] + tokenizer.encode('(')[-1:]
    filter_single += [] if allow_linebreak else filter_n

    context_tokens = tokenizer.encode(prompt)
    out = sample_sequence(
        model=model,
        context=context_tokens,
        length=length,
        temperature=1,
        top_k=0,
        top_p=0.9,
        device=device,
        filter_single=filter_single,
        filter_double=filter_n,
        num_samples=num_samples,
    ).to('cpu')

    prompt = tokenizer.decode(context_tokens)
    len_prompt = len(prompt)
   
    replies = [out[item, :].tolist() for item in range(len(out))]
    text = [tokenizer.decode(item)[len_prompt:] for item in replies]
    reg_text = [re.match(r'[\w\W]*[\.!?]\n', item) for item in text]
    reg_text2 = [re.match(r'[\w\W]*[\.!?]', item) for item in text]
    result = [reg_item[0] if reg_item else reg_item2[0] if reg_item2 else item for reg_item, reg_item2, item in zip(reg_text, reg_text2, text)]
    logger.info(result)
    return result
parser = argparse.ArgumentParser()
parser.add_argument("--prompt", default=None, type=str, required=True,
                    help="Prompt")
parser.add_argument("--length", default=100, type=int, required=True,
                    help="length")
parser.add_argument("--count", type=int, default=3)
parser.add_argument("--allow_breakline", type=bool, default=False)
args = parser.parse_args()

sample = get_sample(model, args.prompt, args.length, args.count, args.allow_breakline)

def split_string(str, limit, sep=" "):
    words = str.split()
    if max(map(len, words)) > limit:
        raise ValueError("limit is too small")
    res, part, others = [], words[0], words[1:]
    for word in others:
        if len(sep)+len(word) > limit-len(part):
            res.append(part)
            part = word
        else:
            part += sep+word
    if part:
        res.append(part)
    return res

for s in sample:
    print('\n'.join(split_string(str=s, limit=100)))
