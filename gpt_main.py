from pathlib import Path
from turtle import setup
from transformers.utils import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
import math
import datasets
from sklearn.model_selection import train_test_split
import torch.optim as optim
from tqdm import tqdm
import random
import evaluate
from data import *
import argparse
import os

logging.set_verbosity_error()

def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)

def get_lambda(step, all_steps, warmup_steps, type):
    if type == 'formula':
        return min(step ** (-0.5), step * (warmup_steps ** (-1.5)))
    elif type == 'linear':
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return max(
            0.0, float(all_steps - step) / float(max(1, all_steps - warmup_steps))
        )

def train(args):
    setup_seed(args.seed)
    print(args)
    raw_data = datasets.load_dataset('imdb', split ='train')
    tokenizer = AutoTokenizer.from_pretrained(args.initial_point)
    model = AutoModelForCausalLM.from_pretrained(args.initial_point)
    tokenizer.add_special_tokens({"pad_token": "<|endoftext|>"})
    tokenizer.padding_side = 'left'
    device = 'cuda' if torch.cuda.is_available else 'cpu'
    train_data, val_data = train_test_split(raw_data,test_size = args.test_size)
    trainset = IMDB_GPT(tokenizer,  train_data, args.max_len_dial, args.prompt, train = True)
    valset = IMDB_GPT(tokenizer,  val_data, args.max_len_dial, args.prompt, train = False)
    print(len(trainset),len(valset))
    optimizer = optim.Adam(model.parameters(), lr = args.lr)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_bz, shuffle=True, num_workers=0)
    valloader = torch.utils.data.DataLoader(valset, batch_size=args.val_bz, shuffle=False, num_workers=0)
    n_epoches = args.epoch
    accumulate_step = args.acc_step
    total_steps = math.ceil(n_epoches*len(trainset)/(args.train_bz * accumulate_step))
    wm_steps = int(total_steps*args.wm_ratio)
    model = model.to(device)
    
    print('***********do training***********')
    batch_counter = 0
    step_counter = 0
    log_step = args.log_step
    log_file = open('{}.txt'.format(args.logfile_name),'w')
    print_loss = True
    PATH = args.ptr_model_path
    if not os.path.exists(PATH):
        os.mkdir(PATH)
    train_loss = 0.0
    for epoch in range(n_epoches):
        for id,item in tqdm(enumerate(trainloader)):
            model.train()
            data,label = item
            data = data.to(device)
            label = label.to(device)
            outputs = model(data, labels=label)
            loss = outputs[0]
            loss = loss / accumulate_step
            batch_counter += 1
            if batch_counter % accumulate_step == (accumulate_step - 1):
                loss.backward()
                if args.schedule != 'constant':
                    lr = args.lr * get_lambda((step_counter/accumulate_step), total_steps, wm_steps, args.schedule)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                optimizer.step()
                optimizer.zero_grad()
                step_counter += 1
                print_loss = True
            train_loss = train_loss + loss.item()
            if (step_counter % log_step == log_step - 1) and print_loss:
                    print('[%d, %d] loss: %.3f' % (epoch + 1, step_counter + 1, train_loss / log_step))
                    log_file.write('[%d, %d] loss: %.3f' % (epoch + 1, step_counter + 1, train_loss / log_step)+'\n')
                    train_loss = 0.0
                    print_loss = False


        pred_list = []
        with torch.no_grad():
            for id,item in tqdm(enumerate(valloader)):
                model.eval()
                data,label= item
                data = data.to(device)
                label = label.to(device)
                pred_tokens = model.generate(data,max_length = args.max_len_dial + args.max_len_gen, pad_token_id=tokenizer.eos_token_id, \
                                             no_repeat_ngram_size=args.no_repeat_ngram_size, do_sample=args.do_sample, top_k=args.top_k,\
                                                  top_p=args.top_p, temperature = args.temp)
                pred_tokens = pred_tokens[:,data.shape[-1]:]
                pred_strings = tokenizer.batch_decode(pred_tokens,skip_special_tokens=True)
                pred_list = pred_list + pred_strings if len(pred_list) else pred_strings
        print('check outputs')
        print(pred_list[:10])
        log_file.write('\n'.join(pred_list))
        torch.save(model.state_dict(), PATH + '/model.pth')

def play(args):
    print(args)
    device = 'cuda' if torch.cuda.is_available else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(args.initial_point)
    model = AutoModelForCausalLM.from_pretrained(args.initial_point)
    model = model.to(device)
    log_file = open('gen_results.txt','w')
    tokenizer.add_special_tokens({"pad_token": "<|endoftext|>"})
    tokenizer.padding_side = 'left'
    model_path = args.ptr_model_path + '/model.pth'
    model.load_state_dict(torch.load(model_path))
    raw_data = datasets.load_dataset('imdb', split ='train')
    _, val_data = train_test_split(raw_data,test_size = args.test_size)
    dataset = IMDB_GPT(tokenizer,  val_data, args.max_len_dial, args.prompt)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.val_bz, shuffle=False, num_workers=0)
    pred_list = []
    with torch.no_grad():
        for id,item in tqdm(enumerate(dataloader)):
            model.eval()
            data,label= item
            data = data.to(device)
            label = label.to(device)
            pred_tokens = model.generate(data,max_length = args.max_len_dial + args.max_len_gen, pad_token_id=tokenizer.eos_token_id, \
                                            no_repeat_ngram_size=args.no_repeat_ngram_size, do_sample=args.do_sample, top_k=args.top_k,\
                                                top_p=args.top_p, temperature = args.temp)
            pred_tokens = pred_tokens[:,data.shape[-1]:]
            pred_strings = tokenizer.batch_decode(pred_tokens,skip_special_tokens=True)
            pred_list = pred_list + pred_strings if len(pred_list) else pred_strings
    log_file.write('\n'.join(pred_list))
    log_file.close()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', action='store_true', default=False)
    parser.add_argument('--play', action='store_true', default=False)
    parser.add_argument('--initial_point', type=str, default = 'gpt2-large')
    parser.add_argument('--ptr_model_path', type=str, default = './gpts')
    parser.add_argument('--schedule', type=str, default = 'linear')
    parser.add_argument('--wm_ratio', type =float, default = 0.1)
    parser.add_argument('--max_len_dial', type=int, default=512)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--test_size', type=float, default=0.1)
    parser.add_argument('--train_bz', type=int, default=1)
    parser.add_argument('--val_bz', type=int, default=10)
    parser.add_argument('--acc_step', type=int, default=32)
    parser.add_argument('--epoch', type=int, default=2)
    parser.add_argument('--log_step', type=int, default=200)
    parser.add_argument('--logfile_name', type=str, default = 'training_log')
    parser.add_argument('--max_len_gen', type=int, default=200)
    parser.add_argument('--no_repeat_ngram_size', type=int, default=3)
    parser.add_argument('--do_sample', action='store_true', default=True)
    parser.add_argument('--top_k', type = int, default = 100)
    parser.add_argument('--top_p', type =float, default = 0.7)
    parser.add_argument('--temp', type =float, default = 0.8)
    args = parser.parse_args()
    if args.prompt:
        args.do_sample = False
        args.temp = 0.0
    if args.play:
        play(args)
    else:
        train(args)
    
if __name__=='__main__':
    main()