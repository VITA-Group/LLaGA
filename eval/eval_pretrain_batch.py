import sys
sys.path.append("./")
sys.path.append("./utils")
import argparse
import torch
import os
import json
from tqdm import tqdm, trange
import shortuuid

from utils.constants import GRAPH_TOKEN_INDEX, DEFAULT_GRAPH_TOKEN, DEFAULT_GRAPH_PAD_ID, DEFAULT_GRAPH_START_TOKEN, DEFAULT_GRAPH_END_TOKEN
from utils.conversation import conv_templates, SeparatorStyle
from model.builder import load_pretrained_model
from utils.utils import disable_torch_init, tokenizer_graph_token, get_model_name_from_path

import math

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


# def get_chunk(lst, n, k):
#     chunks = split_list(lst, n)
#     return chunks[k]

def load_pretrain_embedding_graph(data_dir, pretrained_embedding_type):
    if pretrained_embedding_type == "simteg":
        simteg_sbert = torch.load(os.path.join(data_dir, "simteg_sbert_x.pt"))
        simteg_roberta = torch.load(os.path.join(data_dir, "simteg_roberta_x.pt"))
        simteg_e5 = torch.load(os.path.join(data_dir, "simteg_e5_x.pt"))
        pretrained_emb = torch.concat([simteg_sbert, simteg_roberta, simteg_e5], dim=-1)
    else:
        pretrained_emb = torch.load(os.path.join(data_dir, f"{pretrained_embedding_type}_x.pt"))
    return pretrained_emb

def load_pretrain_embedding_hop(data_dir, pretrained_embedding_type, hop, mask):
    if pretrained_embedding_type == "simteg":
        simteg_sbert=[torch.load(os.path.join(data_dir, f"simteg_sbert_x.pt"))[mask]] + [torch.load(os.path.join(data_dir, f"simteg_sbert_{i}hop_x.pt"))[mask] for i in range(1, hop + 1)]
        simteg_roberta = [torch.load(os.path.join(data_dir, f"simteg_roberta_x.pt"))[mask]] + [torch.load(os.path.join(data_dir, f"simteg_roberta_{i}hop_x.pt"))[mask] for i in range(1, hop + 1)]
        simteg_e5 = [torch.load(os.path.join(data_dir, f"simteg_e5_x.pt"))[mask]] + [torch.load(os.path.join(data_dir, f"simteg_e5_{i}hop_x.pt"))[mask] for i in range(1, hop + 1)]
        pretrained_embs = [torch.cat([simteg_sbert[i], simteg_roberta[i], simteg_e5[i]], dim=-1) for i in range(hop + 1)]
    else:
        pretrained_embs = [torch.load(os.path.join(data_dir, f"{pretrained_embedding_type}_x.pt"))[mask]]+  [torch.load(os.path.join(data_dir, f"{pretrained_embedding_type}_{i}hop_x.pt"))[mask] for i in range(1, hop+1)]

    return pretrained_embs

def eval_model(args):
    # Model
    disable_torch_init()

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    print(f"Loaded from {model_path}. Model Base: {args.model_base}")
    tokenizer, model, context_len = load_pretrained_model(model_path, args.model_base, model_name,
                                                          cache_dir=args.cache_dir)
    model = model.to(torch.float16).cuda()
    # data_dir=os.path.expanduser(args.data_dir)
    if args.dataset == "arxiv":
        data_dir = "dataset/ogbn-arxiv"
    elif args.dataset == "products":
        data_dir = "dataset/ogbn-products"
    elif args.dataset == "pubmed":
        data_dir = "dataset/pubmed"
    elif args.dataset == "cora":
        data_dir = "dataset/cora"
    else:
        print(f"{args.dataset} not exists")
        raise ValueError
    if args.test_path is not None:
        prompt_file = args.test_path
    elif args.task in  ["nc", "nd", "nda", "nctext"]:
        if args.template == "HO":
            prompt_file = os.path.join(data_dir, f"sampled_2_10_test.jsonl")
        else:
            prompt_file = os.path.join(data_dir, f"sampled_{args.use_hop}_{args.sample_neighbor_size}_test.jsonl")
    elif args.task in ["lp"]:
        if args.template == "HO":
            prompt_file = os.path.join(data_dir, f"edge_sampled_2_10_only_test.jsonl")
        else:
            prompt_file = os.path.join(data_dir, f"edge_sampled_{args.use_hop}_{args.sample_neighbor_size}_only_test.jsonl")
    else:
        raise ValueError
    data_path = os.path.join(data_dir, f"processed_data.pt")
    data = torch.load(data_path)
    print(f"Load from {prompt_file}\n")
    lines = open(prompt_file, "r").readlines()

    if args.start >= 0:
        if args.end < 0:
            args.end = len(lines)
        lines = lines[args.start:args.end]
    elif args.end > 0:
        lines = lines[:args.end]

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    if "tmp" not in args.answers_file and os.path.exists(answers_file):
        line_number = len(open(answers_file, 'r').readlines())
        print(f"{args.answers_file} already exists! it has {line_number} lines!!")
        if line_number >= len(lines):
            return
        lines = lines[line_number:]
        ans_file = open(answers_file, "a")
    else:
        ans_file = open(answers_file, "w")

    questions = [json.loads(q) for q in lines]

    index = None
    if args.template == "ND":
        pretrained_emb = load_pretrain_embedding_graph(data_dir, args.pretrained_embedding_type)
        structure_emb = torch.load(
            f"dataset/laplacian_{args.use_hop}_{args.sample_neighbor_size}.pt")

    elif args.template == "HO":
        n = data.num_nodes
        mask = torch.full([n], fill_value=False, dtype=torch.bool)
        for q in questions:
            idx = q["id"]
            if "lp" in  args.task:
                assert len(idx) == 2
                mask[idx[0]] = True
                mask[idx[1]] = True
            elif args.task  in ["nc", "nd"]:
                assert isinstance(idx, int)
                mask[idx] = True
        pretrained_emb = load_pretrain_embedding_hop(data_dir, args.pretrained_embedding_type, args.use_hop, mask)
        index = torch.full([n], fill_value=n + 1, dtype=torch.long)
        test_index = torch.arange(mask.sum())
        index[mask] = test_index
        structure_emb = None
    else:
        raise ValueError

    if args.task in ["nd", "nda"]:
        qs = f"Please briefly describe the center node of {DEFAULT_GRAPH_TOKEN}."
    elif args.task == "nc":
        if args.dataset == "products":
            qs = f"Given a node-centered graph: {DEFAULT_GRAPH_TOKEN}, where nodes represent products sold in Amazon, and edges between products indicate they are purchased together. We need to classify the center node into 47 classes: Home & Kitchen, Health & Personal Care, Beauty, Sports & Outdoors, Books, Patio, Lawn & Garden, Toys & Games, CDs & Vinyl, Cell Phones & Accessories, Grocery & Gourmet Food, Arts, Crafts & Sewing, Clothing, Shoes & Jewelry, Electronics, Movies & TV, Software, Video Games, Automotive, Pet Supplies, Office Products, Industrial & Scientific, Musical Instruments, Tools & Home Improvement, Magazine Subscriptions, Baby Products, label 25, Appliances, Kitchen & Dining, Collectibles & Fine Art, All Beauty, Luxury Beauty, Amazon Fashion, Computers, All Electronics, Purchase Circles, MP3 Players & Accessories, Gift Cards, Office & School Supplies, Home Improvement, Camera & Photo, GPS & Navigation, Digital Music, Car Electronics, Baby, Kindle Store, Buy a Kindle, Furniture & D&#233;cor, #508510, please tell me which class the center node belongs to?"
        else:
            qs = questions[0]["conversations"][0]['value']
    elif args.task == "lp":
        qs = f"Given two node-centered subgraphs: {DEFAULT_GRAPH_TOKEN} and {DEFAULT_GRAPH_TOKEN}, we need to predict whether these two nodes connect with each other. Please tell me whether two center nodes in the subgraphs should connect to each other."
    else:
        print(f"NOT SUPPORT {args.task}!!!")
        raise ValueError
    cur_prompt = qs

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_graph_token(prompt, tokenizer, GRAPH_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    
    for i in trange(0, len(questions), args.batch_size):
        batch = questions[i:i+args.batch_size]
        gts = []
        idxs = []
        graph_embs = []
        graphs=[]
        for line in batch:
            idx = line["id"]
            if not isinstance(line['graph'][0], list):
                line['graph'] = [line['graph']]
            if args.template == "ND":
                graph = torch.LongTensor(line['graph'])
                mask = graph != DEFAULT_GRAPH_PAD_ID
                masked_graph_emb = pretrained_emb[graph[mask]]
                s, n, d = graph.shape[0], graph.shape[1], masked_graph_emb.shape[1]
                graph_emb = torch.zeros((s, n, d))
                graph_emb[mask] = masked_graph_emb
                if structure_emb is not None:
                    graph_emb = torch.cat([graph_emb, structure_emb.unsqueeze(0).expand(s, -1, -1)], dim=-1)
            elif args.template == "HO":
                for g in range(len(line['graph'])):
                    center_id = line['graph'][g][0]
                    line['graph'][g] = [center_id]*(args.use_hop+1)
                graph = torch.LongTensor(line['graph'])
                center_id = graph[:, 0]
                graph_emb = torch.stack([emb[index[center_id]] for emb in pretrained_emb], dim=1)
            else:
                raise ValueError
            idxs.append(idx)
            graphs.append(graph)
            graph_embs.append(graph_emb)
            gts.append(line["conversations"][1]['value'])
        graph_embs = torch.cat(graph_embs, dim=0)
        graphs = torch.cat(graphs, dim=0)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

        try:
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids.expand(len(batch), -1),
                    graph_emb=graph_embs.half().cuda(),
                    graph=graphs.cuda(),
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    # no_repeat_ngram_size=3,
                    max_new_tokens=1024,
                    use_cache=True)

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
            for i in range(len(outputs)):
                outputs[i] = outputs[i].strip()
                if outputs[i].endswith(stop_str):
                    outputs[i] = outputs[i][:-len(stop_str)]
                outputs[i] = outputs[i].strip()
        except Exception as e:
            print(f"!!!!!!Error!!!!! {e}")
            outputs=[""] * len(batch)
        for i in range(len(batch)):
            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({"question_id": idxs[i],
                                       "prompt": cur_prompt,
                                       "text": outputs[i],
                                       "gt":gts[i],
                                       "answer_id": ans_id}) + "\n")
            ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model_base", type=str, default=None)
    # parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--pretrained_embedding_type", type=str, default="sbert")
    parser.add_argument("--use_hop", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--sample_neighbor_size", type=int, default=5)
    parser.add_argument("--answers_file", type=str, default="answer.jsonl")
    parser.add_argument("--conv_mode", type=str, default="v1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--start", type=int, default=-1)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--test_path", type=str, default=None)
    parser.add_argument("--mm_use_graph_start_end",default=False, action="store_true")
    parser.add_argument("--task", type=str, default="nc")
    parser.add_argument("--dataset", type=str, default="arxiv")
    parser.add_argument("--cache_dir", type=str, default="../../checkpoint")
    parser.add_argument("--template", type=str, default="ND")
    args = parser.parse_args()

    eval_model(args)
