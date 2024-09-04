# <center> LLaGA: Large Language and Graph Assistant</center>



<p align="center">
<a href="https://arxiv.org/abs/2402.08170"><img src="https://img.shields.io/badge/Arxiv-2402.08170-B31B1B.svg"></a>
<a href="https://github.com/VITA-Group/LLaGA"><img src="https://img.shields.io/github/stars/VITA-Group/LLaGA"></a>
</p>

The official implementation of work "LLaGA: Large Language and Graph Assistant".

<img src="doc/main.png" width="90%">

## News
**2024.5.1**: LLaGA has been accepted by **ICML 2024**! 🎉 🎉 🎉



## Step 1: Environment Preparation 

```shell
# create a new environment
conda create -n llaga python=3.10
conda activate llaga

# install pytorch. Modify the command to align with your own CUDA version.
pip3 install torch  --index-url https://download.pytorch.org/whl/cu118

# install related libraries
pip install -r requirements.txt

# install flash-attn
pip install flash-attn --no-build-isolation

# install pyg
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

```
## Step 2:  Data Preparation
Download our datasets from [Box](https://utexas.box.com/s/i7y03rzm40xt9bjbaj0dfdgxeyjx77gb) (updated link prediction data on 4/11/2024). And move the processed data to `./dataset`
```
.
├── README.md
├── __init__.py
├── doc
├── dataset
│   ├── ogbn-arxiv
│   │   ├── sampled_2_10_test.jsonl
│   │   ├── sampled_2_10_train.jsonl
│   │   ├── edge_sampled_2_10_only_test.jsonl
│   │   ├── edge_sampled_2_10_only_train.jsonl
│   │   ├── processed_data.pt
│   │   ├── roberta_x.pt
│   │   ├── sbert_x.pt
│   │   ├── simteg_*_x.pt
│   ├── ogbn-products
│   │   ├── sampled_2_10_test.jsonl
│   │   ├── sampled_2_10_train.jsonl
│   │   ├── edge_sampled_2_10_only_test.jsonl
│   │   ├── edge_sampled_2_10_only_train.jsonl
│   │   ├── processed_data.pt
│   │   ├── roberta_x.pt
│   │   ├── sbert_x.pt
│   │   ├── simteg_*_x.pt
│   └── pubmed
│   │   ├── sampled_2_10_test.jsonl
│   │   ├── sampled_2_10_train.jsonl
│   │   ├── edge_sampled_2_10_only_test.jsonl
│   │   ├── edge_sampled_2_10_only_train.jsonl
│   │   ├── processed_data.pt
│   │   ├── roberta_x.pt
│   │   ├── sbert_x.pt
│   │   ├── simteg_*_x.pt
│   ├── cora
│   │   ├── sampled_2_10_test.jsonl
│   │   ├── sampled_2_10_train.jsonl
│   │   ├── edge_sampled_2_10_only_test.jsonl
│   │   ├── edge_sampled_2_10_only_train.jsonl
│   │   ├── processed_data.pt
│   │   ├── roberta_x.pt
│   │   ├── sbert_x.pt
│   │   ├── simteg_*_x.pt
│   ├── laplacian_2_10.pt
│   ├── laplacian_2_20.pt
│   ├── laplacian_2_5.pt
├── eval
├── model
├── requirements.txt
├── scripts
├── train
├── utils
```
## Step 3: Training
To execute the training process, you can run either `./scripts/train.sh` or `./scripts/train_deepspeed.sh`. The usage instructions are as follows:
```shell
#Auguments
# $1 = model type, e.g. vicuna, vicuna_4hop
# $2 = training task Use nc/lp/nd for single task. For multiple tasks combined, connect these abbreviations with '-', e.g. nc-lp
# $3 = dataset arxiv/products/pubmed/cora  For multiple datasets combined, connect these abbreviations with '-', and use '.n' to repeat multiple times e.g. arxiv-products-pubmed-cora.3 means using arxiv+products+pubmed+cora, and repeat cora for 3 times
# $4 = batch size  default: 16
# $5 = embedding e.g. simteg, sbert, roberta


#  training on single GPU
CUDA_VISIBLE_DEVICES=0 ./scripts/train.sh vicuna nc-lp  arxiv-products-pubmed-cora.3 16 simteg

#  training on multiple GPU
./scripts/train_deepspeed.sh vicuna nc-lp  arxiv-products-pubmed-cora.3 4 simteg
```

We also uploaded four general projectors to huggingface.

|        Setting         | Template |                                                                                   Repo                                                                                    |
|:----------------------:|:--------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|     General Model      |    HO    |    [Runjin/llaga-vicuna-7b-simteg-HO-general_model-2-layer-mlp-projector](https://huggingface.co/Runjin/llaga-vicuna-7b-simteg-HO-general_model-2-layer-mlp-projector)    |
|     General Model      |    ND    |    [Runjin/llaga-vicuna-7b-simteg-ND-general_model-2-layer-mlp-projector](https://huggingface.co/Runjin/llaga-vicuna-7b-simteg-ND-general_model-2-layer-mlp-projector)    |
| Classification Expert  |    HO    | [Runjin/llaga-vicuna-7b-simteg-HO-classification_expert-linear-projector](https://huggingface.co/Runjin/llaga-vicuna-7b-simteg-HO-classification_expert-linear-projector) |
| Classification Expert  |    ND    | [Runjin/llaga-vicuna-7b-simteg-ND-classification_expert-linear-projector](https://huggingface.co/Runjin/llaga-vicuna-7b-simteg-ND-classification_expert-linear-projector) |

## Step 4: Evaluation
You can evaluate LLaGA with the command:

```shell
model_path="/path/to/projector" # local path or huggingface repo
model_base="lmsys/vicuna-7b-v1.5-16k" #meta-llama/Llama-2-7b-hf
mode="v1" # use 'llaga_llama_2' for llama and "v1" for others
dataset="arxiv" #test dataset
task="nc" #test task
emb="simteg"
use_hop=2 # 2 for ND and 4 for HO
sample_size=10
template="ND"
output_path="/path/to/output"

python eval/eval_pretrain.py \
--model_path ${model_path} \
--model_base ${model_base} \
--conv_mode  ${mode} \
--dataset ${dataset} \
--pretrained_embedding_type ${emb} \
--use_hop ${use_hop} \
--sample_neighbor_size ${sample_size} \
--answers_file ${output_path} \
--task ${task} \
--cache_dir ../../checkpoint \
--template ${template}
```

To evaluate your predicted results, please run:
```shell
python eval/eval_res.py --dataset ${dataset} --task ${task}  --res_path ${output_path}
```

## Citation
```
@InProceedings{pmlr-v235-chen24bh,
  title = 	 {{LL}a{GA}: Large Language and Graph Assistant},
  author =       {Chen, Runjin and Zhao, Tong and Jaiswal, Ajay Kumar and Shah, Neil and Wang, Zhangyang},
  booktitle = 	 {Proceedings of the 41st International Conference on Machine Learning},
  pages = 	 {7809--7823},
  year = 	 {2024},
  editor = 	 {Salakhutdinov, Ruslan and Kolter, Zico and Heller, Katherine and Weller, Adrian and Oliver, Nuria and Scarlett, Jonathan and Berkenkamp, Felix},
  volume = 	 {235},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {21--27 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://raw.githubusercontent.com/mlresearch/v235/main/assets/chen24bh/chen24bh.pdf},
  url = 	 {https://proceedings.mlr.press/v235/chen24bh.html},
  }
```