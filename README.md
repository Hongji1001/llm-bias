# Bias Benchmark

## Features
- Classification Bias Benchmark without mitigation
- Classification Bias Benchmark with mitigation
- Generation Bias Benchmark without mitigation
- Generation Bias Benchmark with mitigation
- Generate Generation Bias Dataset automatically

## Installation
We assume conda / miniconda has been download. Install the dependencies through conda:
```bash
# clone our repo
git clone https://github.com/Hongji1001/llm-bias.git
cd llm-bias
# remove old env if necessary
conda deactivate; conda env remove --name bias-benchmark
conda env create -f env.yaml --name bias-benchmark
```

## Classification Bias Benchmark without mitigation
Benchmark LLM on Classification tasks. Our benchmark dataset includes Adult, ACS, MDgender, wikibias, wikitalks
```bash
export CUDA_VISIBLE_DEVICES=<GPU_ID>
# Provide a list of model IDs or file paths that can be recognized by Hugging Face
python classification_benchamrk.py --model_name_or_path bert-base-uncased roberta-base albert-base-v2 distilbert-base-uncased MoritzLaurer/deberta-v3-large-zeroshot-v2.0
```

## Classification Bias Benchmark with mitigation
Benchmark LLM on Classification tasks. Our benchmark dataset includes Adult, ACS, MDgender, wikitalks, jigsaw
```bash
# Provide a list of model IDs or file paths that can be recognized by Hugging Face
bash classification_mitigation.sh context-debias|auto-debias|adept
```

## Generation Bias Benchmark without mitigation
Benchmark LLM on Generation tasks. Our benchmark dataset includes BOLD, REALTOXICPROMPTS, cnn_dailymai, imdb, jigsaw, wikitext, wikitoxic
```bash
export CUDA_VISIBLE_DEVICES=<GPU_ID>
# Provide a list of model IDs or file paths that can be recognized by Hugging Face
python generation_benchamrk.py --model_name_or_path openai-community/gpt2 xlnet/xlnet-base-cased meta-llama/Llama-2-7b-chat-hf facebook/opt-1.3b Salesforce/ctrl --bias_type gender
```

## Generation Bias Benchmark with mitigation
Benchmark bias debiased LLM on Generation tasks. In our paper, four mitigation method is supported:CDA, INLP, SELFDEBIAS, UNLEARN
```bash
export CUDA_VISIBLE_DEVICES=<GPU_ID>
# reproduce mitigation benchmark using CDA on gender
python generation_mitigation.py --method CDA --bias_type gender
```

## Generate Generation Bias Dataset automatically
Automatically collect generation benchmark dataset from well-known training corpus.
```bash
export CUDA_VISIBLE_DEVICES=<GPU_ID>
export OMP_NUM_THREADS=<num of CPUs>
# generation dataset and categorize dataset using full text
python generation_dataset.py --dataset imdb --num_processes <num of CPUs> --mode text
# categorize realtoxic prompt dataset using prompt
python generation_dataset.py --dataset realtoxic --num_processes <num of CPUs> --mode prompt
```