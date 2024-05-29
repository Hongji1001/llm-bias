#! /bin/bash

eval "$(conda shell.bash hook)"

setup_verifier() {
    local method
    method=$1
    current_dir=$(pwd)
    envs=$(conda env list | awk '{print $1}')

    case "$method" in
    "adept")
        if echo "$envs" | grep -q 'adept'; then
            echo "ADEPT env have been set up."
            conda activate adept
        else
            echo "Install ADEPT for the first time."
            git clone https://github.com/EmpathYang/ADEPT.git
            conda create -n adept python=3.8.5
            conda activate adept
            pip install torch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 nltk
            conda install cudatoolkit=11.0
            cd ADEPT || exit
            pip install -r requirements.txt
            cd ..
        fi
        ;;

    "auto-debias")
        if echo "$envs" | grep -q 'auto-debias'; then
            echo "auto-debias env have been set up."
            conda activate auto-debias
        else
            echo "Install auto-debias for the first time."
            conda create -n auto-debias python=3.10
            conda activate auto-debias
            pip install numpy torch transformers
        fi
        ;;

    "context-debias")
        if echo "$envs" | grep -q 'context-debias'; then
            echo "context-debias env have been set up."
            conda activate context-debias
        else
            echo "Install context-debias for the first time."
            git clone https://github.com/kanekomasahiro/context-debias.git
            conda create -n context-debias python=3.7.3
            conda activate context-debias
            pip install torch==1.5.0 nltk==3.5 transformers==2.8.0 tensorboard==2.0.2
            cd context-debias/transformers || exit
            pip install .
            pip install protobuf==3.20.1
            cd ..
            curl -o data/news-commentary-v15.en.gz https://data.statmt.org/news-commentary/v15/training-monolingual/news-commentary-v15.en.gz
            gunzip data/news-commentary-v15.en.gz
        fi
        ;;
    esac

    # Return to the initial directory
    cd "$current_dir" || exit
}

# Function to show usage
show_usage() {
    echo "Usage: $0 {context-debias|auto-debias|adept}"
    exit 1
}

# Check if the method is provided
if [ $# -ne 1 ]; then
    show_usage
fi

# Method to run
method=$1

case $method in
"context-debias")
    echo "Running context-debias method"

    setup_verifier context-debias
    context_debias_models=("bert" "roberta" "albert" "dbert" "deberta")

    cd context-debias/script || exit
    for model in "${context_debias_models[@]}"; do
        ./preprocess.sh "$model" ../data/news-commentary-v15.en
        ./debias.sh "$model" 2
    done
    cd ../..

    conda activate bias-benchmark
    for model in "${context_debias_models[@]}"; do
        python classification_benchmark.py --model_name_or_path context-debias/debiased_models/42/"$model"/checkpoint-best
    done
    ;;

"auto-debias")
    echo "Running auto-debias method"

    setup_verifier auto-debias
    auto_debias_models=("bert-base-uncased" "roberta-base" "albert-base-v2" "distilbert-base-uncased" "microsoft/deberta-v3-base")
    auto_debias_model_types=("bert" "roberta" "albert" "dbert" "deberta")

    for i in "${!auto_debias_models[@]}"; do
        model_name=${auto_debias_models[i]}
        model_type=${auto_debias_model_types[i]}
        # python generate_prompts.py --debias_type gender --model_type "$model_type" --model_name_or_path "$model_name"
        prompts_file="prompts_${model_name}_gender"
        python auto-debias.py --debias_type gender --model_type "$model_type" --model_name_or_path "$model_name" --prompts_file "$prompts_file"
        prompts_file="prompts_${model_name}_religion"
        python auto-debias.py --debias_type race --model_type "$model_type" --model_name_or_path "$model_name" --prompts_file "$prompts_file"
    done

    # conda activate bias-benchmark
    # for model in "${auto_debias_models[@]}"; do
    #     python classification_benchmark.py --model_name_or_path Auto-Debias/model/debiased_model_bert-base-uncased_gender
    # done
    ;;

"adept")
    echo "Running adept method"
    setup_verifier adept

    adept_models=("roberta" "albert" "dbert")
    adept_models_types=("roberta-base" "albert-base-v2" "distilbert-base-uncased")
    cd adept/script || exit

    # Iterate over models and run the collect_sentences.sh and debias.sh scripts
    for model in "${adept_models[@]}"; do
        # bash ./collect_sentences.sh "$model" ../data/news-commentary-v15.en gender final
        bash ./debias.sh "$model" 3 ADEPT gender
        # bash ./debias.sh "$model" 3 ADEPT religion # TODO: not run
    done

    # cd ../..
    # conda activate bias-benchmark
    # for i in "${!adept_models[@]}"; do
    #     model_name=${adept_models[i]}
    #     model_type=${adept_models_types[i]}
    #     python classification_benchmark.py --model_name_or_path ADEPT/debiased_models/42/"${model}"/ADEPT/gender/final/best_model_ckpt
    # done
    ;;

"*")
    show_usage
    ;;
esac
