# Not All Votes Count! Programs as Verifiers Improve Self-Consistency of Language Models for Math Reasoning


![](/img/prove_framework.png)

![](/img/prove_improvement.png)


## Setup

1. Create and activate a new Conda environment:
    ```bash
    conda create -n prove python=3.10 -y
    conda activate prove
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Create a `.env` file and add your Hugging Face token:
    ```bash
    HF_TOKEN=<hf_token>
    ```

## Running Prove

### Example Command 
To run Prove using a single GPU, use the following command:

```bash
python eval.py \
    --num_cot 16 \
    --cot_prompt <prompt> \
    --cot_model <model> \
    --cot_temperature 0.7 \
    --cot_max_tokens 1024 \
    --cot_gpu 0 \
    --extract_prompt extract \
    --extract_model phi3_38b \
    --extract_temperature 0.0 \
    --extract_max_tokens 32 \
    --extract_gpu 0 \
    --output_to_program output2pot \
    --program_model phi3_38b \
    --program_temperature 0.0 \
    --program_max_tokens 1024 \
    --program_gpu 0 \
    --pipeline prove \
    --dataset <dataset> \
```


## Running Prove on MATH

### API Setup
Include GPT-4o API key and endpoint in `.env` file:

```
AZURE_OPENAI_KEY=<YOUR_AZURE_OPENAI_KEY>
AZURE_ENDPOINT=<YOUR_AZURE_ENDPOINT>
```

### Example Command
To run Prove using a single GPU, use the following command:

```bash
python eval.py \
    --num_cot 16 \
    --cot_prompt <prompt> \
    --cot_model <model> \
    --cot_temperature 0.7 \
    --cot_max_tokens 1024 \
    --cot_gpu 0 \
    --output_to_program output2pot \
    --program_model gpt4o \
    --program_temperature 0.0 \
    --program_max_tokens 1024 \
    --pipeline prove \
    --dataset math500 \
```


## Supported Models
The following models are supported for the pipeline:

| Model Identifier | Model Name                           |
|------------------|--------------------------------------|
| `qwen2_05b`      | Qwen2-0.5B-Instruct                  |
| `qwen2_15b`      | Qwen2-1.5B-Instruct                  |
| `qwen2_7b`       | Qwen2-7B-Instruct                    |
| `gemma2_2b`      | Gemma-2-2B-it                        |
| `gemma2_9b`      | Gemma-2-9B-it                        |
| `phi3_38b`       | Phi-3-mini-4k-instruct               |
| `mistral_7b`     | Mistral-7B-Instruct-v0.3             |
| `llama2_7b`      | Llama-2-7B-chat                      |
| `llama2_13b`     | Llama-2-13B-chat                     |
| `llama3_8b`      | Llama-3-8B-Instruct                  |
| `llama31_8b`     | Llama-3.1-8B-Instruct                |
| `llama32_1b`     | Llama-3.2-1B-Instruct                |
| `llama32_3b`     | Llama-3.2-3B-Instruct                |

## Supported Prompts
Choose from the following prompts:

- `direct`
- `cot`
- `ps`

## Supported Datasets
Choose from the following datasets:

- `gsm8k`
- `svamp`
- `asdiv`
- `mawpsmultiarith`
- `mawpssingleeq`
- `mawpssingleop`
- `mawpsaddsub`
- `math500`





## Citation
Please consider citing the following article if you found our work useful:
```bibtex
@article{toh2024not,
  title={Not All Votes Count! Programs as Verifiers Improve Self-Consistency of Language Models for Math Reasoning},
  author={Toh, Vernon YH and Ghosal, Deepanway and Poria, Soujanya},
  journal={arXiv preprint arXiv:2410.12608},
  year={2024}
}
```