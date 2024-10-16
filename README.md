# Not All Votes Count! Programs as Verifiers Improve Self-Consistency of Language Models for Math Reasoning

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

3. Create a `.tokens` file and add your Hugging Face token:
    ```bash
    HF_TOKEN=<hf_token>
    ```

## Running Prove

### Example Command (1 GPU: A40 46GB)
To run Prove using a single A40 46GB GPU, use the following command:

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

### Notes:
- For models larger than 7B parameters, such as Llama-2-13B-chat, it is recommended to use at least **1 A100 80GB** GPU or **2 A40 46GB** GPUs.
- You can adjust how models are distributed across multiple GPUs by specifying `--<model>_gpu` parameters for each model in the command.

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
