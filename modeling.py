import os
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import AzureOpenAI
import time

load_dotenv(".env")


def select_model(model_name: str, **kwargs):
    model_map = dict(
        qwen2_05b=Qwen2_05B_Model,
        qwen2_15b=Qwen2_15BModel,
        qwen2_7b=Qwen2_7B_Model,
        gemma2_2b=Gemma2_2BModel,
        gemma2_9b=Gemma2_9B_Model,
        phi3_38b=Phi3_38B_Model,
        mistral_7b=Mistral_7B_Model,
        llama2_7b=Llama2_7B_Model,
        llama2_13b=Llama2_13B_Model,
        llama3_8b=Llama3_8B_Model,
        llama31_8b=Llama31_8B_Model,
        llama32_1b=Llama32_1B_Model,
        llama32_3b=Llama32_3B_Model,
        gpt4o=GPT4oModel,
    )
    model_class = model_map.get(model_name)
    if model_class is None:
        raise ValueError(f"{model_name}. Choose from {list(model_map.keys())}")
    return model_class(**kwargs)


class Qwen2_05B_Model:
    def __init__(self, **kwargs):
        self.model_id = "Qwen/Qwen2-0.5B-Instruct"
        self.dtype = kwargs.get("dtype", torch.bfloat16)
        self.max_tokens = kwargs.get("max_tokens")
        self.temperature = kwargs.get("temperature")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = torch.compile(
            AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=self.dtype,
            ).to(torch.device(f"cuda:{kwargs.get('gpu')}"))
        )

        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def run(self, question, num_outputs):
        messages = [
            {"role": "user", "content": question},
        ]

        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        if self.temperature == 0.0:
            assert num_outputs == 1, "Temperature = 0 but number of outputs > 1"
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=self.max_tokens,
                do_sample=False,
            )
        else:
            assert num_outputs > 1, "Temperature > 0 but number of outputs = 1"
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=self.max_tokens,
                do_sample=True,
                temperature=self.temperature,
                num_return_sequences=num_outputs,
            )

        response = self.tokenizer.batch_decode(
            outputs[:, input_ids.shape[1] :],
            skip_special_tokens=True,
        )

        return response

    def run_batch(self, questions):
        messages = [[{"role": "user", "content": question}] for question in questions]

        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", padding=True
        ).to(self.model.device)

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=self.max_tokens,
            do_sample=False,
        )

        responses = self.tokenizer.batch_decode(
            outputs[:, input_ids.shape[1] :],
            skip_special_tokens=True,
        )

        return responses


class Qwen2_15BModel:
    def __init__(self, **kwargs):
        self.model_id = "Qwen/Qwen2-1.5B-Instruct"
        self.dtype = kwargs.get("dtype", torch.bfloat16)
        self.max_tokens = kwargs.get("max_tokens")
        self.temperature = kwargs.get("temperature")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = torch.compile(
            AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=self.dtype,
            ).to(torch.device(f"cuda:{kwargs.get('gpu')}"))
        )

        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def run(self, question, num_outputs):
        messages = [
            {"role": "user", "content": question},
        ]

        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        if self.temperature == 0.0:
            assert num_outputs == 1, "Temperature = 0 but number of outputs > 1"
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=self.max_tokens,
                do_sample=False,
            )
        else:
            assert num_outputs > 1, "Temperature > 0 but number of outputs = 1"
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=self.max_tokens,
                do_sample=True,
                temperature=self.temperature,
                num_return_sequences=num_outputs,
            )

        response = self.tokenizer.batch_decode(
            outputs[:, input_ids.shape[1] :],
            skip_special_tokens=True,
        )

        return response

    def run_batch(self, questions):
        messages = [[{"role": "user", "content": question}] for question in questions]

        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", padding=True
        ).to(self.model.device)

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=self.max_tokens,
            do_sample=False,
        )

        responses = self.tokenizer.batch_decode(
            outputs[:, input_ids.shape[1] :],
            skip_special_tokens=True,
        )

        return responses


class Qwen2_7B_Model:
    def __init__(self, **kwargs):
        self.model_id = "Qwen/Qwen2-7B-Instruct"
        self.dtype = kwargs.get("dtype", torch.bfloat16)
        self.max_tokens = kwargs.get("max_tokens")
        self.temperature = kwargs.get("temperature")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = torch.compile(
            AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=self.dtype,
            ).to(torch.device(f"cuda:{kwargs.get('gpu')}"))
        )

        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def run(self, question, num_outputs):
        messages = [
            {"role": "user", "content": question},
        ]

        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        if self.temperature == 0.0:
            assert num_outputs == 1, "Temperature = 0 but number of outputs > 1"
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=self.max_tokens,
                do_sample=False,
            )
        else:
            assert num_outputs > 1, "Temperature > 0 but number of outputs = 1"
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=self.max_tokens,
                do_sample=True,
                temperature=self.temperature,
                num_return_sequences=num_outputs,
            )

        response = self.tokenizer.batch_decode(
            outputs[:, input_ids.shape[1] :],
            skip_special_tokens=True,
        )

        return response

    def run_batch(self, questions):
        messages = [[{"role": "user", "content": question}] for question in questions]

        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", padding=True
        ).to(self.model.device)

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=self.max_tokens,
            do_sample=False,
        )

        responses = self.tokenizer.batch_decode(
            outputs[:, input_ids.shape[1] :],
            skip_special_tokens=True,
        )

        return responses


class Gemma2_2BModel:
    def __init__(self, **kwargs):
        self.model_id = "google/gemma-2-2b-it"
        self.dtype = kwargs.get("dtype", torch.bfloat16)
        self.max_tokens = kwargs.get("max_tokens")
        self.temperature = kwargs.get("temperature")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = torch.compile(
            AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=self.dtype,
            ).to(torch.device(f"cuda:{kwargs.get('gpu')}"))
        )

        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def run(self, question, num_outputs):
        messages = [
            {"role": "user", "content": question},
        ]

        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        if self.temperature == 0.0:
            assert num_outputs == 1, "Temperature = 0 but number of outputs > 1"
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=self.max_tokens,
                do_sample=False,
            )
        else:
            assert num_outputs > 1, "Temperature > 0 but number of outputs = 1"
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=self.max_tokens,
                do_sample=True,
                temperature=self.temperature,
                num_return_sequences=num_outputs,
            )

        response = self.tokenizer.batch_decode(
            outputs[:, input_ids.shape[1] :],
            skip_special_tokens=True,
        )

        return response

    def run_batch(self, questions):
        messages = [[{"role": "user", "content": question}] for question in questions]

        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", padding=True
        ).to(self.model.device)

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=self.max_tokens,
            do_sample=False,
        )

        responses = self.tokenizer.batch_decode(
            outputs[:, input_ids.shape[1] :],
            skip_special_tokens=True,
        )

        return responses


class Gemma2_9B_Model:
    def __init__(self, **kwargs):
        self.model_id = "google/gemma-2-9b-it"
        self.dtype = kwargs.get("dtype", torch.bfloat16)
        self.max_tokens = kwargs.get("max_tokens")
        self.temperature = kwargs.get("temperature")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = torch.compile(
            AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=self.dtype,
            ).to(torch.device(f"cuda:{kwargs.get('gpu')}"))
        )

        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def run(self, question, num_outputs):
        messages = [
            {"role": "user", "content": question},
        ]

        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        if self.temperature == 0.0:
            assert num_outputs == 1, "Temperature = 0 but number of outputs > 1"
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=self.max_tokens,
                do_sample=False,
            )
        else:
            assert num_outputs > 1, "Temperature > 0 but number of outputs = 1"
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=self.max_tokens,
                do_sample=True,
                temperature=self.temperature,
                num_return_sequences=num_outputs,
            )

        response = self.tokenizer.batch_decode(
            outputs[:, input_ids.shape[1] :],
            skip_special_tokens=True,
        )

        return response

    def run_batch(self, questions):
        messages = [[{"role": "user", "content": question}] for question in questions]

        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", padding=True
        ).to(self.model.device)

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=self.max_tokens,
            do_sample=False,
        )

        responses = self.tokenizer.batch_decode(
            outputs[:, input_ids.shape[1] :],
            skip_special_tokens=True,
        )

        return responses


class Phi3_38B_Model:
    def __init__(self, **kwargs):
        self.model_id = "microsoft/Phi-3-mini-4k-instruct"
        self.dtype = kwargs.get("dtype", torch.bfloat16)
        self.max_tokens = kwargs.get("max_tokens")
        self.temperature = kwargs.get("temperature")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = torch.compile(
            AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=self.dtype,
            ).to(torch.device(f"cuda:{kwargs.get('gpu')}"))
        )

        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def run(self, question, num_outputs):
        messages = [
            {"role": "user", "content": question},
        ]

        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        if self.temperature == 0.0:
            assert num_outputs == 1
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=self.max_tokens,
                do_sample=False,
            )
        else:
            assert num_outputs > 1
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=self.max_tokens,
                do_sample=True,
                temperature=self.temperature,
                num_return_sequences=num_outputs,
            )

        response = self.tokenizer.batch_decode(
            outputs[:, input_ids.shape[1] :],
            skip_special_tokens=True,
        )

        return response

    def run_batch(self, questions):
        messages = [[{"role": "user", "content": question}] for question in questions]

        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", padding=True
        ).to(self.model.device)

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=self.max_tokens,
            do_sample=False,
        )

        responses = self.tokenizer.batch_decode(
            outputs[:, input_ids.shape[1] :],
            skip_special_tokens=True,
        )

        return responses


class Mistral_7B_Model:
    def __init__(self, **kwargs):
        self.model_id = "mistralai/Mistral-7B-Instruct-v0.3"
        self.dtype = kwargs.get("dtype", torch.bfloat16)
        self.max_tokens = kwargs.get("max_tokens")
        self.temperature = kwargs.get("temperature")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = torch.compile(
            AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=self.dtype,
                token=os.getenv("HF_TOKEN"),
            ).to(torch.device(f"cuda:{kwargs.get('gpu')}"))
        )

        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def run(self, question, num_outputs):
        messages = [
            {"role": "user", "content": question},
        ]

        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        if self.temperature == 0.0:
            assert num_outputs == 1
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=self.max_tokens,
                do_sample=False,
            )
        else:
            assert num_outputs > 1
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=self.max_tokens,
                do_sample=True,
                temperature=self.temperature,
                num_return_sequences=num_outputs,
            )

        response = self.tokenizer.batch_decode(
            outputs[:, input_ids.shape[1] :],
            skip_special_tokens=True,
        )

        return response

    def run_batch(self, questions):
        messages = [[{"role": "user", "content": question}] for question in questions]

        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", padding=True
        ).to(self.model.device)

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=self.max_tokens,
            do_sample=False,
        )

        responses = self.tokenizer.batch_decode(
            outputs[:, input_ids.shape[1] :],
            skip_special_tokens=True,
        )

        return responses


class Llama2_7B_Model:
    def __init__(self, **kwargs):
        self.model_id = "meta-llama/Llama-2-7b-chat-hf"
        self.dtype = kwargs.get("dtype", torch.bfloat16)
        self.max_tokens = kwargs.get("max_tokens")
        self.temperature = kwargs.get("temperature")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = torch.compile(
            AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=self.dtype,
            ).to(torch.device(f"cuda:{kwargs.get('gpu')}"))
        )

        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def run(self, question, num_outputs):
        messages = [
            {"role": "user", "content": question},
        ]

        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        if self.temperature == 0.0:
            assert num_outputs == 1, "Temperature = 0 but number of outputs > 1"
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=self.max_tokens,
                do_sample=False,
            )
        else:
            assert num_outputs > 1, "Temperature > 0 but number of outputs = 1"
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=self.max_tokens,
                do_sample=True,
                temperature=self.temperature,
                num_return_sequences=num_outputs,
            )

        response = self.tokenizer.batch_decode(
            outputs[:, input_ids.shape[1] :],
            skip_special_tokens=True,
        )

        return response

    def run_batch(self, questions):
        messages = [[{"role": "user", "content": question}] for question in questions]

        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", padding=True
        ).to(self.model.device)

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=self.max_tokens,
            do_sample=False,
        )

        responses = self.tokenizer.batch_decode(
            outputs[:, input_ids.shape[1] :],
            skip_special_tokens=True,
        )

        return responses


class Llama2_13B_Model:
    def __init__(self, **kwargs):
        self.model_id = "meta-llama/Llama-2-13b-chat-hf"
        self.dtype = kwargs.get("dtype", torch.bfloat16)
        self.max_tokens = kwargs.get("max_tokens")
        self.temperature = kwargs.get("temperature")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = torch.compile(
            AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=self.dtype,
            ).to(torch.device(f"cuda:{kwargs.get('gpu')}"))
        )

        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def run(self, question, num_outputs):
        messages = [
            {"role": "user", "content": question},
        ]

        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        if self.temperature == 0.0:
            assert num_outputs == 1, "Temperature = 0 but number of outputs > 1"
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=self.max_tokens,
                do_sample=False,
            )
        else:
            assert num_outputs > 1, "Temperature > 0 but number of outputs = 1"
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=self.max_tokens,
                do_sample=True,
                temperature=self.temperature,
                num_return_sequences=num_outputs,
            )

        response = self.tokenizer.batch_decode(
            outputs[:, input_ids.shape[1] :],
            skip_special_tokens=True,
        )

        return response

    def run_batch(self, questions):
        messages = [[{"role": "user", "content": question}] for question in questions]

        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", padding=True
        ).to(self.model.device)

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=self.max_tokens,
            do_sample=False,
        )

        responses = self.tokenizer.batch_decode(
            outputs[:, input_ids.shape[1] :],
            skip_special_tokens=True,
        )

        return responses


class Llama3_8B_Model:
    def __init__(self, **kwargs):
        self.model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        self.dtype = kwargs.get("dtype", torch.bfloat16)
        self.max_tokens = kwargs.get("max_tokens")
        self.temperature = kwargs.get("temperature")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = torch.compile(
            AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=self.dtype,
            ).to(torch.device(f"cuda:{kwargs.get('gpu')}"))
        )

        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

    def run(self, question, num_outputs):
        messages = [
            {"role": "user", "content": question},
        ]

        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        if self.temperature == 0.0:
            assert num_outputs == 1, "Temperature = 0 but number of outputs > 1"
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=self.max_tokens,
                eos_token_id=self.terminators,
                do_sample=False,
            )
        else:
            assert num_outputs > 1, "Temperature > 0 but number of outputs = 1"
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=self.max_tokens,
                eos_token_id=self.terminators,
                do_sample=True,
                temperature=self.temperature,
                num_return_sequences=num_outputs,
            )

        response = self.tokenizer.batch_decode(
            outputs[:, input_ids.shape[1] :],
            skip_special_tokens=True,
        )

        return response

    def run_batch(self, questions):
        messages = [[{"role": "user", "content": question}] for question in questions]

        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", padding=True
        ).to(self.model.device)

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=self.max_tokens,
            eos_token_id=self.terminators,
            do_sample=False,
        )

        responses = self.tokenizer.batch_decode(
            outputs[:, input_ids.shape[1] :],
            skip_special_tokens=True,
        )

        return responses


class Llama31_8B_Model:
    def __init__(self, **kwargs):
        self.model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        self.dtype = kwargs.get("dtype", torch.bfloat16)
        self.max_tokens = kwargs.get("max_tokens")
        self.temperature = kwargs.get("temperature")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = torch.compile(
            AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=self.dtype,
                token=os.getenv("HF_TOKEN"),
            ).to(torch.device(f"cuda:{kwargs.get('gpu')}"))
        )

        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

    def run(self, question, num_outputs):
        messages = [
            {"role": "user", "content": question},
        ]

        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        if self.temperature == 0.0:
            assert num_outputs == 1, "Temperature = 0 but number of outputs > 1"
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=self.max_tokens,
                eos_token_id=self.terminators,
                do_sample=False,
            )
        else:
            assert num_outputs > 1, "Temperature > 0 but number of outputs = 1"
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=self.max_tokens,
                eos_token_id=self.terminators,
                do_sample=True,
                temperature=self.temperature,
                num_return_sequences=num_outputs,
            )

        response = self.tokenizer.batch_decode(
            outputs[:, input_ids.shape[1] :],
            skip_special_tokens=True,
        )

        return response

    def run_batch(self, questions):
        messages = [[{"role": "user", "content": question}] for question in questions]

        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", padding=True
        ).to(self.model.device)

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=self.max_tokens,
            eos_token_id=self.terminators,
            do_sample=False,
        )

        responses = self.tokenizer.batch_decode(
            outputs[:, input_ids.shape[1] :],
            skip_special_tokens=True,
        )

        return responses


class Llama32_1B_Model:
    def __init__(self, **kwargs):
        self.model_id = "meta-llama/Llama-3.2-1B-Instruct"
        self.dtype = kwargs.get("dtype", torch.bfloat16)
        self.max_tokens = kwargs.get("max_tokens")
        self.temperature = kwargs.get("temperature")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = torch.compile(
            AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=self.dtype,
                token=os.getenv("HF_TOKEN"),
            ).to(torch.device(f"cuda:{kwargs.get('gpu')}"))
        )

        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

    def run(self, question, num_outputs):
        messages = [
            {"role": "user", "content": question},
        ]

        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        if self.temperature == 0.0:
            assert num_outputs == 1, "Temperature = 0 but number of outputs > 1"
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=self.max_tokens,
                eos_token_id=self.terminators,
                do_sample=False,
            )
        else:
            assert num_outputs > 1, "Temperature > 0 but number of outputs = 1"
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=self.max_tokens,
                eos_token_id=self.terminators,
                do_sample=True,
                temperature=self.temperature,
                num_return_sequences=num_outputs,
            )

        response = self.tokenizer.batch_decode(
            outputs[:, input_ids.shape[1] :],
            skip_special_tokens=True,
        )

        return response

    def run_batch(self, questions):
        messages = [[{"role": "user", "content": question}] for question in questions]

        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", padding=True
        ).to(self.model.device)

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=self.max_tokens,
            eos_token_id=self.terminators,
            do_sample=False,
        )

        responses = self.tokenizer.batch_decode(
            outputs[:, input_ids.shape[1] :],
            skip_special_tokens=True,
        )

        return responses


class Llama32_3B_Model:
    def __init__(self, **kwargs):
        self.model_id = "meta-llama/Llama-3.2-3B-Instruct"
        self.dtype = kwargs.get("dtype", torch.bfloat16)
        self.max_tokens = kwargs.get("max_tokens")
        self.temperature = kwargs.get("temperature")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = torch.compile(
            AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=self.dtype,
                token=os.getenv("HF_TOKEN"),
            ).to(torch.device(f"cuda:{kwargs.get('gpu')}"))
        )

        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

    def run(self, question, num_outputs):
        messages = [
            {"role": "user", "content": question},
        ]

        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        if self.temperature == 0.0:
            assert num_outputs == 1, "Temperature = 0 but number of outputs > 1"
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=self.max_tokens,
                eos_token_id=self.terminators,
                do_sample=False,
            )
        else:
            assert num_outputs > 1, "Temperature > 0 but number of outputs = 1"
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=self.max_tokens,
                eos_token_id=self.terminators,
                do_sample=True,
                temperature=self.temperature,
                num_return_sequences=num_outputs,
            )

        response = self.tokenizer.batch_decode(
            outputs[:, input_ids.shape[1] :],
            skip_special_tokens=True,
        )

        return response

    def run_batch(self, questions):
        messages = [[{"role": "user", "content": question}] for question in questions]

        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", padding=True
        ).to(self.model.device)

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=self.max_tokens,
            eos_token_id=self.terminators,
            do_sample=False,
        )

        responses = self.tokenizer.batch_decode(
            outputs[:, input_ids.shape[1] :],
            skip_special_tokens=True,
        )

        return responses


class GPT4oModel:
    def __init__(self, **kwargs):
        self.client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version="2024-02-15-preview",
        )

        self.max_tokens = kwargs.get("max_tokens")
        self.temperature = kwargs.get("temperature")
        self.retry = 5

    def run(self, question, num_outputs):
        message_text = [
            {
                "role": "user",
                "content": question,
            }
        ]
        response = []

        for i in range(self.retry):
            try:
                if self.temperature == 0.0:
                    assert num_outputs == 1, "Temperature = 0 but number of outputs > 1"
                    completion = self.client.chat.completions.create(
                        model="GPT4o",
                        messages=message_text,
                        temperature=0,
                        max_tokens=self.max_tokens,
                        stop=None,
                    )
                else:
                    assert num_outputs > 1, "Temperature > 0 but number of outputs = 1"
                    completion = self.client.chat.completions.create(
                        model="GPT4o",
                        messages=message_text,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        stop=None,
                        n=num_outputs,
                    )

                response = [i.message.content for i in completion.choices]
            except Exception as e:
                print(e)
                time.sleep(60)
                print(f"GPT-4 request failed, retrying {(i)} ...")

        return response
