import os
import argparse
import json
import copy
from datetime import datetime
from tqdm import tqdm
from pipeline import select_pipeline
import torch


def main(args):
    if not args.filepath:
        DATA_PATH = f"datasets/{args.dataset}.jsonl"

        directory_name = (
            datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + f"_{os.getpid()}"
        )

        OUTPUT_PATH = f"results/{args.dataset}/{args.cot_model}/{args.pipeline}/{args.cot_prompt}/{args.num_cot}/{args.output_to_program}/{directory_name}"

        os.makedirs(OUTPUT_PATH, exist_ok=True)

        OUTPUT_FILE = f"{OUTPUT_PATH}/output.jsonl"
        CONFIG_FILE = f"{OUTPUT_PATH}/config.json"
        with open(CONFIG_FILE, "w") as f:
            json.dump(vars(args), f, indent=4)
    else:
        OUTPUT_PATH = args.filepath
        OUTPUT_FILE = f"{OUTPUT_PATH}/output.jsonl"
        CONFIG_FILE = f"{OUTPUT_PATH}/config.json"

        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)

        args = argparse.Namespace(**config)

        DATA_PATH = f"datasets/{args.dataset}.jsonl"

    try:
        examples = list(map(json.loads, open(DATA_PATH)))
        pipeline = select_pipeline(
            # CoT Arguments
            num_cot=args.num_cot,
            cot_prompt=args.cot_prompt,
            cot_model=args.cot_model,
            cot_temperature=args.cot_temperature,
            cot_max_tokens=args.cot_max_tokens,
            cot_gpu=args.cot_gpu,
            # Extract Arguments
            extract_prompt=args.extract_prompt,
            extract_model=args.extract_model,
            extract_temperature=args.extract_temperature,
            extract_max_tokens=args.extract_max_tokens,
            extract_gpu=args.extract_gpu,
            # PoT Arguments
            num_program=args.num_program,
            program_prompt=args.program_prompt,
            program_model=args.program_model,
            program_temperature=args.program_temperature,
            program_max_tokens=args.program_max_tokens,
            program_gpu=args.program_gpu,
            # PSC Arguments
            output_to_program=args.output_to_program,
            # Pipeline Arguments
            pipeline=args.pipeline,
            verbose=args.verbose,
        )

        with open(OUTPUT_FILE, "a") as f:
            lines = open(OUTPUT_FILE).readlines()
            start = len(lines)
            scores = [line["score"] for line in map(json.loads, lines)]

            pbar = tqdm(examples[start:], initial=start, total=len(examples))
            for example in pbar:
                question = example["input"]
                answer = example["target"]

                save = copy.deepcopy(example)

                code, prediction = pipeline.run(question)
                try:
                    prediction = float(prediction)
                    score = 1 if abs(prediction - float(answer)) < 1e-3 else 0

                except Exception as e:
                    print(e, flush=True)
                    prediction = ""
                    score = 0

                scores.append(score)

                save["prediction"] = prediction
                save["score"] = score
                save["code"] = code
                f.write(json.dumps(save) + "\n")
                f.flush()
                print(f"Accuracy - {sum(scores) / len(scores)}")

                torch.cuda.empty_cache()

        accuracy = round((sum(scores) / len(scores)) * 100, 2)
        print(accuracy)

        with open("logs.jsonl", "a") as f:
            logs = {"experiment": OUTPUT_PATH, "accuracy": accuracy}
            f.write(json.dumps(logs) + "\n")

        with open(f"{OUTPUT_PATH}/result.jsonl", "w") as f:
            logs = {"experiment": OUTPUT_PATH, "accuracy": accuracy}
            f.write(json.dumps(logs) + "\n")

    except Exception as e:
        print(e)
        with open("logs.jsonl", "a") as f:
            logs = {"experiment": OUTPUT_PATH, "error": str(e)}
            f.write(json.dumps(logs) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # CoT Arguments
    parser.add_argument("--num_cot", default=None, type=int)
    parser.add_argument("--cot_prompt", default=None, type=str)
    parser.add_argument("--cot_model", default=None, type=str)
    parser.add_argument("--cot_temperature", default=None, type=float)
    parser.add_argument("--cot_max_tokens", default=None, type=int)
    parser.add_argument("--cot_gpu", default=None, type=int)

    # Extract Arguments
    parser.add_argument("--extract_prompt", default=None, type=str)
    parser.add_argument("--extract_model", default=None, type=str)
    parser.add_argument("--extract_temperature", default=None, type=float)
    parser.add_argument("--extract_max_tokens", default=None, type=int)
    parser.add_argument("--extract_gpu", default=None, type=int)

    # Program Arguments
    parser.add_argument("--num_program", default=None, type=int)
    parser.add_argument("--program_prompt", default=None, type=str)
    parser.add_argument("--program_model", default=None, type=str)
    parser.add_argument("--program_temperature", default=None, type=float)
    parser.add_argument("--program_max_tokens", default=None, type=int)
    parser.add_argument("--program_gpu", default=None, type=int)

    # PSC Arguments
    parser.add_argument("--output_to_program", default=None, type=str)

    # Pipeline Arguments
    parser.add_argument("--dataset", default="gsm8k", type=str)
    parser.add_argument("--pipeline", default=None, type=str)
    parser.add_argument("--verbose", action="store_true")

    # Utilities
    parser.add_argument("--filepath", default=None, type=str)

    args = parser.parse_args()

    main(args)
