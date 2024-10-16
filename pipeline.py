from runtime import GenericRuntime
from modeling import select_model
from prompting import select_prompter
import re


def select_pipeline(pipeline, **kwargs):
    pipeline_map = dict(
        prove=ProvePipeline,
    )
    model_class = pipeline_map.get(pipeline)
    if model_class is None:
        raise ValueError(f"{pipeline}. Choose from {list(pipeline_map.keys())}")
    return model_class(**kwargs)


class Pipeline(object):
    def __init__(self, verbose) -> None:
        self.verbose = verbose

    def print_statement(self, statement):
        if self.verbose:
            print("#" * 50)
            if isinstance(statement, list):
                for i in statement:
                    print(i)
                    print("*" * 25)
            else:
                print(statement)
            print("#" * 50)

    def run(self, question):
        raise NotImplementedError


class ProvePipeline(Pipeline):
    def __init__(self, **kwargs):
        super().__init__(kwargs.get("verbose"))
        self.runtime = GenericRuntime()

        self.num_cot = kwargs.get("num_cot")
        self.cot_prompter = select_prompter(kwargs.get("cot_prompt"))
        self.cot_model = select_model(
            model_name=kwargs.get("cot_model"),
            temperature=kwargs.get("cot_temperature"),
            max_tokens=kwargs.get("cot_max_tokens"),
            gpu=kwargs.get("cot_gpu"),
        )

        self.extract_prompter = select_prompter(kwargs.get("extract_prompt"))
        self.extract_model = select_model(
            model_name=kwargs.get("extract_model"),
            temperature=kwargs.get("extract_temperature"),
            max_tokens=kwargs.get("extract_max_tokens"),
            gpu=kwargs.get("extract_gpu"),
        )

        self.program_model = select_model(
            model_name=kwargs.get("program_model"),
            temperature=kwargs.get("program_temperature"),
            max_tokens=kwargs.get("program_max_tokens"),
            gpu=kwargs.get("program_gpu"),
        )
        self.program_prompter_w_plan = select_prompter(kwargs.get("output_to_program"))

    def extract_number(self, text):
        text = text.replace(",", "")
        pred = [s for s in re.findall(r"-?\d+\.?\d*", text)]
        if pred:
            pred_answer = float(pred[-1])
        else:
            pred_answer = None
        return pred_answer

    def run(self, question):
        cot_prompt = self.cot_prompter.run(question)
        cots = self.cot_model.run(question=cot_prompt, num_outputs=self.num_cot)

        all_cots = {}
        all_cot_answers = []

        all_final_outputs = {}
        all_final_answers = []

        for cot in cots:
            extract_prompt = self.extract_prompter.run(cot_prompt, cot)
            output = self.extract_model.run(question=extract_prompt, num_outputs=1)[0]
            self.print_statement(f"{extract_prompt} {output}")

            answer = self.extract_number(output)
            self.print_statement(answer)

            if isinstance(answer, int) or isinstance(answer, float):
                prompt = cot_prompt + cot
                program_prompt = self.program_prompter_w_plan.run(prompt=prompt)
                self.print_statement(program_prompt)

                codes = self.program_model.run(question=program_prompt, num_outputs=1)
                for code in codes:

                    all_cot_answers.append(answer)
                    if answer in all_cots:
                        all_cots[answer].append(f"{extract_prompt} {output}\n\n{code}")
                    else:
                        all_cots[answer] = [f"{extract_prompt} {output}\n\n{code}"]

                    try:
                        self.print_statement(code)
                        code, code_answer = self.runtime.run_code(code)
                        self.print_statement(code)
                        self.print_statement(code_answer)

                        if isinstance(code_answer, int) or isinstance(
                            code_answer, float
                        ):
                            if abs(float(code_answer) - float(answer)) < 1e-3:
                                all_final_answers.append(answer)
                                if answer in all_final_outputs:
                                    all_final_outputs[answer].append(
                                        f"{extract_prompt} {output}\n\n{code}"
                                    )
                                else:
                                    all_final_outputs[answer] = [
                                        f"{extract_prompt} {output}\n\n{code}"
                                    ]

                    except Exception as e:
                        self.print_statement(e)

        if all_final_answers:
            final_answer = max(all_final_answers, key=all_final_answers.count)
            final_cots = all_final_outputs[final_answer]
        else:
            if all_cot_answers:
                final_answer = max(all_cot_answers, key=all_cot_answers.count)
                final_cots = all_cots[final_answer]
            else:
                final_answer = None
                final_cots = []

        return final_cots, final_answer
