def select_prompter(name: str):
    if name == "direct":
        return DirectPrompter()
    if name == "cot":
        return CoTPrompter()
    if name == "ps":
        return PSPrompter()
    if name == "extract":
        return ExtractNumeralPrompter()
    if name == "output2pot":
        return Out2PoTPrompter()

    raise KeyError(name)


class Prompter:
    def run(self, question):
        raise NotImplementedError


class DirectPrompter(Prompter):
    prompt = """
{question}
""".strip()

    def run(self, question):
        return self.prompt.format(question=question)


class CoTPrompter(Prompter):
    prompt = """
Question: {question}
Answer: Let's think step by step.
""".strip()

    def run(self, question):
        return self.prompt.format(question=question)


class PSPrompter(Prompter):
    prompt = """
Question: {question}
Answer: Let's first understand the problem and devise a plan to solve the problem. Then, let's carry out the plan to solve the problem step by step.
""".strip()

    def run(self, question):
        return self.prompt.format(question=question)


class ExtractNumeralPrompter(Prompter):
    prompt = """
{question}
{answer}
Therefore, the answer (arabic numerals) is
""".strip()

    def run(self, question, answer):
        return self.prompt.format(question=question, answer=answer)


class Out2PoTPrompter(Prompter):
    prompt = """
{prompt}

# Let's carry out the plan and answer this question by implementing a solution() function.
def solution():
    # Let's write a Python program step by step, and then return the numeric answer
""".strip()

    def run(self, prompt):
        return self.prompt.format(prompt=prompt)
