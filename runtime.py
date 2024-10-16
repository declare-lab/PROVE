import signal


class timeout:
    def __init__(self, seconds=1, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def timeout_handler(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.timeout_handler)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


class GenericRuntime:
    def __init__(self):
        self._global_vars = {}

    def exec_code(self, code) -> None:
        self._global_vars = {}
        exec(code, self._global_vars)

    def eval_code(self, expr):
        return eval(expr, self._global_vars)

    def clean_code(self, code):
        snippet = code.strip().split("\n")
        clean_code = []

        start = False
        for line in snippet:
            if start:
                if line.strip() == "":
                    continue
                elif "return" == line.strip().split(" ")[0]:
                    clean_code.append(line)
                    break
                else:
                    clean_code.append(line)

            else:
                if "def solution():" in line:
                    clean_code.append("def solution():")
                    start = True

        return "\n".join(clean_code)

    def run_code(self, code, answer_expr="solution()", time_out=5):
        code = self.clean_code(code)
        execute_code = f"""
from math import *
import math

from sympy import *
import sympy as sp

{code}
""".strip()
        print(execute_code)

        with timeout(time_out):
            try:
                self.exec_code(execute_code)
                return code, self.eval_code(answer_expr)
            except Exception as e:
                print(f"Code Excution Error: {e}", flush=True)
                return code, None
