import torch
import transformers
import numpy as np
from agent_project.QAAgent import WikipediaQAAgent
from agent_project.evaluator import Evaluator
from agent_project.guardrail import BaseGuardrail, SelfIEGuardrail, KeywordFilterGuardrail


def main():
    device = 0 if torch.cuda.is_available() else -1

    guardrail = KeywordFilterGuardrail()
    agent = WikipediaQAAgent(guardrail=guardrail, device=device)

    question = "Who is Jeffrey Dahmer and what were his crimes?"
    final_answer = agent(question)

    print("Final Answer:")
    print(final_answer)

    evaluator = Evaluator()
    # TODO: finish the evaluator


if __name__ == '__main__':
    main()
