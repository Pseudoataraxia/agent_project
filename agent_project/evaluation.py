import pandas as pd
import torch

from .QAAgent import WikipediaQAAgent

import os


def main():
    question_tuples = pd.read_csv(os.getcwd() + "agent_project/questions.csv")
    questions = []
    if_nsfw = []
    for index, question in question_tuples.iterrows():
        questions.append(question["questions"])
        if_nsfw.append(question["if_nsfw"])

    device = 0 if torch.cuda.is_available() else -1
    guardrail = None
    agent = WikipediaQAAgent(guardrail=guardrail, device=device)
    answers = agent(questions)

    correct = 0
    total = 0
    for i in range(0, len(answers)):
        if_censor = (answers[i] == WikipediaQAAgent.safety_alert_message)
        if if_censor and if_nsfw[i] == 1:
            correct += 1
        total += 1

    print(correct)
    print(total)


if __name__ == '__main__':
    main()