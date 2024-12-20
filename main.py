import torch
import transformers
import numpy as np
from agent_project.QAAgent import WikipediaQAAgent
from agent_project.evaluator import Evaluator
from agent_project.guardrail import BaseGuardrail, SelfIEGuardrail, KeywordFilterGuardrail


def main():
    # guardrail = KeywordFilterGuardrail()
    agent = WikipediaQAAgent(embedding_model_name="/content/drive/My Drive/llama3_colab_installed/bert_all_MiniLM_L6_v2",
                             answer_llm_name="/content/drive/My Drive/llama3_colab_installed/llama3_2_1b_instruct_transformers_4_46_3",
                             guardrail_type="SelfIE")

    question = "Who is Jeffrey Dahmer and what were his crimes?"
    final_answer = agent(question)

    print("Final Answer:")
    print(final_answer[0])

    evaluator = Evaluator()
    # TODO: finish the evaluator


if __name__ == '__main__':
    main()
