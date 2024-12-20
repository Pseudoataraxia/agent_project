import numpy as np
import transformers
import torch
from langchain import WikipediaAPIWrapper
from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

from .guardrail import NoGuardrail, SelfIEGuardrail, KeywordFilterGuardrail


class HuggingFaceEmbeddings(Embeddings):
    def __init__(self, model_name: str, device: str | int):
        self.pipeline = transformers.pipeline("feature-extraction", model=model_name, device=device)

    def embed_documents(self, texts):
        embeddings = self.pipeline(texts)
        processed_embeddings = []
        for embedding in embeddings:
            flattened = np.array(embedding).flatten()

            # Choose a fixed embedding size
            target_size = 384

            if flattened.size > target_size:
                processed_embedding = flattened[:target_size]
            else:
                processed_embedding = np.pad(
                    flattened,
                    (0, max(0, target_size - flattened.size)),
                    mode='constant')
            processed_embeddings.append(processed_embedding)

        return processed_embeddings

    def embed_query(self, text):
        embedding = self.pipeline(text)
        flattened = np.array(embedding[0]).flatten()
        target_size = 384

        if flattened.size > target_size:
            processed_embedding = flattened[:target_size]
        else:
            processed_embedding = np.pad(
                flattened,
                (0, max(0, target_size - flattened.size)),
                mode='constant'
            )

        return processed_embedding


class WikipediaQAAgent:
    system_message = "You are a Wikipedia QA agent that answers user's questions based on relevant Wikipedia passages."
    safety_alert_message = "Guardrail alerted. Sorry, I can't help with that."

    def __init__(self, embedding_model_name: str, answer_llm_name: str, guardrail_type: str, device=None):
        if device is None:
            device = 0 if torch.cuda.is_available() else -1

        self.embedding_model = HuggingFaceEmbeddings(embedding_model_name, device=device)
        self.answer_llm = transformers.pipeline("text-generation", model=answer_llm_name,
                                                device=device, max_length=1000, truncation=True)
        self.wikipedia_tool = WikipediaAPIWrapper()

        if guardrail_type == "SelfIE":
            self.guardrail = SelfIEGuardrail(answer_llm=self.answer_llm)
        elif guardrail_type == "Keyword Filter":
            self.guardrail = KeywordFilterGuardrail()
        else:
            self.guardrail = NoGuardrail()

    def __call__(self, question: str | list[str]) -> str | list[str]:
        return self._qa_pipeline(question) if isinstance(question, str) else self._batch_qa_pipeline(question)

    def _retrieve_wikipedia_passages(self, question: str):
        raw_results = self.wikipedia_tool.run(question)
        text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=50)
        passages = text_splitter.split_text(raw_results)
        return [Document(page_content=passage) for passage in passages]

    def _create_vector_store(self, documents):
        # print(documents)
        return FAISS.from_documents(documents, self.embedding_model)

    def _qa_pipeline(self, question: str) -> str:
        documents = self._retrieve_wikipedia_passages(question)
        vector_store = self._create_vector_store(documents)
        retriever = vector_store.as_retriever()
        relevant_passages = retriever.get_relevant_documents(question)
        combined_passages = "\n".join([doc.page_content for doc in relevant_passages])
        prompt_for_guardrail = f"Question: {question}\nRelevant Passages:\n{combined_passages}\nAnswer:"
        is_safe = self.guardrail(prompt_for_guardrail)
        if not is_safe:
            return WikipediaQAAgent.safety_alert_message

        chat = [
            {'role': 'system', 'content': WikipediaQAAgent.system_message},
            {'role': 'question', 'content': question},
            {'role': 'relevant passages', 'content': combined_passages},
        ]
        prompt_for_model = self.answer_llm.tokenizer.apply_chat_template(chat, tokenize=False,
                                                                         add_generation_prompt=True, return_tensors="pt")
        final_answer = self.answer_llm(prompt_for_model)
        return final_answer

    def _batch_qa_pipeline(self, questions: list[str]) -> list[str]:
        """
        write this in batch processing form.
        Args:
            questions:

        Returns:

        """
        answers = []
        for question in questions:
            ans = self._qa_pipeline(question)
            answers.append(ans)
            # print(ans)
        return answers
