import numpy as np
import transformers
import torch
from langchain import WikipediaAPIWrapper
from langchain.docstore.document import Document
# from langchain.llms import HuggingFacePipeline
from langchain.embeddings.base import Embeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

from .guardrail import NoGuardrail


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
    safety_alert_message = "Sorry, I can't help with that."
    # TODO: might change this message to a better one?

    def __init__(self, embedding_model_name, answer_llm_name, guardrail, device=None):
        if device is None:
            device = 0 if torch.cuda.is_available() else -1

        self.embedding_model = HuggingFaceEmbeddings(embedding_model_name, device=device)
        # TODO: make embeddings models loaded from locally saved weights

        self.answer_llm = transformers.pipeline("text-generation", model=answer_llm_name, repo_type='local',
                                                device=device, max_length=1000, truncation=True)
        # TODO: change answer_model to a locally saved model; also change the model to llama 3.1 7b-instruct
        # I have already downloaded llama 3.1 7b-instruct on my google drive.

        self.wikipedia_tool = WikipediaAPIWrapper()
        self.guardrail = guardrail if guardrail is not None else NoGuardrail()

    def __call__(self, question: str | list[str]) -> str | list[str]:
        return self._qa_pipeline(question) if isinstance(question, str) else self._batch_qa_pipeline(question)

    def _retrieve_wikipedia_passages(self, question: str):
        raw_results = self.wikipedia_tool.run(question)
        text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=50)
        passages = text_splitter.split_text(raw_results)
        return [Document(page_content=passage) for passage in passages]

    def _create_vector_store(self, documents):
        print(documents)
        return FAISS.from_documents(documents, self.embedding_model)

    def _qa_pipeline(self, question: str) -> str:
        documents = self._retrieve_wikipedia_passages(question)
        vector_store = self._create_vector_store(documents)
        retriever = vector_store.as_retriever()
        relevant_passages = retriever.get_relevant_documents(question)
        combined_passages = "\n".join([doc.page_content for doc in relevant_passages])
        prompt = f"Question: {question}\nPassages:\n{combined_passages}\nAnswer:"
        is_safe = self.guardrail(prompt)
        if not is_safe:
            return WikipediaQAAgent.safety_alert_message
        final_answer = self.answer_llm(prompt)
        return final_answer

    def _batch_qa_pipeline(self, questions: list[str]) -> list[str]:
        answers = []
        for question in questions:
            ans = self._qa_pipeline(question)
            answers.append(ans)
            print(ans)
        return answers
