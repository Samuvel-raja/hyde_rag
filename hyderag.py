import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableConfig
from config import Config

settings = Config()

class ProductionHyDE:
    def __init__(self, retriever, hypo_docs_count=3):
        self.retriever = retriever
        self.hypo_docs_count = hypo_docs_count

        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            max_retries=3,
            api_key=settings.openai_api_key
        )

        self.hypo_prompt = ChatPromptTemplate.from_template("""
        Write a detailed hypothetical document that answers this question perfectly.
        Question: {question}
        Document:
        """)

        self.hyde_chain = self.hypo_prompt | self.llm | StrOutputParser()

        self.answer_prompt = ChatPromptTemplate.from_template("""
        Answer the question based ONLY on the provided context.
        Context: {context}
        Question: {question}
        Answer:
        """)

        self.rag_chain = self._build_rag_chain()

    async def _get_context(self, query: str) -> str:
        hypo_tasks = [
            self.hyde_chain.ainvoke({"question": f"{query} (aspect {i})"})
            for i in range(self.hypo_docs_count)
        ]
        hypo_docs = await asyncio.gather(*hypo_tasks)

        retrieval_tasks = [self.retriever.ainvoke(doc) for doc in hypo_docs]
        docs_list = await asyncio.gather(*retrieval_tasks)

        seen = set()
        context_chunks = []

        for docs in docs_list:
            for doc in docs:
                if doc.page_content not in seen:
                    seen.add(doc.page_content)
                    context_chunks.append(doc.page_content)

        return "\n\n".join(context_chunks[:8])

    def _build_rag_chain(self):
        return (
            {
                "context": RunnableLambda(lambda x: self._get_context(x["question"])),
                "question": lambda x: x["question"]
            }
            | self.answer_prompt
            | self.llm
            | StrOutputParser()
        )

    async def __call__(self, query: str, config: RunnableConfig = None) -> str:
        return await self.rag_chain.ainvoke(
            {"question": query},
            config=config
        )
