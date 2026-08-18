from codeatlas.context.builder import ContextBuilder
from codeatlas.llm.provider import LLMProvider
from codeatlas.search.service import SearchService


class RAGService:
    def __init__(
        self,
        search_service: SearchService,
        context_builder: ContextBuilder,
        llm_provider: LLMProvider,
    ):
        self.search_service = search_service
        self.context_builder = context_builder
        self.llm_provider = llm_provider

    def answer(
        self,
        question: str,
        limit: int = 5,
    ) -> str:
        results = self.search_service.search(
            query=question,
            limit=limit,
        )

        context = self.context_builder.build(results)

        return self.llm_provider.generate(
            question=question,
            context=context,
        )