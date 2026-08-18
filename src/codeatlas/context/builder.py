from codeatlas.search.models import SearchResult


class ContextBuilder:
    def build(self, results: list[SearchResult]) -> str:
        sections = []

        for result in results:
            sections.append(
                f"File: {result.file_path}\n"
                f"Language: {result.language}\n\n"
                f"{result.content}"
            )

        return "\n\n---\n\n".join(sections)