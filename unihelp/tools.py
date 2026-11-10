"""Read-only handbook search the reply drafter may call for one extra lookup."""

import json

from langchain_core.tools import tool

from unihelp.retrieval import search_handbook


@tool
def search_policy(query: str) -> str:
    """Find up to three relevant resident handbook sections. Write a short query in English."""
    return json.dumps([document.metadata | {"text": document.page_content}
                       for document in search_handbook(query)], ensure_ascii=False)
