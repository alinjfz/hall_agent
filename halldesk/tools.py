"""Read-only tools scoped to the verified location of one report."""

import json
from dataclasses import asdict

from langchain_core.tools import ToolException, tool

from halldesk.retrieval import search_handbook


def triage_tools(store, verified_location):
    @tool
    def get_open_jobs(location: str) -> str:
        """List open and assigned maintenance jobs at the verified report location."""
        if location.strip().casefold() != verified_location.casefold():
            raise ToolException("Use the verified location supplied with this report.")
        return json.dumps(store.jobs(verified_location), ensure_ascii=False)

    @tool
    def search_policy(query: str) -> str:
        """Find up to three relevant resident handbook sections for a short search query."""
        return json.dumps([asdict(section) for section in search_handbook(query)], ensure_ascii=False)

    get_open_jobs.handle_tool_error = True
    return get_open_jobs, search_policy
