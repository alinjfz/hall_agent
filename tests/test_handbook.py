from unihelp.context import retrieve_policy
from unihelp.retrieval import handbook_index, search_handbook


def test_handbook_targets_match_structured_policy():
    sections, _ = handbook_index()
    rules = retrieve_policy()
    for section, bands in (
        ("repair-targets", (rules["lighting"]["other_lighting_works"], rules["lighting"]["no_other_lighting"],
                            rules["heating"]["target"], rules["appliance"]["target"])),
        ("water-leaks", (rules["plumbing"]["target"],)),
    ):
        for band in bands:
            assert f'{band["priority"]} with a {band["days"]} day target' in sections[section].text or (
                f'{band["priority"]}, {band["days"]} day target' in sections[section].text
            )


def test_handbook_search_ranks_relevant_sections():
    assert search_handbook("carbon monoxide alarm gas leak")[0].id == "gas-and-carbon-monoxide"
    assert search_handbook("tap shower dripping")[0].id == "water-leaks"
    assert search_handbook("xyzzy") == []


def test_handbook_search_normalizes_terms_and_rejects_weak_matches():
    assert search_handbook("leaking")[0].id == "water-leaks"
    assert search_handbook("lights")[0].id == "repair-targets"
    assert search_handbook("the staff") == []


class KeywordEmbeddings:
    """A stand-in embedding model: one dimension per concept, in any language."""
    CONCEPTS = (("radiator", "heating", "calefacción"), ("tap", "drip", "grifo", "water"), ("quiet", "silencio"))

    def embed_query(self, text):
        text = text.casefold()
        return [float(any(word in text for word in words)) for words in self.CONCEPTS] + [0.01]

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]


def test_embeddings_recover_matches_that_keywords_miss():
    from unihelp.retrieval import handbook_retriever

    assert search_handbook("Mi grifo gotea") == []  # Lexical search cannot cross languages.
    retriever = handbook_retriever(embeddings=KeywordEmbeddings(), model_name="keyword-test")
    assert retriever.metadata == {"retrieval_mode": "hybrid"}
    found = retriever.invoke("Mi grifo gotea")
    assert found[0].id == "water-leaks" and found[0].metadata["matched"] == ["dense"]
    both = retriever.invoke("tap shower dripping")[0]
    assert both.id == "water-leaks" and both.metadata["matched"] == ["lexical", "dense"]


def test_embedding_outage_falls_back_to_keyword_search():
    from unihelp.retrieval import handbook_retriever

    class Broken(KeywordEmbeddings):
        def embed_query(self, text):
            raise ConnectionError

    class BrokenIndex(KeywordEmbeddings):
        def embed_documents(self, texts):
            raise ConnectionError

    retriever = handbook_retriever(embeddings=Broken(), model_name="broken-query-test")
    assert [document.id for document in retriever.invoke("tap shower dripping")] == ["water-leaks"]
    retriever = handbook_retriever(embeddings=BrokenIndex(), model_name="broken-index-test")
    assert retriever.metadata == {"retrieval_mode": "lexical"}
    assert [document.id for document in retriever.invoke("tap shower dripping")] == ["water-leaks"]


def test_query_adds_handbook_terms_for_the_interpreted_facts():
    from unihelp.context import handbook_query

    assert handbook_query("Mi grifo gotea") == "Mi grifo gotea"
    query = handbook_query("Mi grifo gotea", intent="new", category="plumbing", fitting="grifo")
    assert search_handbook(query)[0].id == "water-leaks"
    assert search_handbook(handbook_query("Any update?", intent="chase"))[0].id == "chasing-repairs"
