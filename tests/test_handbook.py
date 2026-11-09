from halldesk.context import retrieve_policy
from halldesk.retrieval import handbook_index, search_handbook


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
