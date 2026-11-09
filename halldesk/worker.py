import argparse
import os
from time import sleep

from dotenv import load_dotenv

from halldesk.models import configured_provider, setup_issue
from halldesk.store import Store
from halldesk.workflow import process_pending


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Prepare queued UniHelp handoffs without an open dashboard.")
    parser.add_argument("--once", action="store_true", help="Process the current queue and exit.")
    args = parser.parse_args()
    if issue := setup_issue(configured_provider()):
        parser.error(issue)
    store = Store(os.getenv("HALL_DB", "halldesk.sqlite3"))
    store.seed(samples=os.getenv("HALL_SEED_SAMPLE", "").lower() == "true")
    try:
        while True:
            if process_pending(store):
                continue
            if args.once:
                return
            sleep(3)
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()
