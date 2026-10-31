"""Show how much of Hu Tao's ordered life-memory collection is complete."""

import argparse

from .builder import LifeBuilder
from .store import LifeMemoryStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", default="character/life_stages.json")
    parser.add_argument("--db", default="local/hu_tao_life.sqlite")
    args = parser.parse_args()

    with LifeMemoryStore(args.db) as store:
        builder = LifeBuilder(store, args.plan)
        for row in builder.progress():
            status = row["status"].upper()
            print(
                f'{row["stage_id"]}: {row["title"]} | {status} | '
                f'{row["reviewed_count"]}/{row["memory_count"]} reviewed memories'
            )
        print("READY TO AWAKEN:", "YES" if builder.ready_to_awaken() else "NO")


if __name__ == "__main__":
    main()
