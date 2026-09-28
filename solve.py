import time

import pandas as pd

from core import (
    OUTPUT_PATH,
    TOP_K,
    build_context,
    generate_candidates_for_query,
    load_data,
)


def solve():
    t0 = time.time()

    print("[1/4] Loading data...")
    train, queries, items = load_data()

    print(
        f"      train={train.shape} "
        f"queries={queries.shape} "
        f"items={items.shape}"
    )

    print("[2/4] Building indices...")
    ctx = build_context(
        train,
        items,
    )

    print(
        "[3/4] Generating candidates..."
    )

    rows = []

    for qi, q in enumerate(
        queries.itertuples(
            index=False
        ),
        start=1,
    ):
        top = generate_candidates_for_query(
            q,
            ctx,
            top_k=TOP_K,
        )

        # Строгий контракт benchmark.
        top = list(
            dict.fromkeys(
                top
            )
        )[:TOP_K]

        rows.append(
            (
                str(q.query_id),
                " ".join(
                    str(x)
                    for x in top
                ),
            )
        )

        if qi % 200 == 0:
            print(
                f"      processed "
                f"{qi}/{len(queries)}"
            )

    print("[4/4] Writing answer.csv...")

    out = pd.DataFrame(
        rows,
        columns=[
            "query_id",
            "answer",
        ],
    )

    assert len(out) == len(queries)
    assert out["query_id"].astype(str).is_unique

    lengths = (
        out["answer"]
        .str.split()
        .str.len()
    )

    assert lengths.max() <= TOP_K

    out.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    print(
        f"rows={len(out)}"
    )
    print(
        f"answer length: "
        f"min={lengths.min()} "
        f"median={int(lengths.median())} "
        f"max={lengths.max()}"
    )
    print(
        f"total time: "
        f"{time.time() - t0:.1f}s"
    )
    print(
        f"saved: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    solve()