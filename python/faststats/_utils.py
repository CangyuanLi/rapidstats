import concurrent.futures
import multiprocessing
from typing import Literal, Union

import tqdm


def _run_concurrent(
    fn,
    iterable,
    executor: Union[
        Literal["threads", "processes"],
        concurrent.futures.ThreadPoolExecutor,
        concurrent.futures.ProcessPoolExecutor,
    ],
    preserve_order: bool = False,
    **executor_kwargs,
):
    if executor_kwargs.get("max_workers") == 1:
        return [fn(i) for i in iterable]

    if executor == "threads":
        executor = concurrent.futures.ThreadPoolExecutor(**executor_kwargs)
    elif executor == "processes":
        if "context" not in executor_kwargs:
            executor_kwargs["context"] = multiprocessing.get_context("spawn")
        executor = concurrent.futures.ProcessPoolExecutor(**executor_kwargs)

    if preserve_order:
        with executor as pool:
            res = pool.map(fn, iterable)

        return res

    with executor as pool:
        futures = [pool.submit(fn, i) for i in iterable]
        res = []
        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            res.append(future.result())

    return res
