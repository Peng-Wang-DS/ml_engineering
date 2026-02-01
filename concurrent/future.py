from concurrent import futures
# %% use concurrent future threadpoolexecutor for multithreading
from parallel_with_ray.ray_example2_main import dummy_forecast_all_item_normal, timer, ITEMS
import math,random,time,os
from parallel_with_ray.utilities import register_step
def dummy_forecast_one_item_concurrent(item, seed):
    """Forecast one item with heavy computation -> simulate ML tasks"""
    random.seed(seed)
    computation = sum([
        sum([math.sin(i * k) * math.cos(i / k) + math.exp(k / 1000) * math.log(i + k)
             for k in range(1, 10000)])
        for i in random.sample(range(1, 10_000_000), 500)
    ])
    forecast = random.sample(range(0, 100), 10)
    time.sleep(2)
    return {'item': item, 'forecast': forecast, 'computation': computation}


@timer
@register_step
def dummy_forecast_all_item_threadpool():
    """Multiprocessing with ThreadPoolExecutor.
    Returns a dict: {item: result}
    """

    results = {}

    with futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        future_to_item = {
            executor.submit(
                dummy_forecast_one_item_concurrent, item, 42 + idx
            ): item
            for idx, item in enumerate(ITEMS)
        }

        for count, future in enumerate(futures.as_completed(future_to_item), 1):
            item = future_to_item[future]
            res = future.result()
            results[item] = res
            # print(f"[{count}] Item={item!r}, result={res!r}")

    return results

# %% use concurrent future processpoolexecutor for multithreading


@timer
@register_step
def dummy_forecast_all_item_processpool():
    """Multiprocessing with ProcessPoolExecutor.
    Returns a dict: {item: result}
    """

    results = {}

    with futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        future_to_item = {
            executor.submit(
                dummy_forecast_one_item_concurrent, item, 42 + idx
            ): item
            for idx, item in enumerate(ITEMS)
        }

        for count, future in enumerate(futures.as_completed(future_to_item), 1):
            item = future_to_item[future]
            res = future.result()
            results[item] = res
            # print(f"[{count}] Item={item!r}, result={res!r}")

    return results


if __name__ == '__main__':
    # results_seq = dummy_forecast_all_item_normal()
    
    threadpool_re = dummy_forecast_all_item_threadpool()
    
    processpool_re = dummy_forecast_all_item_processpool()
