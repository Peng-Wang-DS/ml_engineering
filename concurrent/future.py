from concurrent import futures
# %% use concurrent future threadpoolexecutor for multithreading
from parallel_with_ray.ray_example2_main import dummy_forecast_all_item_normal, timer, ITEMS
import math,random,time,os


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
def dummy_forecast_all_item_threadpool():
    """Multithreading with ThreadPoolExecutor"""
    print("=" * 60)
    print('Multithreading with ThreadPoolExecutor')
    print("=" * 60)
    with futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        to_do: list[futures.Future] = []

        for idx, item in enumerate(ITEMS):
            future = executor.submit(
                dummy_forecast_one_item_concurrent,
                item,
                42 + idx,
            )
            to_do.append(future)

        for count, future in enumerate(futures.as_completed(to_do), 1):
            res: str = future.result()
            print(f"{future} result: {res!r}")

    return count

# %% use concurrent future processpoolexecutor for multithreading
@timer
def dummy_forecast_all_item_processpool():
    """Multiprocessing with ProcessPoolExecutor"""
    print("=" * 60)
    print('Multiprocessing with ProcessPoolExecutor')
    print("=" * 60)
    with futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        to_do = [
            executor.submit(dummy_forecast_one_item_concurrent, item, 42 + idx)
            for idx, item in enumerate(ITEMS)
        ]

        for count, future in enumerate(futures.as_completed(to_do), 1):
            res = future.result()
            print(f"{future} result: {res!r}")

    return count


if __name__ == '__main__':
    print("=" * 60)
    print("SEQUENTIAL PROCESSING")
    print("=" * 60)
    results_seq = dummy_forecast_all_item_normal()
    
    print("=" * 60)
    print("MultiThread PROCESSING")
    print("=" * 60)
    threadpool_re = dummy_forecast_all_item_threadpool()

    print("=" * 60)
    print("MultiProcess PROCESSING")
    print("=" * 60)
    processpool_re = dummy_forecast_all_item_processpool()
