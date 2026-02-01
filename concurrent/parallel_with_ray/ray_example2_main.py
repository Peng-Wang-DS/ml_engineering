# %% python concurrency - parallel with ray - example code
import os
import warnings
# Suppress warnings BEFORE importing ray
warnings.filterwarnings('ignore', category=FutureWarning)
os.environ['RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO'] = '0'
os.environ["RAY_DISABLE_METRICS"] = "1"
os.environ["RAY_METRICS_EXPORT_PORT"] = ""

import ray
import time
import math
import random

def pretty_print(label: str, seconds: float, colour: str = "green"):
    
    colours = {
        "green": "\033[92m",
        "blue": "\033[94m",
        "yellow": "\033[93m",
        "red": "\033[91m",
        "bold": "\033[1m",
        "reset": "\033[0m",
    }
    c = colours.get(colour, "")
    r = colours["reset"]

    print(f"\n{c}{'=' * 10} {label}: {seconds:.2f} seconds {'=' * 10}{r}\n")
def init_ray():
    ray.init(
        ignore_reinit_error=True,
        logging_level='ERROR',
        log_to_driver=False,
        configure_logging=False,
        include_dashboard=False
    )

ITEMS = ['apple', 'orange', 'pear', 'cucumber', 'watermelon', 'mango', 'grape', 'tomato',
         'banana', 'strawberry', 'blueberry', 'pineapple', 'kiwi', 'peach', 'plum',
         'cherry', 'lemon', 'avocado', 'broccoli', 'carrot']

def timer(fn):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = fn(*args, **kwargs)
        end_time = time.time()
        pretty_print(label=fn.__name__, seconds=end_time - start_time, colour='green')
        return result
    return wrapper

@ray.remote
def dummy_forecast_one_item_ray(item, seed):
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
def dummy_forecast_all_item_normal():
    """Sequential/Normal processing"""
    results = []
    for idx, item in enumerate(ITEMS):
        time.sleep(2)
        print(f'forecasting {item}...')
        random.seed(42 + idx)
        computation = sum([
            sum([math.sin(i * k) * math.cos(i / k) + math.exp(k / 1000) * math.log(i + k)
                 for k in range(1, 10000)])
            for i in random.sample(range(1, 10_000_000), 500)
        ])
        forecast = random.sample(range(0, 100), 10)
        results.append({'item': item, 'forecast': forecast, 'computation': computation})
    return results

@timer
def dummy_forecast_all_item_ray():
    """Parallel processing with Ray"""
    futures = [dummy_forecast_one_item_ray.remote(item, 42 + idx) 
               for idx, item in enumerate(ITEMS)]
    return ray.get(futures)

if __name__ == "__main__":
    print("=" * 60)
    print("SEQUENTIAL PROCESSING")
    print("=" * 60)
    dummy_forecast_all_item_normal()
    
    print("\n" + "=" * 60)
    print("PARALLEL PROCESSING WITH RAY")
    print("=" * 60)
    init_ray()
    dummy_forecast_all_item_ray()
    
    ray.shutdown()