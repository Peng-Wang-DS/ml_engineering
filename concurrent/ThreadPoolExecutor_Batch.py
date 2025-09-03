import concurrent.futures
import time
import random
import os
from math import floor
# ----- Data -----
team = ['Jake', 'Mary', 'Eric', 'Jo', 'Mike', 'Peng', 'May', 'Lucy',
        'Ma', 'Luo', 'Qui', 'Wang', 'Chen', 'Ally', 'Paul']

# ----- Work -----
def take_a_break(name, batch_id):
    # x = random.randint(1,5)
    x = 1
    time.sleep(x)
    print(f"[Batch {batch_id}] {name} had a break for {x} seconds")
    return {"batch": batch_id, "Name": name, "break_time": x}

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


# ----- Settings -----
batch_size = 3                 # how many per batch
# max parallel tasks inside a batch
workers_per_batch = floor(os.cpu_count() / batch_size)
max_parallel_batches = batch_size  # how many batches run at once

batches = list(chunks(team, batch_size))


# ----- Batch processor -----
def process_batch(batch_id, batch_members):
    # run this batch's people in parallel (up to workers_per_batch at a time)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers_per_batch) as inner:
        futs = [inner.submit(take_a_break, name, batch_id) for name in batch_members]
        for f in concurrent.futures.as_completed(futs):
            results.append(f.result())
    return results


# ----- Run batches in parallel -----
all_results = []
with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel_batches) as outer:
    outer_futs = [
        outer.submit(process_batch, batch_id, batch_members)
        for batch_id, batch_members in enumerate(batches, start=1)
    ]
    for f in concurrent.futures.as_completed(outer_futs):
        all_results.extend(f.result())

print("\nResults:")
print({result['Name']: result['break_time'] for result in all_results})
