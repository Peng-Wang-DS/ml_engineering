# Python Concurrency Comparison

A comprehensive exploration of Python's concurrency models, comparing sequential processing against multithreading, multiprocessing, and Ray distributed computing.

## Overview

This repository demonstrates performance comparisons across different execution paradigms:
- **Sequential Processing** - Baseline single-threaded execution
- **Multithreading** - Python's `ThreadPoolExecutor` for I/O-bound tasks
- **Multiprocessing** - Python's `ProcessPoolExecutor` for CPU-bound tasks
- **Ray** - Distributed computing framework for scalable parallel processing

## Repository Structure

```
concurrent/
│
├── parallel_with_ray/
│   ├── parallel_ray_example.pdf          # Results: Ray parallel processing with linear models
│   ├── ray_example1_main.py              # Ray implementation: Linear models example
│   ├── ray_example2_main.py              # Ray implementation: Simple function example
│   └── utilities.py                      # Utility functions
│
├── future.py                             # Comparison: Sequential vs ThreadPool vs ProcessPool
├── python_concurrency_diagram.pdf        # Comprehensive Python concurrency cheatsheet
└── ThreadPoolExecutor_Batch.py           # Batch processing implementation
```

## Key Components

### Standard Library Implementations
- **`future.py`** - Direct comparison of Python's standard library approaches showing performance differences between sequential, threading, and multiprocessing execution models

### Ray Framework Examples
- **`ray_example1_main.py`** - Demonstrates Ray's parallel processing capabilities using linear models
- **`ray_example2_main.py`** - Illustrates Ray with simple dummy functions for clear performance benchmarking

### Resources
- **`python_concurrency_diagram.pdf`** - A comprehensive visual reference for Python concurrency patterns (best viewed on desktop)

## Contributing

Feel free to open issues or submit pull requests with additional concurrency patterns or performance benchmarks.