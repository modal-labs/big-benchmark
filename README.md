# 🅱️ig 🅱️enchmark

_A solution for benchmarking many LLMs under many different configurations in parallel on [Modal](https://modal.com)._

## Setup

### Install dependencies

```bash
pip install -e .
```

## Run and plot multiple benchmarks

To run multiple benchmarks at once, first deploy the Datasette UI, which will let you easily view the results later:

```
(cd src && modal deploy -m big_benchmark);
```

Then, start a benchmark suite from a configuration file:

```bash
bb configs/llama3.yaml
```

Once the suite has finished, you will be given a URL to a UI where you can view your results, and a command to download a JSONL file with your results.

## Contributing

We welcome contributions, including those that add tuned benchmarks to our collection.
See the [CONTRIBUTING](/CONTRIBUTING.md) file and the [Getting Started](https://github.com/modal-labs/big-benchmark/wiki/Getting-Started) document for more details on contributing to Big Benchmark.

## License

Big Benchmark is available under the MIT license. See the [LICENSE](/LICENSE.md) file for more details.
