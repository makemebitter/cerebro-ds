# Cerebro-DS: Cerebro on Data Systems
This is repo is for code release of our paper *Distributed Deep Learning on Data Systems: A Comparative Analysis of Approaches* for the sake of reproducibility. In the paper we developed four different approaches (Cerebro-Spark, UDAF, CTQ, and DA) of bringing deep learning to DBMS-resident data. We showed analyses and experiments to study the trade-offs of these approaches. This repo contains the data, source code, original log files, and other artifacts such as the plotting code. These are required to re-produce the results we presented in the paper.

## Prerequists
We used [Greenplum Database](https://greenplum.org/) and [Apache Spark](https://spark.apache.org/). [Cerebro](https://github.com/ADALabUCSD/cerebro-system) is also needed. To run the experiments you will also need a GPU-enabled cluster (we used 8 nodes) with at least 150 GB RAM available on each node. 

## Data

We used two datasets in the paper: [ImageNet 2012](http://image-net.org/challenges/LSVRC/2012/) and [Criteo](http://labs.criteo.com/2013/12/download-terabyte-click-logs/). You can find ways to download them in their respective distribution pages.

## Source code
The source code includes the actual implementation of the different appraoches, data processing, experiment and simulation scripts, and the plotters.
### Implementations
- **Cerebro-Spark**: For this approach please refer to [Cerebro-Spark](https://github.com/ADALabUCSD/cerebro-system).

- **User Defined Aggregate Functions (UDAF)**: This approach has been incorporated into [Apache MADlib](https://github.com/apache/madlib) and released there.
- **Concurrent Targeted Queries (CTQ)**: This approache has two components, the in-DB part can be found and installed from my [Fork of MADlib](https://github.com/makemebitter/madlib/tree/cerebro). The other part is at `src/ctq.py`
- **Direct Access (DA)**: This approache can be found in `src/da.py`.

### Data processing

- **Pre-processing for ImageNet and Criteo**: `src/preprocessing/*`. You may need the Cerebro system running at standalone mode for some of the scripts.
- **Loading ImageNet and Criteo into Greenplum**: Get MADlib's [DB loader](https://github.com/apache/madlib-site/tree/asf-site/community-artifacts/Deep-learning) (included in `src`). Use `src/load_imagenet.sh` and `src/load_criteo.sh` to start the process.
- **Unloading datasets from Greenplum to filesystem and type-casting**: `src/{run_unload_criteo.sh, run_unload_imagenet.sh, etl_imagenet.py, etl_criteo.py}`

### Experiment scripts
- **End-to-end tests**:

```
src/
  run_imagenet.sh #MA, Imagenet
  run_mop.sh #UDAF, Imagenet
  run_ctq.sh #CTQ, Imagenet
  run_da_cerebro_standalone.sh #DA, Imagenet
  run_filesystem_cerebro_standalone.sh #Cerebro-Standalone, Imagenet
  run_spark.sh #Cerebro-Spark, Imagenet
  run_criteo_collection.sh #All approaches, Criteo
  run_breakdown.sh #Breakdown tests, Imagenet
```
- **Drill-down tests**:
```
src/
  run_scalability.sh #Scalability tests
  hetro_simluator.ipynb #Heterogenous workloads simulations
  run_imagenet_model_size.sh #Model size tests
```
### Plotters
`plots/plots.ipynb`

### Loggers
- **CPU logger**: logs/cpu_logger.sh
- **GPU logger**: logs/gpu_logger.sh

### Experiment logs
All past run logs generated during our experiments can be downloaded at [Google Drive Link](https://drive.google.com/file/d/12HxEOhEntq7wLgGQqs1_C2pG5uUraRUS/view?usp=sharing)(15.3GB). Extract them in to `logs` directory. Use these files and the plotter files you can re-produce all the test figures in the paper.



