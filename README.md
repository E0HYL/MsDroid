# MsDroid (TDSC'22)

### Preparations

1. Install [Androguard](https://androguard.readthedocs.io/en/latest/intro/installation.html) 3.4.0 from source code.
2. Install Pytorch and [Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).
3. Download [`lite_dataset_10.csv`](https://github.com/pkumza/Data_for_LibRadar/blob/master/lite_dataset_10.csv) into `./src/feature/LibRadar/Data`.

### [Source Code](./src)

1. Run the [`train` script](./src/train.py) (see more configurations in the code):

```shell
python train.py -i $input_dir
```

Three new folders are generated in `./src/training`, e.g.,

- ```shell
  # generated behavior subgraphs
  ./training/Graphs/
  └── $input_dir_name
      └── HOP_2
          └── TPL_True
              ├── dataset.pt
              ├── FeatureLen.txt
              └── processed
                  ├── data_0_0.pt
                  ├── ...
  ```

- ```shell
  # mappings of behavior subgraphs and (APK, API) pairs
  ./training/Mappings/
  └── TestAPK_2_True.csv
  ```

- ```shell
  # experiment results with different settings
  ./training/Experiments/
  ├── $timestamp
  │   ├── exp_log.log
  │   ├── models
  │   │   ├── $precission'_'$recall'_'$accuracy'_'$f1'_'$f2
  │   │   └── last_epoch_$epoch_number
  │   ├── scores
  │   │   └── $precission'_'$recall'_'$accuracy'_'$f1'_'$f2.csv
  │   ├── tensorboard
  │   │   └── events.out.tfevents.$timestamp.ecs-tech-research
  │   └── TrainTest
  │       ├── test.pt
  │       └── train.pt
  ├── exp_configs.csv
  └── performance.csv
  ```

2. Run the [`test` script](./src/main.py) (with a trained model `model.pkl` in `./src/classification`):

```shell
python main.py -i $input_dir -o $outputdir
```

For example, the structure of the `$input_dir` is

```shell
$input_dir
├── app-debug.apk
└── Test
    └── app-debug.apk
```

, then the output folder `$output_dir` looks like

```shell
$outputdir
├── decompile
│   ├── app-debug
│   │   └── call.gml
│   └── Test
│       └── app-debug
│           └── call.gml
├── FeatureLen.txt
├── prediction.csv
├── processed
│   ├── data_0_0.pt
│   └── ...
└── result
    ├── opcode
    │   ├── app-debug.csv
    │   └── Test
    │       └── app-debug.csv
    ├── permission
    │   ├── app-debug.csv
    │   └── Test
    │       └── app-debug.csv
    └── tpl
        ├── app-debug.csv
        └── Test
            └── app-debug.csv
```

- `prediction.csv` classification results (*APK ID*, *APK Path*, *Class*).

### [Processed Graph Data](https://github.com/MalwareDetection/GraphDroid/tree/main/Datasets)

`.pt` file is named after *APK ID* and *Behavior Subgraph ID*. 

Mappings between (*APK ID*, *Behavior Subgraph ID*) and (*APK Hash*, *API Name*) for each dataset are available in `Datasets/mappings`

## Citation

If you find this work useful for your research, please consider citing our [paper](https://ieeexplore.ieee.org/document/9762803) ([PDF](https://www.researchgate.net/publication/360208933_MsDroid_Identifying_Malicious_Snippets_for_Android_Malware_Detection)):

```
@ARTICLE{he2023msdroid,
  author={He, Yiling and Liu, Yiping and Wu, Lei and Yang, Ziqi and Ren, Kui and Qin, Zhan},
  journal={IEEE Transactions on Dependable and Secure Computing}, 
  title={MsDroid: Identifying Malicious Snippets for Android Malware Detection}, 
  year={2023},
  volume={20},
  number={3},
  pages={2025-2039},
  doi={10.1109/TDSC.2022.3168285}
}
```
