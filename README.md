# Raidionics processing backend for performing segmentation and computation of standardized report (RADS)

[![License](https://img.shields.io/badge/License-BSD%202--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)
[![](https://img.shields.io/badge/python-(3.7)|3.8|3.9|3.10|(3.11)-blue.svg)](https://www.python.org/downloads/)
[![Paper](https://zenodo.org/badge/DOI/10.3389/fneur.2022.932219.svg)](https://www.frontiersin.org/articles/10.3389/fneur.2022.932219/full)

The code corresponds to the Raidionics backend for running processing pipelines over MRI/CT scans. The segmentation of
a few organs or tumor types, as well as the generation of standardized reports are included.  
The module can either be used as a Python library, as CLI, or as Docker container.

## [Installation](https://github.com/dbouget/raidionics_rads_lib/installation)

```
pip install git+https://github.com/dbouget/raidionics_rads_lib.git
```

## [Continuous integration](https://github.com/dbouget/raidionics_rads_lib/continuous-integration)

| Operating System | Status                                                                                                      |
|------------------|-------------------------------------------------------------------------------------------------------------|
| **Windows**      | ![CI](https://github.com/dbouget/raidionics_rads_lib/workflows/Build%20Windows/badge.svg?branch=master)     |
| **Ubuntu**       | ![CI](https://github.com/dbouget/raidionics_rads_lib/workflows/Build%20Ubuntu/badge.svg?branch=master)      |
| **macOS**        | ![CI](https://github.com/dbouget/raidionics_rads_lib/workflows/Build%20macOS/badge.svg?branch=master)       |
| **macOS ARM**    | ![CI](https://github.com/dbouget/raidionics_rads_lib/workflows/Build%20macOS%20ARM/badge.svg?branch=master) |

## [How to cite](https://github.com/dbouget/raidionics_rads_lib#how-to-cite)
If you are using Raidionics in your research, please cite the following references.

The final software including updated performance metrics for preoperative tumors and introducing postoperative tumor segmentation:
```
@article{bouget2023raidionics,
    author = {Bouget, David and Alsinan, Demah and Gaitan, Valeria and Holden Helland, Ragnhild and Pedersen, André and Solheim, Ole and Reinertsen, Ingerid},
    year = {2023},
    month = {09},
    pages = {},
    title = {Raidionics: an open software for pre-and postoperative central nervous system tumor segmentation and standardized reporting},
    volume = {13},
    journal = {Scientific Reports},
    doi = {10.1038/s41598-023-42048-7},
}
```

For the preliminary preoperative tumor segmentation validation and software features:
```
@article{bouget2022preoptumorseg,
    title={Preoperative Brain Tumor Imaging: Models and Software for Segmentation and Standardized Reporting},
    author={Bouget, David and Pedersen, André and Jakola, Asgeir S. and Kavouridis, Vasileios and Emblem, Kyrre E. and Eijgelaar, Roelant S. and Kommers, Ivar and Ardon, Hilko and Barkhof, Frederik and Bello, Lorenzo and Berger, Mitchel S. and Conti Nibali, Marco and Furtner, Julia and Hervey-Jumper, Shawn and Idema, Albert J. S. and Kiesel, Barbara and Kloet, Alfred and Mandonnet, Emmanuel and Müller, Domenique M. J. and Robe, Pierre A. and Rossi, Marco and Sciortino, Tommaso and Van den Brink, Wimar A. and Wagemakers, Michiel and Widhalm, Georg and Witte, Marnix G. and Zwinderman, Aeilko H. and De Witt Hamer, Philip C. and Solheim, Ole and Reinertsen, Ingerid},
    journal={Frontiers in Neurology},
    volume={13},
    year={2022},
    url={https://www.frontiersin.org/articles/10.3389/fneur.2022.932219},
    doi={10.3389/fneur.2022.932219},
    issn={1664-2295}
}
```

<details>
<summary>

## [Usage](https://github.com/dbouget/raidionics_rads_lib#usage)
</summary>

## CLI
```
raidionicsrads CONFIG
```

CONFIG should point to a configuration file (*.ini), specifying all runtime parameters,
according to the pattern from [**blank_main_config.ini**](https://github.com/dbouget/raidionics-rads-lib/blob/master/blank_main_config.ini).

## Python module
```
from raidionicsrads.compute import run_rads
run_rads(config_filename="/path/to/main_config.ini")
```

## Docker
:warning: The Docker image can only perform inference using the CPU, there is no GPU support at this stage.
```
docker pull dbouget/raidionics-rads:v1.1
```

For opening the Docker image and interacting with it, run:  
```
docker run --entrypoint /bin/bash -v /home/<username>/<resources_path>:/home/ubuntu/resources -t -i --runtime=nvidia --network=host --ipc=host dbouget/raidionics-rads:v1.1
```

The `/home/<username>/<resources_path>` before the column sign has to be changed to match a directory on your local 
machine containing the data to expose to the docker image. Namely, it must contain folder(s) with images you want to 
run inference on, as long as a folder with the trained models to use, and a destination folder where the results will 
be placed.

For launching the Docker image as a CLI, run:  
```
docker run -v /home/<username>/<resources_path>:/home/ubuntu/resources -t -i --runtime=nvidia --network=host --ipc=host dbouget/raidionics-rads:v1.1 -c /home/ubuntu/resources/<path>/<to>/main_config.ini -v <verbose>
```

The `<path>/<to>/main_config.ini` must point to a valid configuration file on your machine, as a relative path to the `/home/<username>/<resources_path>` described above.
For example, if the file is located on my machine under `/home/myuser/Data/RADS/main_config.ini`, 
and that `/home/myuser/Data` is the mounted resources partition mounted on the Docker image, the new relative path will be `RADS/main_config.ini`.  
The `<verbose>` level can be selected from [debug, info, warning, error].

</details>

<details>
<summary>

# Models
</summary>

The trained models are automatically downloaded when running Raidionics or Raidionics-Slicer.
Alternatively, all existing Raidionics models can be browsed [here](https://github.com/dbouget/Raidionics-models/releases/tag/1.2.0) directly.
</details>

<details>
<summary>

# Developers
</summary>

```
git clone https://github.com/dbouget/raidionics_rads_lib.git --recurse-submodules
```
For running inference on GPU through the raidionics_seg_lib backend, your machine must be properly configured
(cf. [here](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html))  

The ANTs library can be manually installed (from source) and be used as a cpp backend rather than Python.
Visit https://github.com/ANTsX/ANTs.


</details>