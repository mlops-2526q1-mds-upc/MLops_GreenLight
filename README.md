# MLOps_GreenLight

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Detection and classification of traffic lights

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         mlops_greenlight and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── mlops_greenlight   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes mlops_greenlight a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

## How to run the pipeline

First, build your Docker image of choice:

- Dockerfile: Environment for CUDA/Nvidia devices
- DockerfileMac: Environment for Apple Silicon devices
- Dockerfile_cpu: General environment for all devices using CPU

Enter the environment with Bash:

```bash
docker run --gpus all -it --rm -v $(pwd):/workspace <image_name>
```

> [!NOTE]
> Use ```--gpus all``` if you are using an environment that supports execution of code in GPU

Run the pipeline by reproducing it using DVC:

``` bash
dvc repro
```

Enjoy!

## MLflow tracking

- Local runs are not committed: `mlruns/` is gitignored.
- Set a remote tracking server via `.env`:

```bash
MLFLOW_TRACKING_URI=https://dagshub.com/<user>/<repo>.mlflow
```

### References

**Behrendt, K. & Novak, L. (2017).**  
*A Deep Learning Approach to Traffic Lights: Detection, Tracking, and Classification.*  
In *Proceedings of the IEEE International Conference on Robotics and Automation (ICRA).* IEEE.  

**Wu, Y., Kirillov, A., Massa, F., Lo, W.-Y., & Girshick, R. (2019).**  
*Detectron2.*  
Available at: [https://github.com/facebookresearch/detectron2](https://github.com/facebookresearch/detectron2)
