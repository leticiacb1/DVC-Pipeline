## 🔧 DVC Pipeline

**Pipeline** represents a series of sequential steps.

In [DVC](https://dvc.org/), each step (node of the DAG) in the pipeline is called stage. Stages are responsible for processing data and executing code to generate outputs, which can include machine learning models or other artifacts.

* Download the dataset used [here](https://github.com/Insper/mlops/tree/main/content/classes/16-data-versioning/data). Put the file in data folder. 

### Dependencies

Create a `venv` and install dependencies

```bash
    # Create environment
    $ python3 -m venv venv  

    # Activate environment
    $ source venv/bin/activate

    # Install dependencies
    $ pip install -r requirements.txt
```

### Describing the files 

`src/preproc.py`

Perform small transformations on the data to make it suitable for training.

Generate .parquet file that will be use in train.py file.

`src/train.py`

Takes the outputs generated by preproc.py and uses them to build two models: a *one-hot encoder* and a classifier using *RandomForestClassifier*.

Generate .csv with performance metrics and a confusion matrix image, both information saved in results/

* Check if everything is working:

```bash
    $ python src/preproc.py
    $ python src/train.py 
```   

* Init dvc at the root of the repository:
```bash
 $ dvc init
```

### Create Pipeline

#### 1. Preprocessing Stage

For `src/preproc.py` run successfully:

* Dependencies : `data/banck.csv`
* Execute with : `python3 src/preproc.py`
* Output: `data/bank_preproc.parquet`

Create stage:

```bash
# Project root
  $ dvc stage add --name preproc \
                  --deps src/preproc.py \
                  --deps data/bank.csv \
                  --outs data/bank_preproc.parquet \
                  python src/preproc.py
```

> This will create, at the root of the repository, a YAML file `dvc.yaml` containing the information that defines the pipeline.

```bash
  # Run pipeline
  $ dvc repro
```

#### 2. Train Stage

For `src/train.py` run successfully:

* Dependencies : `data/bank_preproc.parquet`
* Execute with: `python3 src/train.py`
* Output: `models/model.pickle`, `models/ohe.pickle`
`results/confusion_matrix.png`, `results/model_test_metrics.csv`

Create stage:

```bash
  # Project root
  $ dvc stage add --name train \
                  --deps src/train.py \
                  --deps data/bank_preproc.parquet \
                  --outs models/model.pickle \
                  --outs models/ohe.pickle \
                  --outs results/confusion_matrix.png \
                  --outs results/model_test_metrics.csv \
                  python src/train.py
```

> This will update the YAML file `dvc.yaml` adding the information that defines the train pipeline.

```bash
   # Run pipeline
   $ dvc repro
```

View the dag in the terminal, run:

```bash
  $ dvc dag
```

<br>

@2024, Insper. 9° Semester, Computer Engineering.
Machine Learning Ops & Interviews Discipline