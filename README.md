# Margins in Contrastive Learning

Files for training are contained within the root directory, and `run.py` is the main entry point for training.

## Requirements

Requirements are found in `requirements.txt`. The following packages are required:

- pandas
- numpy
- sentence-transformers
- torch
- faiss (faiss-gpu or faiss-cpu)
- tqdm
- wandb (optional, for logging, a csv logger is added as well)

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Data generation

- run `python generate.py` in the `data_generation` folder.

  - if you do not do this, you will be prompted if you attempt to run `run.py` without the data:

  ```bash
  > No data found in data_generation/data. Do you want to generate it? [y/n]
  ```

- This will generate data according to the _Example Generation_ described in the paper.
- The semantic similarity for example generation is computed using the `e5-small` model.

## Training

`run.py` is the main file. It runs as-is, but supports the following cli arguments:

| Argument    | Description                                          | Default | Choices/Options                                                    |
| ----------- | ---------------------------------------------------- | ------- | ------------------------------------------------------------------ |
| `--model`   | Sentence-transformer model for training              | "all"   | Any model from Hugging Face hub or the ones described in the paper |
| `--dataset` | An existing dataset                                  | "all"   | sst2, sarcastic-headlines                                          |
| `--loss`    | Loss configuration                                   | "all"   | Predefined loss configurations: all/default/best                   |
| `--samples` | Number of training samples                           | 50000   | A list, e.g., "10000 20000"                                        |
| `--project` | Name of the wandb project for logging                | "tmp"   | Any string                                                         |
| `--logcsv`  | Enable additional logging to a CSV file (True/False) | False   | True or False                                                      |

There are additional training parameters defined in the `params.yml` file that you can configure.

## Example runs

```bash
python run.py
$ python run.py --dataset sarcastic-headlines --loss default
$ python run.py --model e5-small --dataset sst2 --loss best --samples 20000
```

## Evaluation

- Code to reproduce results are found in the `figures_and_tables` folder, under respective subfolders for each section.
