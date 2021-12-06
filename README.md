[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/bond005/impartial_text_cls/blob/master/LICENSE)
![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg)

# yandex-shifts-weather

The repository contains information about my solution for the Weather Prediction track in the Yandex Shifts challenge https://research.yandex.com/shifts/weather

This information includes two Jupyter-notebooks, several auxiliary Python modules, and trained models.

Tнe proposed solution is the best on this track (see <b><i>SNN Ens U MT Np SpecFE</i></b> method in the corresponding leaderboard).

Reproducibility
---------------

For reproducibility you need to Python 3.7 or later (I checked all with Python 3.7). You have to clone this repository and install all dependencies (I recommend do it in a special Python [environment](https://docs.python.org/3/glossary.html#term-virtual-environment)):

```
git clone https://github.com/bond005/yandex-shifts-weather.git
pip install -r requirements.txt
```

My solution is based on deep learning and uses [Tensorflow](https://www.tensorflow.org). So, you need a CUDA-compatible GPU for more efficient reproducibility of my code (I used two variants: [nVidia V100](https://www.nvidia.com/en-us/data-center/v100) and [nVidia GeForce 2080Ti](https://www.nvidia.com/en-us/geforce/graphics-cards/rtx-2080-ti/)).

After installing the dependencies, you need to do the following steps:

1. Download all data of the Weather Prediction track in the Yandex Shifts challenge and unarchive it into the `data/yandex-shifts/weather` subdirectory. The training and development data can be downloaded [here](https://storage.yandexcloud.net/yandex-research/shifts/weather/canonical-trn-dev-data.tar), and the data for final evaluation stage is available below this [link](https://storage.yandexcloud.net/yandex-research/shifts/weather/canonical-eval-data.tar). As a result, there will be four CSV-files in the abovementioned subdirectory:

- `train.csv`;
- `dev_in.csv`;
- `dev_out.csv`;
- `eval.csv`.

2. Download the baseline models developed by the challenge organizers at the [link](https://storage.yandexcloud.net/yandex-research/shifts/weather/baseline-models.tar), and unarchive their into the `models/yandex-shifts/weather-baseline-catboost` subdirectory. This step is optional, and it is only needed to compare my solution with the baseline on the development set (my pre-trained models are part of this repository, and they are available in the `models/yandex-shifts/weather` subdirectory).

3. Run Jupyter notebook `deep_ensemble_with_uncertainty_and_spec_fe_eval.ipynb` for inference process reproducing. As a result, you will see the quality evaluation of my pre-trained models on the development set in comparison with the baseline. You will also get the predictions on the development and evaluation sets in the corresponding CSV files `models/yandex-shifts/weather/df_submission_dev.csv` and `models/yandex-shifts/weather/df_submission.csv`.

4. If you want to reproduce the united process of training and inference (and not just inference, as in step 3), then you have to run another Jupyter notebook `deep_ensemble_with_uncertainty_and_spec_fe.ipynb`. All pre-trained models in the `models/yandex-shifts/weather` subdirectory will be rewritten. After that predictions will be generated (similar to step 3, but using new models). This step is optional.
