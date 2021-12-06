[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/bond005/impartial_text_cls/blob/master/LICENSE)
![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg)

# yandex-shifts-weather

The repository contains information about my solution for the Weather Prediction track in the Yandex Shifts challenge https://research.yandex.com/shifts/weather

This information includes two my Jupyter-notebooks (`deep_ensemble_with_uncertainty_and_spec_fe_eval.ipynb` for evaluating only, and `deep_ensemble_with_uncertainty_and_spec_fe.ipynb` for all stages of my experiment), several auxiliary Python modules (from https://github.com/yandex-research/shifts/tree/main/weather), and my pre-trained models (see the `models/yandex-shifts/weather` subdirectory).

The proposed solution is the best on this track (see <b><i>SNN Ens U MT Np SpecFE</i></b> method in the corresponding leaderboard).

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

- `train.csv`
- `dev_in.csv`
- `dev_out.csv`
- `eval.csv`

2. Download the baseline models developed by the challenge organizers at the [link](https://storage.yandexcloud.net/yandex-research/shifts/weather/baseline-models.tar), and unarchive their into the `models/yandex-shifts/weather-baseline-catboost` subdirectory. This step is optional, and it is only needed to compare my solution with the baseline on the development set (my pre-trained models are part of this repository, and they are available in the `models/yandex-shifts/weather` subdirectory).

3. Run Jupyter notebook `deep_ensemble_with_uncertainty_and_spec_fe_eval.ipynb` for inference process reproducing. As a result, you will see the quality evaluation of my pre-trained models on the development set in comparison with the baseline. You will also get the predictions on the development and evaluation sets in the corresponding CSV files `models/yandex-shifts/weather/df_submission_dev.csv` and `models/yandex-shifts/weather/df_submission.csv`.

4. If you want to reproduce the united process of training and inference (and not just inference, as in step 3), then you have to run another Jupyter notebook `deep_ensemble_with_uncertainty_and_spec_fe.ipynb`. All pre-trained models in the `models/yandex-shifts/weather` subdirectory will be rewritten. After that predictions will be generated (similar to step 3, but using new models). This step is optional.

Key ideas of the proposed method
--------------------------------

A special deep ensemble (i.e.  ensemble of deep neural networks) with uncertainty, hierarchicalmultitask learning and some other features is proposed. It is built on the basis of the following key techniques:

1.  A simple **preprocessing** is applied **to the input data**:
- *imputing*:  the missing values are replaced in all input columns following a simple constant strategy (fill value is −1);
- *quantization*: each input column is discretized into quantile bins, and the number of these bins is detected automatically; after that the bin identifier can be considered as a quantized numerical value of the original feature;
- *standardization*: each quantized column is standardized by removing the mean and scaling to unit variance;
- *decorrelation*:  all possible linear correlations are removed from the feature vectors discretized by above-mentioned way; the decorrelation is implemented using [principal component analysis (PCA)](https://en.wikipedia.org/wiki/Principal_component_analysis).

2. An even simpler **preprocessing** is applied **to targets**: it is based only on removing the mean and scaling to unit variance.

3. A **deep neural network** is built for regression with uncertainty:

- *self-normalization*: this is a self-normalized neural network, also known as SNN \[[Klambaueret al. 2017](https://proceedings.neurips.cc/paper/2017/file/5d44ee6f2c3f71b73125876103c8f6c4-Paper.pdf)\];
- *inductive bias*: neural network weights are tuned using a hierarchical multitask learning \[[Caruana 1997](https://www.cs.cornell.edu/~caruana/mlj97.pdf); [Søgaardet al. 2016](https://aclanthology.org/P16-2038.pdf)\] with the temperature prediction as a high-level regression task and the coarsened temperature class recognition as a low-level classification task;
- *uncertainty*: a special loss function similar to [RMSEWithUncertainty](https://catboost.ai/en/docs/concepts/loss-functions-regression#RMSEWithUncertainty) is applied to training;
- *robustness*: a supervised contrastive learning based on N-pairs loss \[[Sohn2016](https://proceedings.neurips.cc/paper/2016/file/6b180037abbebea991d8b1232f8a8ca9-Paper.pdf)\] is applied instead of the crossentropy-based classification as a low-level task in the hierarchical multitask learning; it provides more robustness of the trainable neural network \[[Khosla et al. 2020](https://proceedings.neurips.cc/paper/2020/file/d89a66c7c80a29b1bdbab0f2a1a94af8-Paper.pdf)\]

4. A special technique of **deep ensemble** creation is implemented: it uses a model average approach like [bagging](https://en.wikipedia.org/wiki/Bootstrap_aggregating), but new training and validation sub-sets for corresponding ensemble items are generated using stratification based on coarsened temperature classes.

The deep ensemble size is 20. Hyper-parameters of each neural network in the ensemble (hidden layer size, number of hidden layers and alpha-dropout as special version of dropout in [SNN](https://proceedings.neurips.cc/paper/2017/file/5d44ee6f2c3f71b73125876103c8f6c4-Paper.pdf) are the same. They are selected using a hybrid approach: first automatically, and then manually. The automatic approach is based on a [Bayesian optimization with Gaussian Process regression](https://scikit-optimize.github.io/stable/auto_examples/bayesian-optimization.html), and it discovered the following hyper-parameter values for training the neural network in a single-task mode:

- hidden layer size is 512;
- number of hidden layers is 12;
- alpha-dropout is 0.0003.

After the implementation of the hierarchical multitask learning, the depth was manually increased up to 18 hidden layers: the low-level task (classification or supervised constrastive learning) was added after the 12th layer, and the high-level task (regression with uncertainty) was added after the 18th layer. General architecture of single neural network is shown on the following figure. All "dense" components are feed-forward layers with self-normalization. Alpha-dropout is not shown to oversimplify the figure. The `weather_snn_1_output` layer estimates the mean and the standard deviation of normal distribution, which is implemented using the`weather_snn_1_distribution` layer. Another output layer named as `weather_snn_1_projection` calculates low-dimensional projections for the supervised contrastive learning.


![][nn_structure]

[nn_structure]: images/deep_ensemble.png "Architecture of the deep neural network with hierarchical multitask learning."

Rectified ADAM \[[L. Liu et al. 2020](https://arxiv.org/pdf/1908.03265)\] with Lookahead \[[M. R. Zhang et al. 2019](https://proceedings.neurips.cc/paper/2019/file/90fd4f88f588ae64038134f1eeaa023f-Paper.pdf)\] is used for training with the following parameters: learning rate is 0.0003, synchronization period is 6, and slow weights step size is 0.5. You can see the more detailed description of other training hyper-parameters in the Jupyter notebook `deep_ensemble_with_uncertainty_and_spec_fe.ipynb`.

Experiments
-----------

Experiments were conducted according to data partitioning in \[[Malining et al. 2021](https://arxiv.org/pdf/2107.07455.pdf), section 3.1\]. Development and evaluation sets were not used for training and for hyper-parameter search. The quality of each modification of the method was first estimated on the development set, because all participants had access to the development set targets. After the preliminary estimation, the selected modification of the method was submitted to estimate R-AUC MSE on the evaluation set with targets concealed from participants.

In comparison with the baseline, the quality of the deep learning method is better (i.e. R-AUC MSE is less) on both datasets for testing:

- the development set (for preliminary testing):

1. **proposed deep ensemble = 1.015**;
2. baseline (CatBoost ensemble) = 1.227;

- the evaluation set (for final testing):

1. **proposed deep ensemble = 1.141**;
2. baseline (CatBoost ensemble) = 1.335.

Also, the results of the deep learning method are better with all possible values of the uncertainty threshold for retention.

![][error_retenction_curves]

[error_retenction_curves]: images/devset-results.png "rror retention curves on the development set."

The total number of all submitted methods at the evaluation phase is 73. Six selected results (top-5 results of all participants and the baseline result) are presented in the following table. The first three places are occupied by the following modifications of the proposed deep learning method:

- `SNN Ens U MT Np SpecFE` is the final solution with "all-inclusive";
- `SNN Ens U MT Np v2` excludes the feature quantization;
- `SNN Ens U MT` excludes the feature quantization too, and classification is used instead of supervised contrastive learning as the low-level task in the hierarchical multitask learning.

| Rank | Team           | Method                   | R-AUC MSE    |
| ---- | -------------- | ------------------------ | -----------: |
| 1    | bond005        | `SNN Ens U MT Np SpecFE` | 1.1406288012 |
| 2    | bond005        | `SNN Ens U MT Np v2`     | 1.1415403291 |
| 3    | bond005        | `SNN Ens U MT`           | 1.1533415892 |
| 4    | CabbeanWeather | `Steel box v2`           | 1.1575201873 |
| 5    | KDDI Research  | `more seed ens`          | 1.1593224114 |
| 55   | Shifts Team    | `Shifts Challenge`       | 1.3353865316 |
