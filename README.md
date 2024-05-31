# Deep forests with tree-embeddings and label imputation for weak-label scenarios
*International Joint Conference on Neural Networks 2024*

**Pedro Ilidio<sup>1</sup>, Ricardo Cerri<sup>2</sup>, Celine Vens<sup>3,4</sup>, Felipe Kenji Nakano<sup>3,4</sup>**


<sup>1</sup> Instituto de Física de São Carlos, Universidade de São Paulo, São Carlos, São Paulo, Brazil

<sup>2</sup> Instituto de Ciências Matemáticas e de Computação, Universidade de São Paulo, São Carlos, SP, Brazil

<sup>3</sup> KU Leuven, Campus KULAK, Dept. of Public Health and Primary Care, Kortrijk, Belgium

<sup>4</sup> Itec, imec research group at KU Leuven, Kortrijk, Belgium

Email: pedro.ilidio@kuleuven.be, cerri@icmc.usp.br, celine.vens@kuleuven.be and felipekenji.nakano@kuleuven.be

---

## Repository description

This repository contains the code necessary for reproducing the results of the paper "Deep forests with tree-embeddings and label imputation for weak-label scenarios", presented at the International Joint Conference on Neural Networks 2024. In the study, we explore new strategies for dealing with weak-label scenarios by adapting deep forests algorithms. We combine tree-embeddings with two newly proposed label imputation techniques, and demonstrate superior or competitive performance to the state-of-the-art.

The `experiments` folder contains the code for running the experiments and generating the results presented in the paper.
Please see the file `experiments/MLproject` for more information and examples on how to run the experiments.
For more information on MLflow Projects, please refer to the [official documentation](https://mlflow.org/docs/latest/projects.html).

The `data` folder contains the datasets used in the experiments. The datasets were directly obtained from [Nakano *et al*. (2021)](https://itec.kuleuven-kulak.be/pattern-recognition-2021/). More information about them is provided by the [original source](http://www.uco.es/kdis/mllresources/).

The `paper` folder contains the LaTeX source code for generating the camera-ready version of the paper.

Please feel free to contact Pedro Ilídio at the following email addresses in case you have any questions or need further information:
```
pedro.ilidio@kuleuven.be
```
```
ilidio@alumni.usp.br
```
```
pcbilidio@gmail.com
```

## Paper abstract
Due to recent technological advances, a massive amount of data is generated on a daily basis. Unfortunately, this is not always beneficial as such data may present weak-supervision, meaning that the output space can be incomplete, inexact, and inaccurate. This kind of problems is investigated in weakly-supervised learning. In this work, we explore weak-label learning, a structured output prediction task for weakly-supervised problems where positive annotations are reliable, whereas negatives are missing.
For the first time in this class of problems, we investigate deep forest algorithms based on tree-embeddings, a recently proposed feature representation strategy leveraging the structure of decision trees. 
Furthermore, we propose two new procedures for label-imputation in each layer, named Strict Label Complement (SLC), which provides fixed conservative estimates for the number of missing labels and employs them to restrict imputations, and Fluid Label Addition (FLA), which performs such estimations on every layer and uses them to adjust the imputer's predicted probabilities without any restrictions.
We combine the new approaches with deep forest architectures to produce four new algorithms: SLCForest and FLAForest, using output space feature augmentation, and also the cascade forest embedders CaFE-SLC and CaFE-FLA, employing both tree-embeddings and the output space.
Our results reveal that our methods provide superior or competitive performance to the state-of-the-art. Furthermore, we also noticed that our methods are associated with better results even in cases without weak-supervision.