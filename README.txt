This repository includes the code and data presented in the paper, "Confronting Domain Shift in Trained Neural Networks", Martinez, C., Najera-Flores, D. A., Brink, A. R., Quinn, D. D., Chatzi, E., & Forrest, S. (2021, July). Confronting Domain Shift in Trained Neural Networks. In NeurIPS 2020 Workshop on Pre-registration in Machine Learning (pp. 176-192). PMLR.

The code was tested with Python 3.6.8 and dependencies (found in requirements.txt) were installed with pip 21.0.1.

Model and data configurations should be set with config files in .json format.  An example config file is found here: configs/example_train.json

To train a model:

python train.py configs/example_train.json

To run inference:

python infer.py configs/example_infer.json

Data used in the experiments for the paper are located in the ./data directory.


SAND2021-15138O

