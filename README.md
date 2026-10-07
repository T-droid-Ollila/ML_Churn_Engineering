
<img width="1979" height="794" alt="Machine Learning Workflow Diagram" src="https://github.com/user-attachments/assets/a7d6a840-c2d0-481e-95cb-fe0587b56d04" />



<img width="1776" height="886" alt="Machine Learning Workflow Pipeline" src="https://github.com/user-attachments/assets/066d22d6-cb8d-4b23-831c-1a5c534513d7" />

Learn Ordinal Encoding , mapping , visualizing feature distributions , dealing with unbalanced datasets

Encountering problems along the way such as why is precision so consistently across three different models
but recall consistently is above 98% some models even hitting 100% , made me ask is this class imbalance or is my model
prioritising recall because in this churn scenario we would want our model to predict those who were most likely to churn
to help prioritise business numbers don't want to risk losing customers

Nested cross-validation (CV) is often used to train a model in which hyperparameters also need to be optimized . Nested CV estimates the generalization error of the underlying model and its (hyper)parameter search. Choosing the parameters that maximize non-nested CV biases the model to the dataset, yielding an overly-optimistic score.

Model selection without nested CV uses the same data to tune model parameters and evaluate model performance. Information may thus “leak” into the model and overfit the data. The magnitude of this effect is primarily dependent on the size of the dataset and the stability of the model.

Do I use RandomSearchCV with cross_validate ?

MY ANSWER : When implemeting NestedCV

Nested == LOOP BASED

Nested CV estimates the generalization error of the underlying model and its (hyper)parameter search. Choosing the parameters that maximize non-nested CV biases the model to the dataset, yielding an overly-optimistic score.

A variable param_opt which is a dict for optimizing hyperparameter

<img width="1600" height="900" alt="Docker_converted" src="https://github.com/user-attachments/assets/c5afe8f5-3425-48f5-be3b-75fa46626e8f" />

RandomSearch allows much more hyperparameter values , and faster searching tus saving on time

What is the purpose of n_iter in RandomSearch module ?

DVC does provide an option to specify the template/type of visualisation we need but for we will need a template right now and we shall use the ones specified in our visualizations functions in our visualizations script

<img width="1919" height="820" alt="Setting Up Local Remote Storage" src="https://github.com/user-attachments/assets/57848c07-bc17-495b-9d29-bf7d8d2b84ea" />


<img width="1797" height="875" alt="MLOps CI_CD Continual Learning Workflow" src="https://github.com/user-attachments/assets/3a207954-0973-44ec-8cd1-4cc1ace47be3" />


