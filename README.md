



Learn Ordinal Encoding , mapping , visualizing feature distributions , dealing with unbalanced datasets

Encountering problems along the way such as why is precision so consistently across three different models
but recall consistently is above 98% some models even hitting 100% , made me ask is this class imbalance or is my model
prioritising recall because in this churn scenario we would want our model to predict those who were most likely to churn
to help prioritise business numbers don't want to risk losing customers

Allow yourself to build things along the way to enable ou encouunter things like Random Grid Searvh for optimal searching for hyperparameters

Nested cross-validation (CV) is often used to train a model in which hyperparameters also need to be optimized . Nested CV estimates the generalization error of the underlying model and its (hyper)parameter search. Choosing the parameters that maximize non-nested CV biases the model to the dataset, yielding an overly-optimistic score.

Model selection without nested CV uses the same data to tune model parameters and evaluate model performance. Information may thus “leak” into the model and overfit the data. The magnitude of this effect is primarily dependent on the size of the dataset and the stability of the model.

Do I use RandomSearchCV with cross_validate ?

MY ANSWER : When implemeting NestedCV

Nested == LOOP BASED

Nested CV estimates the generalization error of the underlying model and its (hyper)parameter search. Choosing the parameters that maximize non-nested CV biases the model to the dataset, yielding an overly-optimistic score.

A variable param_opt which is a dict for optimizing hyperparameter

RandomSearch allows much more hyperparameter values , and faster searching tus saving on time

Get familiar with scipy.stats module and np.linspace and random and randit

What is the purpose of n_iter in RandomSearch module ?

Ask yourself mening of a estimator


DVC does provide an option to specify the template/type of visualisation we need but for learning purposes we will not need a template right now , we shall use the ones speficied in our visualizations functions in our visualizations script
