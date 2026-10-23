# GITFOLIO :⚡ ML ENGINEERING

**Support:**
<br>
Star 🌟 the Project: ![GitHub stars](https://img.shields.io/github/stars/T-droid-Ollila/ML_Churn_Engineering.svg?style=social&label=Star)
<br>
Follow 🤝 on GitHub: ![GitHub followers](https://img.shields.io/github/followers/T-droid-Ollila.svg?style=social&label=Follow)
<br>

                                                      ` Design · Develop · Deploy · Iterate `
---
## Lessons
---
Learn how to combine machine learning with software engineering to design, develop, deploy and iterate on production-grade ML applications.

<img width="1979" height="794" alt="Machine Learning Workflow Diagram" src="https://github.com/user-attachments/assets/a7d6a840-c2d0-481e-95cb-fe0587b56d04" />

---
<h2>🧰 MLOps Tools Covered </h2>

  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/pandas/pandas-original.svg" width="45" alt="Pandas"/> <img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/scikitlearn/scikitlearn-original.svg" width="45" alt="Scikit-learn"/>
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/docker/docker-original.svg" width="45" alt="Docker"/>
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/git/git-original.svg" width="45" alt="Git"/>
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/github/github-original.svg" width="45" alt="GitHub"/>
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/githubactions/githubactions-original.svg" width="45" alt="GitHub Actions"/>
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/pytest/pytest-original.svg" width="45" alt="pytest"/>
</p>

<p align="left">
  <img src="https://img.shields.io/badge/MLflow-0194E2?style=flat-square" alt="MLflow"/>
  <img src="https://img.shields.io/badge/DVC-945DD6?style=flat-square" alt="DVC"/>
  <img src="https://img.shields.io/badge/Ray-028CF0?style=flat-square" alt="Ray"/>
  <img src="https://img.shields.io/badge/XGBoost-EC6B23?style=flat-square" alt="XGBoost"/>
  <img src="https://img.shields.io/badge/Great_Expectations-FF6310?style=flat-square" alt="Great Expectations"/>
</p>

---

## Setup

In this project we shall use [RAY](https://www.anyscale.com/)
Your personal laptop (single machine) will act as the cluster, where one CPU will be the head node and some of the remaining CPU will be the worker nodes.

### **1. Fork and clone**
- Create a repository by following these instructions: [Create a new repository](https://github.com/new) → name it Preferred-Name → Toggle Add a README file (very important as this creates a main branch    Click Create repository (scroll down)

- Execute the command below in your terminal ( Command prompt or shell )

```
cd Preferred-Name
git clone https://T-droid-Ollila/ML_Churn_Engineering.git.

```

Every time a commit is made and you want to push changes made to your created repo , first set up a remote repo of your choice

```
git remote set-url origin https://github.com/<their-user>/<Preferred-Name>.git
git push -u origin main

```

### **2. Virtual environment**

```
$env:PYTHONPATH = "$env:PYTHONPATH;$pwd     # MacOs export PYTHONPATH=$PYTHONPATH:$PWD
py 3.12 -m venv virtual  # recommend using Python 3.12  # For MacOS python3.12 -m venv .virtual
virtual\Scripts\activate  # On MacOS source virtual/bin/activate
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -r requirements.txt
pre-commit = 3.8.0 install
pre-commit autoupdate
pip list

# Then retrieve the versioned data: On first use, authenticate with your own Google account.
# Your Google account must first be granted access to the project's
# Google Drive DVC folder.

To request access, contact the repository owner and provide the
Google account that should be authorized.

dvc pull
```
### Setting Up Google Drive Remote Storage and retrieve data from remote storage

<img width="1536" height="1024" alt="cff8f784-eaa1-48c9-a2c8-8b2b02280a79" src="https://github.com/user-attachments/assets/d141f94c-9b6b-4c96-91d4-1a339e3ef264" />

To configure your local DVC remote storage using Google Drive, follow these steps to ensure secure authentication and proper synchronization

**Prerequisites:**
*   A created Google Drive folder with [google drive account](https://drive.google.com/)
*   A project in your [google cloud console](https://console.cloud.google.com/).
*   Client ID and secret obtained from your respective project in the console .

FolderID refers to the ID at the top of the search tab after creating a folder or when you entered into one

Typically it will look like this

**Google Drive folder URL:** `https://drive.google.com/drive/folders/15W2JmKpx8IdwoLEfotwNwupL7Cawx52dLe`

**Google Drive Folder ID:**

`15W2JmKpx8IdwoLEfotwNwupL7Cawx52dLe`

<img width="1919" height="820" alt="Setting Up Local Remote Storage" src="https://github.com/user-attachments/assets/57848c07-bc17-495b-9d29-bf7d8d2b84ea" />

- Ensure project setup has **credentials** and under **API and Services** are both created and enabled.
- select OAuth client ID. Choose **Desktop app** and click Create to proceed with a default client name.
-
.json file is file created after creating a **credentials** and accessing **ID** and **secret**

```
dvc add remote -d gdrive_remote gdrive://folderID

# Use the commands to set the credentials
dvc remote modify --local myremote gdrive_client_id 'client-id'
dvc remote modify --local myremote gdrive_client_secret 'client-secret'

# Then :
dvc push
```
- For more info on dvc remote storage setup with [Google drive](https://doc.dvc.org/user-guide/data-management/remote-storage/google-drive) and other remote stoarages see the [docs](https://doc.dvc.org/user-guide/data-management/remote-storage)

### Scripts
```
churn_src
├── config.py
├── data.py
├── evaluate.py
├── models.py
├── predict.py
├── serve.py
├── train.py
├── tune.py
└── utils.py
```
### Setup experiment tracking
- We'll use [MLflow](https://mlflow.org/) to track our experiments and store our models and the MLflow Tracking UI to view our experiments. We have been saving our experiments to a local directory but note that in an actual production setting, we would have a central location to store all of our experiments. It's easy/inexpensive to spin up your own MLflow server for all of your team members to track their experiments on

<img width="1776" height="886" alt="Machine Learning Workflow Pipeline" src="https://github.com/user-attachments/assets/066d22d6-cb8d-4b23-831c-1a5c534513d7" />

### Training



### Evaluation

```

def promote_model_to_production(
    client: MlflowClient, run_id: str, model_path: str, model_name: str, alias: str
):
    """Promote a model to production by registering it and setting an alias."""

    model_uri = f"runs:/{run_id}/{model_path}"

    with mlflow.start_run(run_id=run_id):  # start_run, not the nonexistent mlflow.run
        model_version = mlflow.register_model(model_uri=model_uri, name=model_name)

        # model_info.registered_model_version gives the REAL version, not a hardcoded "1"
        client.set_registered_model_alias(
            name=model_name, alias=alias, version=model_version.version
        )

        mlflow.end_run()

model = mlflow.sklearn.load_model("models:/production_tree_model@production")

```
### Predict

```
$env:MODEL = 'artifacts\prod_model\model.pkl'
$env:PROCESSOR = 'artifacts/processed/preprocessor.pkl'
$env:DATASET = "C:\Data\messy_churn_10000.csv"

python -m pytest --dataset-loc="$env:DATASET" tests/data --verbose --disable-warnings

python churn_src\predict.py $env:PROCESSOR $env:DATASET

```

## Production

- **Production environments are anything but smooth because code that worked in development environment no longer work in production but thanks to conayiner we can be able to ***simulate*** the same environment to ensure our codes works as expected.**

- **The workflow that we shall implement is like so:**

<img width="1055" height="896" alt="Deployment   Monitoring Workflow" src="https://github.com/user-attachments/assets/71c42b3b-e690-45b5-a810-8f531fac895f" />

### Docker

- **We shall be deploying this model locally as a containerised job.**

 <img width="2057" height="764" alt="Docker Pipeline_ File to Container" src="https://github.com/user-attachments/assets/23cd69ad-7822-4615-bb31-32ad4939811b" />

```
 /app/
├── churn_src/
│   └── predict.py
│
├── artifacts/
│   ├── model.pkl
│   └── preprocessor.pkl
│
├── data/
│   └── incoming.csv
│
└── predictions_output/

```
- **The Dockerfile provides blue print of how we want the structure of our project file to look like in the containerised environment**

```
docker build -t Churn . # The first command looks at the blue print from the dockerfile to build our custom image

docker run --rm `
--mount "type=bind,source=$($env:DATASET),target=/app/incoming.csv,readonly" `
--mount "type=bind,source=$($PWD.Path)/predictions_output,target=/app/predictions_output" `
first `
python churn_src/predict.py `
/app/artifacts/processed/preprocessor.pkl `
/app/incoming.csv `
/app/artifacts/prod_model/model.pkl

```
- **The mounts represent the data and mounted volume that we are going to use to predict and store our predicted results respectively.Mount is used when your container requires a particular file but was not build alongside your container.For me I considered it computationally efficient to put them separately.**

### CI\CD WITH GITHUB ACTIONS

- **We're not going to manually deploy our application every time we make a change. Instead, we'll automate this process using GitHub Actions!**

- **In short, Docker + CI/CD = streamlined DevOps magic. ✨**

<img width="568" height="555" alt="Deployment   Monitoring Workflow" src="https://github.com/user-attachments/assets/de76e1f2-ed5d-4a9a-910a-6d31be34cca2" />

### Continual Learning.

- **With our CI/CD workflow in place to deploy our application, we can now focus on continually improving our model. It becomes really easy to extend on this foundation to connect to scheduled runs (cron), data pipelines, drift detected through monitoring, online evaluation, etc. And we can easily add additional context such as comparing any experiment with what's currently in production (directly in the PR even), etc.**

<img width="1797" height="875" alt="MLOps CI_CD Continual Learning Workflow" src="https://github.com/user-attachments/assets/3a207954-0973-44ec-8cd1-4cc1ace47be3" />

### Extra Thoughts

- **Encountering problems along the way such as why is precision so consistently across three different models but recall consistently is above 98% some models even hitting 100% , made me ask is this class imbalance or is my model prioritising recall because in this churn scenario we would want our model to predict those who were most likely to churn to help prioritise business numbers don't want to risk losing customers.**

- **Nested cross-validation (CV) is often used to train a model in which hyperparameters also need to be optimized . Nested CV estimates the generalization error of the underlying model and its (hyper)parameter search. Choosing the parameters that maximize non-nested CV biases the model to the dataset, yielding an overly-optimistic score.**

- **Model selection without nested CV uses the same data to tune model parameters and evaluate model performance. Information may thus “leak” into the model and overfit the data. The magnitude of this effect is primarily dependent on the size of the dataset and the stability of the model.**

- **Do I use Random Search CV with cross_validate ? MY ANSWER : When implementing Nested CV , Nested == LOOP BASED , Nested CV estimates the generalization error of the underlying model and its (hyper)parameter search. Choosing the parameters that maximize non-nested CV biases the model to the dataset, yielding an overly-optimistic score.**


 [![Anurag's GitHub stats](https://github-stats-extended.vercel.app/api/?username=T-droid-Ollila&show_icons=true&theme=calm&rank_icon=github&include_all_commits=true&custom_title=Ollila's+Stats&disable_animations=true&number_format=long&show=prs_merged_percentage,prs_reviewed)](https://github-stats-extended.vercel.app/api/?username=T-droid-Ollila&show_icons=true&theme=calm&rank_icon=github&include_all_commits=true&custom_title=Ollila's+Stats&disable_animations=true&number_format=long&show=prs_merged_percentage,prs_reviewed)

---
<h2>🌐 Connect With Me</h2>

<p align="left">
  <a href="https://www.linkedin.com/in/ollila-simiyu-a9b886363/?isSelfProfile=true">
    <img src="https://img.shields.io/badge/LINKEDIN-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn"/>
  </a>

  <a href="ollilasimiyu7@gmail.com">
    <img src="https://img.shields.io/badge/EMAIL-C62828?style=for-the-badge&logo=gmail&logoColor=white" alt="Email"/>
  </a>

  <a href="https://github.com/T-droid-Ollila">
    <img src="https://img.shields.io/badge/GITHUB-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub"/>
  </a>
</p>

---
