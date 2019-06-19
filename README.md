# ML Engine

Lets deploy a `Keras` model to the google cloud platform's (gcp) AI platform, and train it on some data.

What we will be looking at here:
 - Training a model locally
 - Deploying a model to the AI platform
 - Training this model on the AI platform
 - Get predictions for data from our trained model in the AI platform

Most of what we will be doing will happen trough a terminal.

## Clone repository

Lets start by `cloning` a repository containing a model which uses `Keras` to train on the United States Census data.

```bash
git clone REPO
```

## Install dependencies

We are here using `pipenv` to install our `python` dependencies.

### Installing `pipenv`

#### Mac and linux
See https://github.com/pypa/pipenv#installation

#### Windows
??

#### Using virtual env

```bash
pip install --user --upgrade virtualenv
```

```bash
virtualenv cmle-env
source cmle-env/bin/activate
```

### Installing dependencies with `pipenv`

When we have `pipenv` installed we can activate the `pipenv` environment and install the dependencies:
```bash
pipenv shell
pipenv install --dev
```

### Installing dependencies with `pip`

An alternative to using `pipenv` is using `pip` to install our dependencies
```bash
pip install -r requirements.txt
```

## Setup gcloud config

To be able to communicate with our gcp-project with the correct configurations, we can create a `gcloud` configurations setup.

```bash
# Change with what you want to call the configuration
GCLOUD_CONFIG_NAME="a_config_name"
# Change with the user you want to use to communicate with gcloud with (probably your .@computas.com mail)
EMAIL_ACCOUNT="e@mail.com"
# Change with the id of your gcp project
PROJECT_ID="gcp-project-id"

gcloud config configurations create "$GCLOUD_CONFIG_NAME"
gcloud config set account "$EMAIL_ACCOUNT"
gcloud config set project "$PROJECT_ID"
```

## Create bucket

Lets create a bucket in our gcp project where we will store our job(s)

The bucket name has to be unique across google cloud storage, but so does our gcp-project-id need to be, so lets use our project-id and some extra name, to name our bucket.
```bash
PROJECT_ID=$(gcloud config list project --format "value(core.project)")
BUCKET_NAME=${PROJECT_ID}-mlengine
```

Then we can set a region, which we want our bucket to live in. This one can not be reginal.
See here for options: https://cloud.google.com/ml-engine/docs/tensorflow/regions

`europe-west1` seems to have the options we would want, and is relatively close.
```bash
REGION=europe-west1
```

With this set up, we can create our bucket.
```bash
gsutil mb -l $REGION gs://$BUCKET_NAME
```

## Train model locally

Lets train a model locally

We specify a variable which will be the output directory of our local run.
```bash
MODEL_DIR=output
```

We run our training locally, like so:
```bash
gcloud ai-platform local train \
  --package-path trainer \
  --module-name trainer.task \
  --job-dir $MODEL_DIR
```

With `tensorboard` we can have a look at the progression of our training. We launch `tensorboard`.
```bash
tensorboard --logdir=$MODEL_DIR --port=8080
```

## Train model using AI platform

Lets train a model on the AI platform in gcp.

First, lets name our job, and set the directory we want it in the cloud.
```bash
JOB_NAME="my_keras_job"
JOB_DIR="gs://$BUCKET_NAME/keras-job-dir"
```

Then we submit our job to the cloud.
```bash
gcloud ai-platform jobs submit training $JOB_NAME \
  --package-path trainer/ \
  --module-name trainer.task \
  --region $REGION \
  --python-version 3.5 \
  --runtime-version 1.13 \
  --job-dir $JOB_DIR \
  --stream-logs
```

## Create model in AI platform

We now create a model, and a version for this model, in the AI platform.

Create the model
```bash
MODEL_NAME="my_keras_model"

gcloud ai-platform models create $MODEL_NAME \
  --regions $REGION
```

Create a model version
```bash
MODEL_VERSION="v1"
```

Get path to latest trained model in cloud
```bash
SAVED_MODEL_PATH=$(gsutil ls $JOB_DIR/keras_export | tail -n 1)
```

Create model version in cloud, could take a couple of minutes
```bash
gcloud ai-platform versions create $MODEL_VERSION \
  --model $MODEL_NAME \
  --runtime-version 1.13 \
  --python-version 3.5 \
  --framework tensorflow \
  --origin $SAVED_MODEL_PATH
```

## Prepare prediction input

We will now create a `prediction_input.json` which will be sent to our model in the cloud. We have some python scripts to help us with that, so let's dive into the `python` interpreter.

Like so:
```bash
python
```

We start by importing the `util` package from the `trainer` module.
```python
from trainer import util
```

We then load some data.
```python
_, _, eval_x, eval_y = util.load_data()
```

We then select `20` random rows from the loaded input data, like so:
```python
prediction_input = eval_x.sample(20)
```

And select the actual values for these rows, like so:
```python
prediction_targets = eval_y[prediction_input.index]
```

We can then have a look at the selected input data:
(As these are randomly selected, they will differ from selection to selection)
```python
prediction_input
```

As we can see from this, the number fields have been scaled to a [`z-score`](https://www.statisticshowto.datasciencecentral.com/probability-and-statistics/z-score/), and string values have been converted to int values. Some columns have also been dropped.
For more information on cleaning data see [cleaning data](https://developers.google.com/machine-learning/crash-course/representation/cleaning-data).

We can also have a look at the `prediction_targets`, which is the actual values each of these rows have. Like so:
```python
prediction_targets
```

To create the actual `prediction_input.json` which we want to predict we will create a json-file with newline-delimiters.

We start by importing the `json` package, and dump each row as a json-object:
```python
import json

with open('prediction_input.json', 'w') as json_file:
  for row in prediction_input.values.tolist():
    json.dump(row, json_file)
    json_file.write('\n')
```

We should now have `prediction_input.json`-file in our root of this package. This one will be sent to our model in the cloud, and the model will make prediction on these, and classify these as above 50K or below. So we can exit our python interpreter.
```python
exit()
```

## Submitting the prediction input

```bash
gcloud ai-platform predict \
  --model $MODEL_NAME \
  --version $MODEL_VERSION \
  --json-instances prediction_input.json
```

The response we get here is what the model's prediction for the `prediction_input.json`. Where 0 to 0.5 is a negative value (below 50k) and 0.5 to 1 is a positive value (above 50k).

How does this compare to the expected `prediction_values`?

## Wrap up

We have now:
  - Run a `keras` `tensorflow` model locally
  - Submitted this to model to the cloud
  - Created a model version in the cloud
  - Sent a set of `prediction_inputs` to our cloud-model, and gotten prediction values on these.

Forwards:
  - Create a custom model
  - Train on a different dataset
  - Run model locally
  - Deploy custom model to the AI platform
  - Get predictions from the custom model
  - Create an api to be able to send prediction inputs to
  - Use predictions to something fun
