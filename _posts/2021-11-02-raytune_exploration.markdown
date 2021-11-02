---
layout: post
title:  "Exploring RayTune for the first time"
categories: jekyll update
---
I tried out Raytune. It was satisfying, but I was shocked by how ineffective it was at improving my results.

$x^2 + 2x + 1 = 0$

{% highlight ruby %}
from lenet import * 
from dataloaders import *
from train import *
from tests import *
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from functools import partial


def run_experiment(net_params, experiment_params, train_func):
    
    gpus_per_trial = experiment_params['gpus_per_trial']
    cpus_per_trial = experiment_params['cpus_per_trial']
    max_num_epochs = experiment_params['max_num_epochs']
    grace_period = experiment_params['grace_period']
    num_samples = experiment_params['num_samples']
    experiment_name = experiment_params['experiment_name']

    scheduler = ASHAScheduler(
        metric = "accuracy",
        mode = "max",
        max_t = max_num_epochs,
        grace_period=grace_period,
        reduction_factor=2)

    reporter = CLIReporter(
        metric_columns=["loss", "accuracy", "training_iteration"])


    data = load_data(net_params)

    results = tune.run(
        tune.with_parameters(train_func, data=data, saveFilePath='./models/'+experiment_name, checkpoint_dir='./checkpints'),
        resources_per_trial={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
        config=net_params,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter = reporter,
        local_dir="./tune_results",
        name=experiment_name
    )

    best_trial = results.get_best_trial("accuracy", "max", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))
{% endhighlight %}

Check out the [Jekyll docs][jekyll-docs] for more info on how to get the most out of Jekyll. File all bugs/feature requests at [Jekyll’s GitHub repo][jekyll-gh]. If you have questions, you can ask them on [Jekyll Talk][jekyll-talk].

[jekyll-docs]: https://jekyllrb.com/docs/home
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-talk]: https://talk.jekyllrb.com/
