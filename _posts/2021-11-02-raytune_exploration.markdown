---
layout: post
title:  "Exploring Ray Tune for Pytorch Hyperparameter Optimization"
categories: jekyll update
---
I tried out Ray Tune based on the tutorial found on the [PyTorch.org tutorials](https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html). Setting up was not straightforward, as I ran into some issues, but I quickly resolved them I will detail my final implementation and then explain some of the initial setup issues.

## Problems encountered during setup
From what I can remember, I encountered two primary issues.
1. Radis errors of unkown origin.
2. Numpy errors when calling tune.report
3. Idle Processes lying around

# 1 Radis Errors
From what I could tell, I had followed the tutorial exactly. However, the script was crashing as soon as the dataset was loaded in with the following error.

{% highlight bash %}
2021-11-02 21:01:38,924 WARNING experiment.py:302 -- No name detected on trainable. Using DEFAULT.
2021-11-02 21:01:38,924 INFO registry.py:66 -- Detected unknown callable for trainable. Converting to class.
Traceback (most recent call last):
  File "/home/everett/.local/lib/python3.8/site-packages/redis/connection.py", line 706, in send_packed_command
    sendall(self._sock, item)
  File "/home/everett/.local/lib/python3.8/site-packages/redis/_compat.py", line 9, in sendall
    return sock.sendall(*args, **kwargs)
BrokenPipeError: [Errno 32] Broken pipe

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "main.py", line 10, in <module>
    main()
  File "main.py", line 6, in main
    run_experiment(net_params, experiment_params, train_cifar)
  File "/home/everett/G/CMU/8-F21/10715/Homework/hw6/raytune.py", line 33, in run_experiment
    results = tune.run(
  File "/home/everett/.local/lib/python3.8/site-packages/ray/tune/tune.py", line 406, in run
    experiments[i] = Experiment(
  File "/home/everett/.local/lib/python3.8/site-packages/ray/tune/experiment.py", line 159, in __init__
    self._run_identifier = Experiment.register_if_needed(run)
  File "/home/everett/.local/lib/python3.8/site-packages/ray/tune/experiment.py", line 305, in register_if_needed
    register_trainable(name, run_object)
  File "/home/everett/.local/lib/python3.8/site-packages/ray/tune/registry.py", line 73, in register_trainable
    _global_registry.register(TRAINABLE_CLASS, name, trainable)
  File "/home/everett/.local/lib/python3.8/site-packages/ray/tune/registry.py", line 147, in register
    self.flush_values()
  File "/home/everett/.local/lib/python3.8/site-packages/ray/tune/registry.py", line 169, in flush_values
    _internal_kv_put(_make_key(category, key), value, overwrite=True)
  File "/home/everett/.local/lib/python3.8/site-packages/ray/_private/client_mode_hook.py", line 89, in wrapper
    return func(*args, **kwargs)
  File "/home/everett/.local/lib/python3.8/site-packages/ray/experimental/internal_kv.py", line 86, in _internal_kv_put
    updated = ray.worker.global_worker.redis_client.hset(
  File "/home/everett/.local/lib/python3.8/site-packages/redis/client.py", line 3050, in hset
    return self.execute_command('HSET', name, *items)
  File "/home/everett/.local/lib/python3.8/site-packages/redis/client.py", line 900, in execute_command
    conn.send_command(*args)
  File "/home/everett/.local/lib/python3.8/site-packages/redis/connection.py", line 725, in send_command
    self.send_packed_command(self.pack_command(*args),
  File "/home/everett/.local/lib/python3.8/site-packages/redis/connection.py", line 717, in send_packed_command
    raise ConnectionError("Error %s while writing to socket. %s." %
redis.exceptions.ConnectionError: Error 32 while writing to socket. Broken pipe.
{% endhighlight %}

Since I have absolutely zero experience with radis, and honestly I'm not sure what a socket or a pipe are, I was a little bewildered. I felt like I had followed the tutorial exactly, but it just didn't work at all. Thankfully, I quickly ran into [This Glorious Thread](https://github.com/ray-project/ray/issues/12157) which pointed to the [User Guide on Handling Large Datasets](https://docs.ray.io/en/latest/tune/user-guide.html#handling-large-datasets). It seemed odd, since CIFAR10 is hardly a large dataset. However, it worked flawlessly. I'm still a bit confused why. All I had to do was change

{% highlight python %}
results = tune.run(
    partial(train_func, data=data, saveFilePath='./models/'+experiment_name, checkpoint_dir='./checkpints'),
    resources_per_trial={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
    config=net_params,
    num_samples=num_samples,
    scheduler=scheduler,
    progress_reporter = reporter,
    local_dir="./tune_results",
    name=experiment_name)
{% endhighlight %}

to

{% highlight python %}
results = tune.run(
    tune.with_parameters(train_func, data=data, saveFilePath='./models/'+experiment_name, checkpoint_dir='./checkpints'),
    resources_per_trial={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
    config=net_params,
    num_samples=num_samples,
    scheduler=scheduler,
    progress_reporter = reporter,
    local_dir="./tune_results",
    name=experiment_name)
{% endhighlight %}

# 2 Numpy Issues
The most time consuming issue that I encountered while setting up Ray Tune was a strange Numpy issue. It appeared that the values being bassed to tune.report during my trianing loop were incompatible with numpy. After a while of searching, made some changes until I finally added ".item()" to the metrics that I was passing into tune.report. This appeared to fix the issue. However, I was completely unable to reproduce the issue while writing this article. I guess I'll never know what was really going wrong.

# Idle processes persisting
There were several times where Ray Tune crashed. When Ray Tune doesn't end properly, it tends to leave a ton of idle processes lying around. To address these quickly, the following command was rather useful

{% highlight bash %}
kill -9 $(ps aux | grep "[r]ay" | awk '{print $2}')
{% endhighlight %}


# Final Working Setup

At the top level, in `main.py` we just do two things:
1. pick a test
2. run the experiement

{% highlight python %}
from raytune import *
from tests import *

def main(): 
    net_params, experiment_params = pool_test
    run_experiment(net_params, experiment_params, train_cifar)

    print("done")
{% endhighlight %}

To keep everything orginized, I set up a file `tests.py` that holds each experiment. I wish python were explicitly typed, so I could typedef the experiment datastructure. An experiement is a tuple of (net_parameters, experiment_parameters), where each parameter set is a dictionar of values. This keeps everything nice and orginized and unsures I have a catalouge of tests to choose from. Here is an example:

{% highlight python %}
lr_small_test = ({'model': 'lenet',
        'display_step': 250, 
        'batch_size': 256,
        'iterations': 6_000, # 3000 or 500 
        'initial_lr': tune.loguniform(0.2,0.01),
        'lr_decay': 0.8,
        'adjust_lr_step': 500,
        'initial_momentum': 0.9,
        'final_momentum': 0.95,
        'momentum_change_steps': 2_500,
        'adjust_momentum_step': 1_500,
        'apply_weight_norm': True,
        'weight_norm': 3.5,
        'adjust_norm_step': 1_000,
        'output_l2_decay': 0.001,
        'pooling': 'max', # TODO try three options: 'max', 'avg','no' 
        'activation':'relu',
        'c1_out': 6,
        'c2_out': 16,
        'l1_out': 120,
        'l2_out': 84,
        'store_trajectories' : False,
        'ray_tune' : True,
        'random_seed': 0},

        {'gpus_per_trial'  : 1,
        'cpus_per_trial' : 16,
        'max_num_epochs': 16,
        'grace_period': 16,
        'num_samples': 100,
        'experiment_name': 'lr_small_test'}
        )
{% endhighlight %}

The run_experiment function, in `raytune.py`, pretty much follows the instructions from the [PyTorch Ray Tune Tutorial](https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html). Right now there is a lot of room for improvement. I should definitely make this function more general, so that I can be supplied with an arbitrary dataset and model. 

ASHAScheduler also seems fairly simple, and it seems a silly choice since I'm only using one GPU. However, it seemed to work fine for my given purposes.

{% highlight python %}
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

`train.py` contains the helper function that initializes each `LeNet` instance and calls the `.fit` method.

{% highlight python %}
from lenet import * 
from dataloaders import *

def train_cifar(params, data=None, saveFilePath=None, checkpoint_dir=None):
    clf = LeNet(params)

    device = t.device('cuda' if t.cuda.is_available() else 'cpu')

    if t.cuda.is_available():
        device = "cuda:0"
        if t.cuda.device_count() > 1:
            clf = nn.DataParallel(clf)

    clf.fit(data, checkpoint_dir)
{% endhighlight %}

Lastly, we need to communicate the performance of the model from the training loop. Adding this quick snippet to my `LeNet` class fit method worked perfectly. It spends a lot of time saving the model and calculating metrics, so I set it to run only once every 4 epochs.

{% highlight python %}
#raytune stuff for hyperparam optimization
if (epoch % 4 == 0 and params['ray_tune']):
    with tune.checkpoint_dir(epoch) as checkpoint_dir:
        path = os.path.join(checkpoint_dir, "checkpoint")
        self.save(path)

    out_cross_entropy  = self.evaluate_cross_entropy(outsample_dataloader).item()
    out_accuracy       = self.evaluate_accuracy(outsample_dataloader).item()

    tune.report(loss= out_cross_entropy, accuracy=out_accuracy)
{% endhighlight %}

# Experiments

Experiment results can be visualized easily using `tensorboard --logdir ./tune_results/pool_test`.

Here is an example of a learning rate test that was run.

![Learning Rate Results TensorBoard Plot](/assets/images/tensorboard_lr_test.png)

# Limitations

So far, the most annoying limitation is that Ray Tune seems to run identical instances of tests. while trying to run experiements on the type of pooling in the LeNet, when I ran 3 experiments, it would often run two instances of 'max', one instance of 'avg' and zero instances of 'no'. I need to figure out how to ensure each trail is unique. This isn't a problem when testing from non-discreet distributions.

