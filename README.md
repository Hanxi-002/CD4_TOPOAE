# Run CD4
Go into src/datasets/CD4.py to change the path to access the w6_single.csv at data/CD4_Data/w6_single.csv. <br>
run the examply by using 
```bash
python -m exp.train_model -F test_runs with experiments/train_model/best_runs/CD4/TopoRegEdgeSymmetric.json device='cuda'   
```  

Pickle files will be automatically created. <br>
Current Error: 

Traceback (most recent calls WITHOUT Sacred internals):
  File "/ix/djishnu/Hanxi/CD4_TOPOAE/exp/train_model.py", line 142, in train
    training_loop()
  File "/ix/djishnu/Hanxi/CD4_TOPOAE/src/training.py", line 79, in __call__
    if self.on_epoch_begin(remove_self(locals())):
  File "/ix/djishnu/Hanxi/CD4_TOPOAE/src/training.py", line 42, in on_epoch_begin
    return self._execute_callbacks('on_epoch_begin', local_variables)
  File "/ix/djishnu/Hanxi/CD4_TOPOAE/src/training.py", line 37, in _execute_callbacks
    stop |= bool(getattr(callback, hook)(**local_variables))
  File "/ix/djishnu/Hanxi/CD4_TOPOAE/exp/callbacks.py", line 157, in on_epoch_begin
    losses = self._compute_average_losses(model)
  File "/ix/djishnu/Hanxi/CD4_TOPOAE/exp/callbacks.py", line 127, in _compute_average_losses
    loss, loss_components = model(data)
  File "/ihome/djishnu/xiaoh/mambaforge/envs/toyenv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1501, in _call_impl
    return forward_call(*args, **kwargs)
  File "/ix/djishnu/Hanxi/CD4_TOPOAE/src/models/approx_based.py", line 48, in forward
    latent = self.autoencoder.encode(x)
  File "/ix/djishnu/Hanxi/CD4_TOPOAE/src/models/submodules.py", line 561, in encode
    return self.encoder(x)
  File "/ihome/djishnu/xiaoh/mambaforge/envs/toyenv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1501, in _call_impl
    return forward_call(*args, **kwargs)
  File "/ihome/djishnu/xiaoh/mambaforge/envs/toyenv/lib/python3.10/site-packages/torch/nn/modules/container.py", line 217, in forward
    input = module(input)
  File "/ihome/djishnu/xiaoh/mambaforge/envs/toyenv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1501, in _call_impl
    return forward_call(*args, **kwargs)
  File "/ihome/djishnu/xiaoh/mambaforge/envs/toyenv/lib/python3.10/site-packages/torch/nn/modules/linear.py", line 114, in forward
    return F.linear(input, self.weight, self.bias)
RuntimeError: mat1 and mat2 shapes cannot be multiplied (50x1847 and 101x32)

# Topological Autoencoders

<img src="animations/topoae.gif" width="400"> <img src="animations/vanilla.gif" width="400">

## Reference

Please use the following BibTeX code to cite our [paper](https://arxiv.org/abs/1906.00722),
which is accepted for presentation at [ICML 2020](https://icml.cc/Conferences/2020):

```
@InProceedings{Moor20Topological,
  author        = {Moor, Michael and Horn, Max and Rieck, Bastian and Borgwardt, Karsten},
  title         = {Topological Autoencoders},
  year          = {2020},
  eprint        = {1906.00722},
  archiveprefix = {arXiv},
  primaryclass  = {cs.LG},
  booktitle     = {Proceedings of the 37th International Conference on Machine Learning~(ICML)},
  series        = {Proceedings of Machine Learning Research},
  publisher     = {PMLR},
  volume        = {119},
  editor        = {Hal Daumé III and Aarti Singh},
  pages         = {7045--7054},
  abstract      = {We propose a novel approach for preserving topological structures of the input space in latent representations of autoencoders. Using persistent homology, a technique from topological data analysis, we calculate topological signatures of both the input and latent space to derive a topological loss term. Under weak theoretical assumptions, we construct this loss in a differentiable manner, such that the encoding learns to retain multi-scale connectivity information. We show that our approach is theoretically well-founded and that it exhibits favourable latent representations on a synthetic manifold as well as on real-world image data sets, while preserving low reconstruction errors.},
  pdf           = {http://proceedings.mlr.press/v119/moor20a/moor20a.pdf},
  url           = {http://proceedings.mlr.press/v119/moor20a.html},
}
```  

## Setup
In order to reproduce the results indicated in the paper simply setup an
environment using the provided `Pipfile` and `pipenv` and run the experiments
using the provided makefile:

```bash
pipenv install 
```

Alternatively, the exact versions used in this project can be accessed in ```requirements.txt```, however
this pip freeze contains a superset of all necessary libraries. To install it, run
```bash
pipenv install -r requirements.txt 
```
  
## Running a method:
```bash
python -m exp.train_model -F test_runs with experiments/train_model/best_runs/Spheres/TopoRegEdgeSymmetric.json device='cuda'   
```   
We used device='cuda', alternatively, if no gpu is available, use device='cpu'.

The above command trains our proposed method on the Spheres Data set and writes logging, results and visualizations to `test_runs`. For different methods or datasets
simply adjust the last two directories of the path according to the directory structure.
If the dataset is comparatively small, (e.g. Spheres), you may want to visualize the latent space on the larger training split as well. For this, simply append 
``` evaluation.save_training_latents=True ``` at the end of the above command (position matters due to sacred).


## Calling makefile
The makefile automatically executes all experiments in the experiments folder
according to their highest level folder (e.g. experiments/train_model/xxx.json
calls exp.train_model with the config file experiments/train_model/xxx.json)
and writes the outputs to exp_runs/train_model/xxx/

For this use:
```bash
make filtered FILTER=train_model/repetitions
```
to run the test evaluations (repetitions) of the deep models
and for remaining baselines:
```bash
make filtered FILTER=fit_competitor/repetitions
```

We created testing repetitions by using the config from the best runs of the hyperparameter search (stored in best_runs/)


The models found in `train_model` correspond to neural network architectures.  

## Using Aleph (optional)

In the paper, low-dimensional persistent homology calculations are
implemented in Python directly. However, for higher dimensions, we
recommend to use Aleph, a C++ library. We aim to better integrate this
library into our code base, stay tuned!

Provided that all dependencies are satisfied, the following instructions should be sufficient
to install the module:

    $ git submodule update --init
    $ cd Aleph
    $ mkdir build
    $ cd build
    $ cmake ../
    $ make aleph
    $ cd ../../
    $ pipenv run install_aleph

