# Improve generalisation of a QML model by constraining the data scaling parameter

##Objectives
This project is an investigation into the effect of different methods of data scaling in quantum machine learning models. 
The objectives are:
- O1: Code a simple QML model with 1 dimensional data input and show that unconstrained optimisation of a data scaling 
parameter leads to overfitting. 
- O2: Develop a method to scale the data that does not lead to overfitting, and improves generalisation
 error with respect to models with fixed scaling.
- O3: Generalise the method in O2 to input data of arbitrary dimension and investigate performance on some data sets.
- O4: (if time) Observe a [double descent curve](https://medium.com/mlearning-ai/double-descent-8f92dfdc442f) in a 
simple QML model, and show that this phenomenon relies on a good choice of data scaling. 

##Some intial ideas
There are a number of studies in classical machine learning that investigate generalisation in 'fourier models', 
which share the same basic mathematical structure as QML models. We might be able to get some ideas from these
 papers, although I have not found anything that adresses the scaling of data explicity.
- [Two models of double descent for weak features](https://arxiv.org/pdf/1903.07571.pdf) <br />
Here they show a double descent model in a fourier model. 
- [Occam's Razor](https://proceedings.neurips.cc/paper/2000/file/0950ca92a4dcf426067cfd2246bb5ff3-Paper.pdf) <br />
Something about scaling in a fourier model in bayesian setting, not sure if that relevant. 

Another thing we could look at is somehow using fourier analysis to judiciously chose the data scaling, so that there
is a large overlap between the frequencies avaliable in the quantum
model and the highly weighted frequencies in the data. This will probably involve
estimating the discrete fourier transform of the data. The issue here is that the
data is generally not evenly spaced and there will be missing data, so we need techniques to
estimate the DFT in this setting. I imagine there is a lot of literature that
could help us here, see e.g. [here](https://proceedings.neurips.cc/paper/2003/file/8c1b6fa97c4288a4514365198566c6fa-Paper.pdf). 




## Dataset
Our dataset (`data/custom_dataset.txt`) consists of 1D noisy points sampled from the following function: 

```python
np.sin((2 * np.pi / 10) * x - np.pi)
```

<img width="1475" alt="image" src="https://user-images.githubusercontent.com/49232747/191979312-4c32708c-c23e-47fe-9dd5-7ef9977138d4.png">

## Dummy code
All the testing code that is not relevant to the project can be moved inside a `dummy/` folder, and git will ignore it.
