# Setup

It is recommended to use anaconda and define a new environment. In that environment,

    $ python setup.py develop # installs repo in develop mode and some deps

Also get a Gurobi license by [signing up for an academic account](http://www.gurobi.com/academia/for-universities) (you will need to be [on the CS VPN](https://helpdesk.cs.umd.edu/faq/connecting/vpn/) to do this) and then install Gurobi's python package by:

    $ conda config --add channels http://conda.anaconda.org/gurobi
    $ conda install gurobi

Finally you will probably want to install PyTorch if you haven't by following the instructions on pytorch.org for your system.

# Module structure

The main module just defines the environments, without reference to agents. So the main dependencies are OpenAI Gym, and currently Gurobi (to perform maximal matches).

## Environments

We have a general abstract class, `DynamicSetPackingBinaryEnv` (will have a different one for real-valued actions). It defines a step method. Each of the other methods called should be overridden by concrete classes to define the dynamics of the environment.

        def step(self, action):
            assert self.action_space.contains(action)

            reward = 0.0
            if action == 1:
                chosen_match = self._perform_match(self.state)
                reward = self._run_match(chosen_match)
    
    
            # elements arrive and depart
            self._arrive_and_depart()
    
            # do we need these dummy values?
            info = {'no info': 'no info'}
            done = False

            return self.state, reward, done, info
	    
Currently, two concrete environments are defined (OpenAI Gym requires that each environment have a concrete class that takes no constructor arguments; these are registered in the top-level `__init__.py`):

- `DynamicSetPacking-silly-v0`, class `envs.SillyTestEnv`:

	A toy environment.

- `DynamicSetPacking-gurobitest-v0`, class `envs.GurobiBinaryEnv`:
	
	An environment that has 16 types, and about 4000 feasible sets. This corresponds to donor-patient pairs whose compatibility is restricted only by blood type. Needs to be made more realistic (for example, better arrival/departure probabilities).

## Matchers

The `matchers` module defines classes that perform a maximal match. Currently just define a GurobiMatcher that knows the feasible sets, takes in a state and outputs a maximal match.

# Examples

Actual agents and training loops are defined separately inside the examples
directory. The only reason for this is to avoid having PyTorch, TensorFlow, or
other heavy libraries as dependencies of the environment itself.

For the currently working example, look at `policy_gradient_pytorch.ipynb`. 
