
This is an OpenAI Gym environment for dynamic set packing. 

# Environments

We have a general abstract class, `DynamicSetPackingBinaryEnv` (will have a different one for real-valued actions). It defines a step method. Each of the functions called should be overridden by concrete classes, to define the behavior.

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

# Matchers

The `matchers` module defines classes that perform a maximal match. Currently just define a GurobiMatcher that knows the feasible sets, takes in a state and outputs a maximal match.
