# Introduction

This will be an OpenAI Gym environment for dynamic set packing. The state will consist of a vector of counts of the elements of different types. Elements arrive and depart over time. In addition, a match can take place; when it does, all elements in the match are removed and the agent receives a reward equal to the cardinality of the match.


# 0/1 Action Space

In this situation, the goal of the agent is to decide whether to match or wait for more elements to arrive/depart. If it outputs "match", then a match takes place and it receives a reward. Otherwise, the state space evolves normally.

# Real-valued action space

In this situation, the agent's actions consist of a real-valued vector of the same size as the state space. Essentially, the vector is a weight for each type, specifying how strongly it wants that type to be matched. This vector is passed into the matcher.

# Matcher

We represent the state as a vector of types. We can think of each valid component of a match (e.g. each 2-cycle, 3-cycle, etc.) as a vector in the same space, which together can be a basis for representing a match.

The matcher solves some kind of maximal match linear program, possibly taking
in weights from the agent. Its output perhaps should not just be the match
itself in terms of counts of each type, but rather the match vector in the
basis described above. We will convert this to counts of types outside of the solver, allowing for things like simulating failures in specific cycles.

# Agents

## Match/don't match

Obviously the stupidest case is the agent that always matches. We can also take a Q-learning approach, with linear models or deep learning.

## Real-valued actions

Here, we basically have no choice but to do policy gradient, and just spit out an action.


