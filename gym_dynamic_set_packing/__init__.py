import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(id='DynamicSetPacking-v0', entry_point='gym_dynamic_set_packing.envs:DynamicSetPackingEnv')
