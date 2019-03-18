import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(id='DynamicSetPacking-silly-v0', entry_point='gym_dynamic_set_packing.envs:SillyTestEnv')
register(id='DynamicSetPacking-adversarial-v0', entry_point='gym_dynamic_set_packing.envs:AdversarialEnv')
register(id='DynamicSetPacking-gurobitest-v0', entry_point='gym_dynamic_set_packing.envs:GurobiBinaryEnv')
