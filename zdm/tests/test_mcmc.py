from zdm import mcmc
from zdm import io
from zdm import iteration as it

# Input

input_dict= io.process_jfile('H0_mcmc.json')
state_dict, _, vparam_dict = it.parse_input_dict(
    input_dict)