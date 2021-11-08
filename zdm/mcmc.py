import pymc3 as pm

def pm_model(vparam_dict):
    """ Builds a pymc3 model for zdm

    Args:
        parm_dict (dict): dict with the pymc3 parameters
        tight_ObH (bool, optional): If True, restrict the ObH0 value based on CMB. Defaults to False.
        beta (float, optional): PDF parameter. Defaults to 3..

    Raises:
        IOError: [description]

    Returns:
        pm.Model: pymc3 model
    """
    # Load the model
    with pm.Model() as model:
        # Define Variables
        pm_vars = []
        for key in vparam_dict.keys():
            if vparam_dict[key]['prior'] == 'Uniform':
                pvar = pm.Uniform(key, 
                    lower=vparam_dict[key]['lower'],
                    upper=vparam_dict[key]['upper'])
            else:
                raise ValueError("Bad input!")
        like = pm.Potential(
            'like', 
            calc_likelihood(*pm_vars, vparam_dict=vparam_dict)
    return model

def calc_likelihood(args, vparam_dict=vparam_dict):
    pass