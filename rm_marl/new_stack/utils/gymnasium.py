
def gym_getattr(env, attr):
    while hasattr(env, 'env'):
        if hasattr(env, attr):
            return getattr(env, attr)
        env = env.env
    return None
