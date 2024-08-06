def wrap_env(env, wrappers = None):
    wrappers = wrappers or []
    for wrapper in wrappers:
        cls = wrapper["cls"]
        args = wrapper.get("args", tuple())
        kwargs = wrapper.get("kwargs", dict())
        env = cls(env, *args, **kwargs)
    return env