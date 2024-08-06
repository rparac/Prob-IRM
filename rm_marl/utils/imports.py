def import_cls(path):
    module, clazz = path.rsplit(".", 1)
    return getattr(__import__(module, fromlist=module.split(".")), clazz)