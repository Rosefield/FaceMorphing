save_files = False
wait_for_input = True

class Struct(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def default_val(key, col, default):
    if(key in col):
        return col[key]
    else:
        return default

def prompt():
    input("press enter to continue")

    return

