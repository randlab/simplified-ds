import pickle

def compute(filename, function, *args, **kwargs):
    """
    Cache function result to pickle
    
    Checks if filename exists, then reads value from file.
    Otherwise calls function(*args, **kwargs), saves
    result to filename and returns result
    """
    try:
        with open(filename, 'rb') as fh:
            return pickle.load(fh)
    except FileNotFoundError:
            result = function(*args, **kwargs)
            with open(filename, 'wb') as fh:
                pickle.dump(result, fh)
            return result