import pickle

def save_computation(function, filename):
    try:
        with open(filename, 'rb') as fh:
            result = pickle.load(fh)
        def fun(*args, **kwargs):
            return result
    except FileNotFoundError:
        def fun(*args, **kwargs):
            result = function(*args, **kwargs)
            with open(filename, 'wb') as fh:
                pickle.dump(result, fh)
            return result
    finally:
        return fun