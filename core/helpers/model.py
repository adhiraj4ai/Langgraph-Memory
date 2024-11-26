from decouple import config

def is_reasoning_model_enabled():
    """
    Checks if the current model is a reasoning model.

    Returns:
        bool: True if the current model is a reasoning model, False otherwise.
    """
    if config('ENABLE_REASONING_MODEL')  == 'True':
        if config('REASONING_MODEL') is None:
            raise ValueError("REASONING_MODEL is not set")
        return True
    else:
        return False
