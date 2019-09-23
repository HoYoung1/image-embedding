from Predictor import Predictor


def input_fn(request_body, request_content_type):
    """An input_fn that processes the request body to a tensor"""
    if request_content_type == 'application/binary':
        return request_body
    else:
        # Handle other content-types here or raise an Exception
        # if the content type is not supported.
        raise "Unsupported content type {}".format(request_content_type)


def model_fn(model_dir):
    return Predictor(model_dir)


def predict_fn(input_data, model):
    """Predict using input and model"""
    return model(input_data)


def output_fn(prediction, content_type):
    """Return prediction"""
    return prediction
