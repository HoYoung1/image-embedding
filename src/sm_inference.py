import tempfile

from skimage import io


def input_fn(request_body, request_content_type):
    """An input_fn that loads a pickled tensor"""
    if request_content_type == 'application/binary':

        with tempfile.NamedTemporaryFile("w+b") as f:
            f.write(request_body)
            f.seek(0)
            image = io.imread(f.name)
            return image
    else:
        # Handle other content-types here or raise an Exception
        # if the content type is not supported.
        raise "Unsupported content type {}".format(request_content_type)
