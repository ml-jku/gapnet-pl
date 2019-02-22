import importlib
import inspect
from typing import Union

import numpy as np
from TeLL.config import Config
from torchvision.transforms import Compose


def import_object(objname):
    module_str = objname.split('.', maxsplit=1)
    # small hack for more intuitive import of tf modules
    if module_str[0] == "tf":
        module_str[0] = "tensorflow"
    objmodule = importlib.import_module(module_str[0])
    return get_rec_attr(objmodule, module_str[-1])


def invoke_functional_with_params(method_call):
    mname, params = method_call[:method_call.index("(")], method_call[method_call.index("(") + 1:-1]
    method = import_object(mname)
    if len(params) > 0:
        params = eval(params)
        if not (isinstance(params, list) or isinstance(params, tuple)):
            params = (params,)
        return method(*params)
    else:
        return method()


def get_rec_attr(obj, attrstr):
    """Get attributes and do so recursively if needed"""
    if attrstr is None:
        return None
    if "." in attrstr:
        attrs = attrstr.split('.', maxsplit=1)
        if hasattr(obj, attrs[0]):
            obj = get_rec_attr(getattr(obj, attrs[0]), attrs[1])
        else:
            try:
                obj = get_rec_attr(importlib.import_module(obj.__name__ + "." + attrs[0]), attrs[1])
            except ImportError:
                raise
    else:
        if hasattr(obj, attrstr):
            obj = getattr(obj, attrstr)
    return obj


def invoke_dataset_from_config(config: Config, required: Union[str, list, tuple] = None):
    """
    Initializes datasets from config. Imports specified data reader and instantiates it with parameters from config.
    :param config: config
    :param required: string, list or tuple specifying which datasets have to be loaded (e.g. ["train", "val"])
    :return: initialized data readers
    """
    # Initialize Data Reader if specified
    readers = {}
    if config.has_value("dataset"):
        def to_list(value):
            if value is None:
                result = []
            elif isinstance(value, str):
                result = list([value])
            else:
                result = list(value)
            return result
        
        dataset = config.dataset
        required = to_list(required)
        
        try:
            reader_class = import_object(dataset["reader"])
            reader_args = inspect.signature(reader_class).parameters.keys()
            datasets = [key for key in dataset.keys() if key not in reader_args and key != "reader"]
            global_args = [key for key in dataset.keys() if key not in datasets and key != "reader"]
            
            # check for required datasets before loading anything
            if required is not None:
                required = to_list(required)
                missing = [d for d in required if d not in datasets]
                if len(missing) > 0:
                    raise Exception("Missing required dataset(s) {}".format(missing))
            
            # read "global" parameters
            global_pars = {}
            for key in global_args:
                value = dataset[key]
                global_pars[key] = value
                if isinstance(value, str) and "import::" in value:
                    global_pars[key] = import_object(value[len("import::"):])
                if key == "transforms":
                    global_pars[key] = Compose([invoke_functional_with_params(t) for t in value])
            
            # read dataset specific parameters
            for dset in datasets:
                # inspect parameters and resolve if necessary
                for key, value in dataset[dset].items():
                    if isinstance(value, str) and "import::" in value:
                        dataset[dset][key] = import_object(value[len("import::"):])
                    if key == "transforms":
                        dataset[dset][key] = Compose([invoke_functional_with_params(t) for t in value])
                print("Loading dataset '{}'...".format(dset))
                readers[dset] = reader_class(**{**global_pars, **dataset[dset]})
        except (AttributeError, TypeError) as e:
            print("Unable to import '{}'".format(e))
            raise e
    return readers


class NormalizeByImageNumpy(object):
    """Normalize an tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """
    
    def __call__(self, image):
        """
        Args:
            image (Tensor): Tensor image of size (H, W, C) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        tensor = np.asarray(image, dtype=np.float32)
        for c in range(tensor.shape[-1]):
            t = tensor[:, :, c]
            tensor[:, :, c] = (t - t.mean()) / (t.std() + 1e-7)
        return tensor


class ToNumpyHWC(object):
    def __call__(self, tensor):
        """
        Args:
            image (Tensor): Tensor image of size (H, W, C) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        return tensor.permute(1, 2, 0).numpy()
