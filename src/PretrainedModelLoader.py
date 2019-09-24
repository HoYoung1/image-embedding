# *************************************************************
# * Copyright 2019 Amazon.com, Inc. and its affiliates. All Rights Reserved.
# 
# Licensed under the Amazon Software License (the "License").
#  You may not use this file except in compliance with the License.
# A copy of the License is located at
# 
#  http://aws.amazon.com/asl/
# 
#  or in the "license" file accompanying this file. This file is distributed
#  on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
#  express or implied. See the License for the specific language governing
#  permissions and limitations under the License.
# ***************************************************************

import inspect

import torchvision.models


class PretrainedModelLoader:
    """
    Loads a pretrained model
    """

    def __init__(self):
        self._valid_models = [n for n, _ in inspect.getmembers(torchvision.models, inspect.isfunction)]

    def __call__(self, model_name):
        """
Returns a pretrained model based on the model_name argument. The model name should be one of the models defined in torchvision.models.
        :param model_name: The model name defined in torchvision.models. E.g. 'alexnet', 'densenet121', 'densenet161', 'densenet169', 'densenet201', 'inception_v3', 'resnet101', 'resnet152', 'resnet18', 'resnet34', 'resnet50', 'squeezenet1_0', 'squeezenet1_1' etc
        :return: Pytorch model object
        """
        # Check if the model name passed is a callable
        model_func = getattr(torchvision.models, model_name, None)
        assert callable(
            model_func) == True, "The function torchvision.models.{} must be a callable. The valid list of callables are {}".format(
            model_name, self._valid_models)

        # Create model
        model = model_func(pretrained=True)

        return model
