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
import importlib
import os
import pkgutil

from datasets.custom_dataset_factorybase import CustomDatasetFactoryBase


class DatasetFactory:

    def __init__(self):
        base_dir = os.path.join(os.path.dirname(__file__), "datasets")

        # load subclasses of CustomDatasetFactoryBase from datasets.services
        classname_class_dict = {}
        for (module_loader, name, ispkg) in pkgutil.iter_modules([base_dir]):
            importlib.import_module('datasets.' + name)
            classname_class_dict = {cls.__name__: cls for cls in CustomDatasetFactoryBase.__subclasses__()}

        self._class_name_class_dict = classname_class_dict

    @property
    def dataset_factory_names(self):
        return list(self._class_name_class_dict.keys())

    def get_datasetfactory(self, class_name):
        if class_name in self._class_name_class_dict:
            return self._class_name_class_dict[class_name]()
        else:
            raise ModuleNotFoundError("Module should be in {}".format(self.dataset_factory_names))
