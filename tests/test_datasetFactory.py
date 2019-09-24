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
from unittest import TestCase

from dataset_factory import DatasetFactory
from datasets.custom_dataset_factorybase import CustomDatasetFactoryBase


class TestDatasetFactory(TestCase):

    def test_dataset_factory_names(self):
        # Arrange
        sut = DatasetFactory()

        # act
        class_names = sut.dataset_factory_names

        # assert
        self.assertEqual(len(class_names), 2,
                         " The number of expected dataset factory classes doesnt match.. Check the number of classes that inhert from {}  !".format(
                             type(CustomDatasetFactoryBase)))

    def test_get_datasetfactory(self):
        # Arrange
        sut = DatasetFactory()
        class_names = sut.dataset_factory_names

        # Act
        obj = sut.get_datasetfactory(class_names[0])

        # assert
        self.assertIsInstance(obj, CustomDatasetFactoryBase)
