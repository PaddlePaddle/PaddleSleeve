#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


class ReportGenerator(object):
    """R
    report generator
    """
    dict_ = {"datasets": [],
            "models": [],
            "attacks": []}

    def add_datasets_dict(self, dict):
        """R
        set dataset dict
        """
        self.dict_["datasets"].append(dict)

    def has_dataset_dict(self, dataset_name):
        """R
        test whether dataset has been added
        """
        has_dataset = False
        for dataset in self.dict_["datasets"]:
            if dataset["name"] == dataset_name:
                has_dataset = True
                break

        return has_dataset

    def add_models_dict(self, dict):
        """R
        set model dict
        """
        self.dict_["models"].append(dict)

    def has_model_dict(self, model_name):
        """R
        test whether model has been added
        """
        has_model = False
        for model in self.dict_["models"]:
            if model["name"] == model_name:
                has_model = True
                break

        return has_model

    def add_attacks_dict(self, dict):
        """R
        set attacks dict
        """
        self.dict_["attacks"].append(dict)

    def mem_inf_report(self):
        """R
        print report
        """
        def get_model_str(name, train_acc, test_acc):
            """R
            """
            return "    - name: {model_name}, train_acc: {train_acc}, test_acc: {test_acc}\n".format(
                                                                                model_name=name,
                                                                                train_acc=train_acc,
                                                                                test_acc=test_acc)

        def get_dataset_str(name, is_member, length):
            """R
            """
            return "    - name: {dataset_name}, is_member_dataset: {is_member}, length: {dataset_length}\n".format(
                                                                                dataset_name=name,
                                                                                is_member=is_member,
                                                                                dataset_length=length)

        def get_attack_str(name, desc, acc, auc, precision, recall, recommend):
            """R
            """
            return "    - name: {attack_name}\n      description: {desc}\n      acc: {acc}, auc: {auc}, precision: {precision}, recall: {recall}\n      recommend: {recommend}\n"\
                                                                                .format(attack_name=name,
                                                                                        desc=desc,
                                                                                        acc=acc,
                                                                                        auc=auc,
                                                                                        precision=precision,
                                                                                        recall=recall,
                                                                                        recommend=recommend)

        report_str = "Model Privacy Leakage Analysis Report\n"
        report_str += "  Models:\n"
        for model_dict in self.dict_["models"]:
            report_str += get_model_str(model_dict["name"], model_dict["train_acc"], model_dict["test_acc"])

        report_str += "  Datasets:\n"
        for dataset_dict in self.dict_["datasets"]:
            report_str += get_dataset_str(dataset_dict["name"], dataset_dict["is_member"], dataset_dict["length"])

        report_str += "  Attacks:\n"
        for attack_dict in self.dict_["attacks"]:
            report_str += get_attack_str(attack_dict["name"],
                                         attack_dict["desc"],
                                         attack_dict["acc"],
                                         attack_dict["auc"],
                                         attack_dict["precision"],
                                         attack_dict["recall"],
                                         attack_dict["recommend"])

        return report_str

    def print_extraction_report(self):
        """R
        """
        raise NotImplementedError("Extraction report has not been implemented.")

    def print_inversion_report(self):
        """R
        """
        raise NotImplementedError("Inversion report has not been implemented.")

