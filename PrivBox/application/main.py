# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import sys
import yaml
import logging
import paddle

from dataset_generator import DatasetGenerator
from model_generator import ModelGenerator
from report_generator import ReportGenerator
from privbox.inference.membership_inference import BaselineMembershipInferenceAttack, MLLeaksMembershipInferenceAttack
from privbox.metrics import AUC, MSE, Accuracy, Precision, Recall


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AttackExecutor(object):
    """R
    PrivBox's executor
    """

    def __init__(self, reporter):
        """R
        """
        self.reporter_ = reporter

    def run(self, conf):
        """R
        run for attacks
        """
        type = conf["type"]
        name = conf["name"]
        args = conf["args"]

        logger.info("Launch Attack type is " + type)
        
        if type == "MEMBERSHIP_INFERENCE_ATTACK":
            mem_inf_exe = MemberInf(self.reporter_)
            return mem_inf_exe.run(name, args)
        else:
            raise NotImplementedError("Attack type " + type + " is not supported yet.")


class MemberInf(object):
    """R
    Membership inference attack executor
    """
    def __init__(self, reporter):
        """R
        """
        self.reporter_ = reporter

    def run(self, name, args):
        """R
        run membership inference attack
        """
        logger.info("Launch attack name is " + name)
        ret = {}
        if (name == 'BASELINE'):
            ret["base_attack_result"] = self.base(args)
        elif name == 'ML-LEAK':
            ret["ml_leak_result"] = self.ml_leak(args)
        else:
            raise NotImplementedError("Membership inference attack name " + name + " is not supported yet.")

        return ret

    def base(self, args):
        """R
        run baseline attack
        """
        logger.info("Begin Baseline attack.")
        logger.info("Generate target datasets(member dataset adn non-member dataset).")
        input_datasets = args["test_datasets"]
        dataset_generator = DatasetGenerator()
        mem_dataset, mem_dataset_name = dataset_generator.gen(input_datasets[0])
        non_mem_dataset, non_mem_dataset_name = dataset_generator.gen(input_datasets[1])

        # record dataset report
        if not self.reporter_.has_dataset_dict(mem_dataset_name):
            mem_dataset_len = len(mem_dataset)
            self.reporter_.add_datasets_dict({"name": mem_dataset_name,
                                            "is_member": "true",
                                            "length": mem_dataset_len})
        if not self.reporter_.has_dataset_dict(non_mem_dataset_name):
            non_mem_dataset_len = len(non_mem_dataset)
            
            self.reporter_.add_datasets_dict({"name": non_mem_dataset_name,
                                            "is_member": "false",
                                            "length": non_mem_dataset_len})
        
        logger.info("Generate target model.")
        input_model = args["target_model"]
        model_generator = ModelGenerator()
        target_net, target_model_name = model_generator.gen(input_model)

        model = paddle.Model(target_net)
        model.prepare(metrics=[paddle.metric.Accuracy()])

        # get predict result
        mem_pred = model.predict(mem_dataset, batch_size=128, stack_outputs=True)
        non_mem_pred = model.predict(non_mem_dataset, batch_size=128, stack_outputs=True)

        # record model report
        if not self.reporter_.has_model_dict(target_model_name):
            logger.info("Begin evaluate model with train dataset " + mem_dataset_name)
            train_acc = model.evaluate(mem_dataset, batch_size=128, verbose=0)["acc"]
            
            logger.info("Begin evaluate model with test dataset " + non_mem_dataset_name)
            test_acc = model.evaluate(non_mem_dataset, batch_size=128, verbose=0)["acc"]

            self.reporter_.add_models_dict({"name": target_model_name,
                                            "train_acc": train_acc,
                                            "test_acc": test_acc})
            logger.info("Evaluate model finished.")

        # run base attack
        mem_pred = paddle.argmax(paddle.to_tensor(mem_pred[0]), axis=-1)
        non_mem_pred = paddle.argmax(paddle.to_tensor(non_mem_pred[0]), axis=-1)

        input_data = paddle.concat([mem_pred, non_mem_pred], axis=0)
        input_label = self._get_all_labels([mem_dataset, non_mem_dataset])

        # membership attack
        logger.info("Init attack.")
        attack = BaselineMembershipInferenceAttack()
        logger.info("Begin membership inference attack.")
        result = attack.infer([input_data, input_label])

        # evaluate
        mem_label = paddle.ones(mem_pred.shape)
        non_mem_label = paddle.zeros(non_mem_pred.shape)
        expected = paddle.concat([mem_label, non_mem_label], axis=0)
        logger.info("Begin evaluating attack result")
        eval_res = attack.evaluate(result, expected,
                        metric_list=[Accuracy(False, 2), AUC(False), Precision(), Recall()])
        
        # record attack report
        self.reporter_.add_attacks_dict({"name": "Baseline Attack",
                                         "desc": "An instance is considered as a member "
                                                 "if its predict result is correct. "
                                                 "It only requires data with labels.",
                                         "acc": eval_res[0],
                                         "auc": eval_res[1],
                                         "precision": eval_res[2],
                                         "recall": eval_res[3],
                                         "recommend": "Use dropout or regularization methods "
                                                      "to avoid overfitting when training."})
        logger.info("Baseline attack is finished.")
        return {"Accuracy": eval_res[0],
                "AUC": eval_res[1],
                "Precision": eval_res[2],
                "Recall": eval_res[3]}

    def ml_leak(self, args):
        """R
        run ml_leak attack
        """
        logger.info("Begin ML-LEAK attack.")
        logger.info("Generate target datasets(member dataset adn non-member dataset).")
        dataset_generator = DatasetGenerator()
        target_datasets = args["target_datasets"]
        target_dataset_mem, target_dataset_mem_name = dataset_generator.gen(target_datasets[0])
        target_dataset_non_mem, target_dataset_non_mem_name = dataset_generator.gen(target_datasets[1])

        if not self.reporter_.has_dataset_dict(target_dataset_mem_name):
            self.reporter_.add_datasets_dict({"name": target_dataset_mem_name,
                                            "is_member": "true",
                                            "length": len(target_dataset_mem)})

        if not self.reporter_.has_dataset_dict(target_dataset_non_mem_name):
            self.reporter_.add_datasets_dict({"name": target_dataset_non_mem_name,
                                            "is_member": "false",
                                            "length": len(target_dataset_non_mem)})

        logger.info("Generate shadow datasets(member dataset adn non-member dataset).")
        shadow_datasets = args["shadow_datasets"]
        shadow_dataset_mem, _ = dataset_generator.gen(shadow_datasets[0])
        shadow_dataset_non_mem, _ = dataset_generator.gen(shadow_datasets[1])

        model_generator = ModelGenerator()
        shadow_model, _ = model_generator.gen(args["shadow_model"], False)

        logger.info("Generate target model.")
        target_model, target_model_name = model_generator.gen(args["target_model"])
        target_model = paddle.Model(target_model)
        target_model.prepare(metrics=paddle.metric.Accuracy())

        shadow_epoch = args["shadow_epoch"]
        shadow_lr = args["shadow_lr"]
        classifier_epoch = args["classifier_epoch"]
        classifier_lr = args["classifier_lr"]
        batch_size = args["batch_size"]
        topk = args["topk"]

        logger.info("Init attack.")
        attack = MLLeaksMembershipInferenceAttack(shadow_model, [shadow_dataset_mem, shadow_dataset_non_mem])

        attack_params = {"batch_size": batch_size,
                         "shadow_epoch": shadow_epoch,
                         "classifier_epoch": classifier_epoch,
                         "topk": topk,
                         "shadow_lr": shadow_lr,
                         "classifier_lr": classifier_lr}

        attack.set_params(**attack_params)

        logger.info("Infer target dataset")
        
        mem_pred = target_model.predict(target_dataset_mem, batch_size=batch_size, stack_outputs=True)
        non_mem_pred = target_model.predict(target_dataset_non_mem, batch_size=batch_size, stack_outputs=True)

        # record model report
        if not self.reporter_.has_model_dict(target_model_name):
            logger.info("Begin evaluate model with train dataset " + target_dataset_mem_name)
            train_acc = target_model.evaluate(target_dataset_mem, batch_size=128, verbose=0)["acc"]
            
            logger.info("Begin evaluate model with test dataset " + target_dataset_non_mem_name)
            test_acc = target_model.evaluate(target_dataset_non_mem, batch_size=128, verbose=0)["acc"]

            self.reporter_.add_models_dict({"name": target_model_name,
                                            "train_acc": train_acc,
                                            "test_acc": test_acc})
            logger.info("Evaluation model finished")

        mem_pred = paddle.to_tensor(mem_pred[0])
        non_mem_pred = paddle.to_tensor(non_mem_pred[0])

        data = paddle.concat([mem_pred, non_mem_pred])
        logger.info("Begin membership inference attack.")
        result = attack.infer(data)

        # evaluate
        mem_label = paddle.ones((mem_pred.shape[0], 1))
        non_mem_label = paddle.zeros((non_mem_pred.shape[0], 1))
        expected = paddle.concat([mem_label, non_mem_label], axis=0)
        logger.info("Begin evaluating attack results.")
        eval_res = attack.evaluate(expected, result, metric_list=[Accuracy(), AUC(), Precision(), Recall()])

        # record attack report
        self.reporter_.add_attacks_dict({"name": "ML-LEAK Attack",
                                         "desc": "A membership inference attack based on auxiliary dataset, "
                                                 "shadow model and prediction confidence",
                                         "acc": eval_res[0],
                                         "auc": eval_res[1],
                                         "precision": eval_res[2],
                                         "recall": eval_res[3],
                                         "recommend": 
                                         "Use dropout or regularization methods to avoid overfitting when training. "
                                         "By rounding, labeling, cliping model prediction confidence "
                                         "can also reduce the leakage of membership inference."})
        logger.info("ML-LEAK attack is finished.")
        return {"Accuracy": eval_res[0],
                "AUC": eval_res[1],
                "Precision": eval_res[2],
                "Recall": eval_res[3]}

    def _get_all_labels(self, data_list):
        """
        get labels from multiple dataset
        """
        labels = []
        for dataset in data_list:
            for data in dataset:
                labels.append(data[1])

        return paddle.to_tensor(labels)


if __name__ == '__main__':
    conf_path = sys.argv[1]
    conf_f = open(conf_path, 'r', encoding="utf-8")
    conf_str = conf_f.read()
    conf_f.close()

    reporter = ReportGenerator()

    exe = AttackExecutor(reporter)

    attacks_conf = yaml.load(conf_str)

    results = []

    for attack_conf in attacks_conf:
        results.append(exe.run(attack_conf))

    print(reporter.mem_inf_report())
