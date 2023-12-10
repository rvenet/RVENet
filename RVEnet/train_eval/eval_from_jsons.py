from RVEnet.train_eval.eval import evaluation
import sys
import os
import argparse


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-p","--predicted_json")
    parser.add_argument("-g","--groundtruth_json")
    parser.add_argument("-e","--evaluation_folder")

    args = parser.parse_args()

    # only this information is needed from the model parameters
    model_parameters = {"task_type": "regression"}

    evaluation_parameters = {"evaluation_task_type": "regression",
                            "evaluation_folder": args.evaluation_folder,
                            "is_heartcycle_averaging_needed": True,
                            "num_of_classes": 10}

    if not os.path.exists(args.evaluation_folder):
        os.makedirs(args.evaluation_folder)

    evaluation(model_parameters, evaluation_parameters, args.groundtruth_json, args.predicted_json, save_raw_comparison=True)
