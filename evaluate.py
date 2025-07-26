"""
The following is a simple example evaluation method.

It is meant to run within a container. Its steps are as follows:

  1. Read the algorithm output
  2. Associate original algorithm inputs with a ground truths via predictions.json
  3. Calculate metrics by comparing the algorithm output to the ground truth
  4. Repeat for all algorithm jobs that ran for this submission
  5. Aggregate the calculated metrics
  6. Save the metrics to metrics.json

To run it locally, you can call the following bash script:

  ./do_test_run.sh

This will start the evaluation and reads from ./test/input and writes to ./test/output

To save the container and prep it for upload to Grand-Challenge.org you can call:

  ./do_save.sh

Any container that shows the same behaviour will do, this is purely an example of how one COULD do it.

Reference the documentation to get details on the runtime environment on the platform:
https://grand-challenge.org/documentation/runtime-environment/

Happy programming!
"""

import json


import random
from statistics import mean
from pathlib import Path
from pprint import pformat, pprint
import numpy as np
from helpers import run_prediction_processing, tree
from evalutils.scorers import score_detection 

from torchmetrics.detection.mean_ap import MeanAveragePrecision


INPUT_DIRECTORY = Path("/input")
OUTPUT_DIRECTORY = Path("/output")

map_metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox', max_detection_thresholds=[1,10,1e6], rec_thresholds=np.arange(0,1.01,0.01).tolist())

def main():
    print_inputs()

    metrics = {}
    predictions = read_predictions()

    # We now process each algorithm job for this submission
    # Note that the jobs are not in any specific order!
    # We work that out from predictions.json

    # Use concurrent workers to process the predictions more efficiently
    metrics["results"] = run_prediction_processing(fn=process, predictions=predictions)

    # We have the results per prediction, we can aggregate the results and
    # generate an overall score(s) for this submission
    if metrics["results"]:
        metrics["aggregates"] = {
            "my_metric": mean(result["my_metric"] for result in metrics["results"])
        }

    # Make sure to save the metrics
    write_metrics(metrics=metrics)

    return 0


def process(job):
    # The key is a tuple of the slugs of the input sockets
    interface_key = get_interface_key(job)

    # Lookup the handler for this particular set of sockets (i.e. the interface)
    handler = {
        ("histopathology-region-of-interest-cropout",): process_interf0,
    }[interface_key]

    # Call the handler
    return handler(job)


def process_interf0(
    job,
):
    """Processes a single algorithm job, looking at the outputs"""
    report = "Processing:\n"
    report += pformat(job)
    report += "\n"

    # Firstly, find the location of the results
    location_mitotic_figures = get_file_location(
        job_pk=job["pk"],
        values=job["outputs"],
        slug="mitotic-figures",
    )

    # Secondly, read the results
    result_mitotic_figures = load_json_file(
        location=location_mitotic_figures,
    )

    # Thirdly, retrieve the input file name to match it with your ground truth
    image_name_histopathology_region_of_interest_cropout = get_image_name(
        values=job["inputs"],
        slug="histopathology-region-of-interest-cropout",
    )

    # Fourthly, load your ground truth

    ground_truth_dir = Path("/opt/ml/input/data/ground_truth")
    with open(
        ground_truth_dir / "ground_truth.json", "r"
    ) as f:
        truth = f.read()

    if image_name_histopathology_region_of_interest_cropout not in truth:
        raise NameError(f'Ground truth for processed image {image_name_histopathology_region_of_interest_cropout} not found.')
    
    if 'points' not in result_mitotic_figures:
        raise SyntaxError('Results must contain dictionary with "points" field. ')

    points=[]
    for point in result_mitotic_figures['points']:
            detected_class = 1 if 'name' not in point or point['name']=='mitotic figure' else 0
            detected_thr   = 0.5 if 'probability' not in point else point['probability']

            if 'name' not in point:
                print('Warning: Old format. Field name is not part of detections.')

            if 'probability' not in point:
                print('Warning: Old format. Field probability is not part of detections.')
            
            if 'point' not in point:
                print('Warning: Point is not part of points structure.')
                continue

            points.append([*point['point'][0:3], detected_class, detected_thr])


    # TODO: compare the results to your ground truth and compute some metrics
    filtered_predictions = [[x,y,0] for x,y,z,cls,sc in points if cls==1]

    bbox_size = 0.01125 # equals to 7.5mm distance for horizontal distance at 0.5 IOU
    
    sc = score_detection(ground_truth=truth[image_name_histopathology_region_of_interest_cropout]['annotations'],predictions=filtered_predictions,radius=7.5E-3)._asdict()

    return {
        "image" : image_name_histopathology_region_of_interest_cropout,
        "gt" : truth[image_name_histopathology_region_of_interest_cropout]['annotations'],
        "pred" : points,
        "metrics" : sc,
    }


def print_inputs():
    # Just for convenience, in the logs you can then see what files you have to work with
    print("Input Files:")
    for line in tree(INPUT_DIRECTORY):
        print(line)
    print("")


def read_predictions():
    # The prediction file tells us the location of the users' predictions
    return load_json_file(location=INPUT_DIRECTORY / "predictions.json")


def get_interface_key(job):
    # Each interface has a unique key that is the set of socket slugs given as input
    socket_slugs = [sv["interface"]["slug"] for sv in job["inputs"]]
    return tuple(sorted(socket_slugs))


def get_image_name(*, values, slug):
    # This tells us the user-provided name of the input or output image
    for value in values:
        if value["interface"]["slug"] == slug:
            return value["image"]["name"]

    raise RuntimeError(f"Image with interface {slug} not found!")


def get_interface_relative_path(*, values, slug):
    # Gets the location of the interface relative to the input or output
    for value in values:
        if value["interface"]["slug"] == slug:
            return value["interface"]["relative_path"]

    raise RuntimeError(f"Value with interface {slug} not found!")


def get_file_location(*, job_pk, values, slug):
    # Where a job's output file will be located in the evaluation container
    relative_path = get_interface_relative_path(values=values, slug=slug)
    return INPUT_DIRECTORY / job_pk / "output" / relative_path


def load_json_file(*, location):
    # Reads a json file
    with open(location) as f:
        return json.loads(f.read())


def write_metrics(*, metrics):
    # Write a json document used for ranking results on the leaderboard
    write_json_file(location=OUTPUT_DIRECTORY / "metrics.json", content=metrics)


def write_json_file(*, location, content):
    # Writes a json file
    with open(location, "w") as f:
        f.write(json.dumps(content, indent=4))


if __name__ == "__main__":
    raise SystemExit(main())
