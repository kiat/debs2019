import requests
import json
import pandas as pd
import os

from plugin.load_model import load_graph, return_prediction, object_names_func


def host_url(host, path):
    return "http://" + host + path


def get_scene(host):
    return requests.get(host_url(host, "/scene/"))


def post_answer(host, payload):
    headers = {"Content-type": "application/json"}
    response = requests.post(host_url(host, "/scene/"), json=payload, headers=headers)

    print("Response status is: ", response.status_code)
    if response.status_code == 201:
        return {"status": "success", "message": "updated"}
    if response.status_code == 404:
        return {
            "message": "Something went wrong. No scene exist. Check if the path is correct"
        }


def main():
    print("ENV is ", os.getenv("BENCHMARK_SYSTEM_URL"))

    host = os.getenv("BENCHMARK_SYSTEM_URL")
    if host is None or "":
        print("Error reading Server address!")

    print("Getting scenes for predictions...")

    # Creating the session
    session, img_length, img_height, y_pred_cls, x = load_graph(layers=False, path_to_model="model/two_d_cnn_proj.ckpt")
    object_names = object_names_func()

    # Here is an automated script for getting all scenes
    while True:

        response = get_scene(host)

        if response.status_code == 404:
            print(response.json())
            break

        data = response.json()

        reconstructed_scene = pd.read_json(data["scene"], orient="records")

        result = return_prediction(
            reconstructed_scene,
            session,
            object_names,
            img_length,
            img_height,
            y_pred_cls,
            x,
            True,
            'perspective',
            True
        )

        post_answer(host, result)

    print("Submission for all scenes done successfully!")


if __name__ == "__main__":
    main()
