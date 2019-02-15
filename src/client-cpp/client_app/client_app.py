import requests
import json
import pandas as pd
import os


def host_url(host, path):
    return "http://" + host + path


def get_scene(host):
    return requests.get(host_url(host, '/scene/'))


def post_answer(host, payload):
    headers = {'Content-type': 'application/json'}
    response = requests.post(host_url(host, '/scene/'), json = payload, headers=headers)

    print('Response status is: ', response.status_code)
    if (response.status_code == 201):
        return {'status': 'success', 'message': 'updated'}
    if (response.status_code == 404):
        return {'message': 'Something went wrong. No scene exist. Check if the path is correct'}


if __name__ == "__main__":
    print('ENV is ', os.getenv('BENCHMARK_SYSTEM_URL'))

    host = os.getenv('BENCHMARK_SYSTEM_URL')
    if host is None or '':
        print('Error reading Server address!')

    print('Getting scenes for predictions...')

    # Here is an automated script for getting all scenes
    # and submitting prediction for each of them
    # you may change to fit your needs
    while(True):

        # Making GET request
        # Each request will fetch new scene
        response = get_scene(host)
        if response.status_code == 404:
            print(response.json())
            break

        data = response.json()

        # example of reconstruction json payload from GET request into DataFrame
        reconstructed_scene = pd.read_json(data['scene'], orient='records')

        # DO YOUR PREDICTIONS HERE
        # return the result in format in plain json:
        # for example:
        example_result = {'car': 1, 'armchair': 2}
        # after making sure the result in correct form you need to submit it
        # via POST request:
        post_answer(host, example_result)

    print('Submission for all scenes done successfully!')
