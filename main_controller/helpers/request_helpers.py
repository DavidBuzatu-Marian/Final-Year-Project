from logging import error
import asyncio
from aiohttp import ClientSession


async def post_to_instance(
    url, session, data, headers={"content-type": "multipart/form-data"}
):
    response = await session.post(url=url, data=data, headers=headers)
    return response


def get_instance_data_from_files(files, data_distribution):
    data_keys = [
        "train_data",
        "train_labels",
        "validation_data",
        "validation_labels",
        "test_data",
        "test_labels",
    ]
    instance_data = dict()
    error(files.getlist("train_data"))
    return {
        key: instance_data.get(key, []).append(files.getlist(key)[i])
        for key in data_keys
        for i in data_distribution
        if len(files.getlist(key)) > 0
    }


async def post_data_distribution(files, environment_data_distribution):
    async with ClientSession() as client_session:
        post_requests = []
        for environment_ip, data_distribution in environment_data_distribution.items():
            instance_data = get_instance_data_from_files(files, data_distribution)
            post_requests.append(
                post_to_instance(
                    "http://" + environment_ip + ":5000/dataset/add",
                    client_session,
                    instance_data,
                )
            )

        results = await asyncio.gather(*post_requests)
        error(results)
