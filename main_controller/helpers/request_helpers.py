from logging import error
import asyncio
from aiohttp import ClientSession


async def post_to_instance(url, session, data):
    headers = {"content-type": "multipart/form-data"}
    response = await session.post(url=url, data=data, headers=headers)
    return response.text()


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
    return {
        key: instance_data.get(key, []).append(data[i])
        for key in data_keys
        for data in files.getlist(key)
        for i in data_distribution
    }


async def post_data_distribution(files, environment_data_distribution):
    async with ClientSession() as client_session:
        post_requests = []
        for environment_ip, data_distribution in environment_data_distribution.items():
            instance_data = get_instance_data_from_files(files, data_distribution)
            post_requests.append(
                post_to_instance(
                    environment_ip + "/dataset/distribution",
                    client_session,
                    instance_data,
                )
            )

        results = await asyncio.gather(*post_requests)
        error(results)
