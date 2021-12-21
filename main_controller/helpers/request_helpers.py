from logging import error
import asyncio
from aiohttp import ClientSession


async def post_to_instance(url, session, data):
    response = await session.post(url=url, data=data)
    return response.text()


async def post_data_distribution(environment_data_distribution):
    async with ClientSession() as client_session:
        post_requests = []
        for environment_ip, data_distribution in environment_data_distribution.items():
            post_requests.append(
                post_to_instance(
                    environment_ip + "/dataset/distribution",
                    client_session,
                    data_distribution,
                )
            )

        results = await asyncio.gather(*post_requests)
        error(results)
