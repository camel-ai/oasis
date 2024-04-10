import asyncio
from social_platform.platform import Platform 

if __name__ == "__main__":
    platform = Platform()
    asyncio.run(platform.run())