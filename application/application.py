import asyncio

from dotenv import load_dotenv
load_dotenv()
import immich_tools


async def main():
    pictures_list = await immich_tools.search_immich_assets("cats")
    print(pictures_list)

if __name__ == "__main__":
    asyncio.run(main())
