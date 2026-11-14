import os
from langchain_openai import AzureChatOpenAI
from immich_tools import immich_tools

class ai_model:
    def get():
        """Returns AzureChatOpenAI configured from environment variables."""
        azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        
        if not azure_api_key or not azure_endpoint:
            raise ValueError(
                "AZURE_OPENAI_API_KEY și AZURE_OPENAI_ENDPOINT must configured in .env file!"
            )
        
        deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")
        
        llm = AzureChatOpenAI(
            azure_endpoint=azure_endpoint,
            azure_deployment=deployment_name,
            api_version=api_version,
            api_key=azure_api_key,
            temperature=0.0,
        )
        return llm.bind_tools(immich_tools, parallel_tool_calling=False)
