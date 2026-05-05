import json

import httpx
from loguru import logger
from sqlalchemy import inspect, text
from sqlmodel import SQLModel

from agentchat.api.services.agent import AgentService
from agentchat.api.services.llm import LLMService
from agentchat.api.services.mcp_server import MCPService
from agentchat.api.services.tool import ToolService
from agentchat.core.agents.structured_response_agent import StructuredResponseAgent
from agentchat.database import AgentTable, SystemUser, ToolTable, engine, ensure_mysql_database
from agentchat.database.dao.agent import AgentDao
from agentchat.database.models.user import AdminUser
from agentchat.prompts.mcp import McpAsToolPrompt
from agentchat.schema.mcp import MCPResponseFormat
from agentchat.services.mcp.manager import MCPManager
from agentchat.services.storage import storage_client
from agentchat.settings import app_settings
from agentchat.utils.convert import convert_mcp_config
from agentchat.utils.helpers import get_provider_from_model


async def init_database():
    try:
        ensure_mysql_database()
        SQLModel.metadata.create_all(engine)
        ensure_knowledge_file_schema()
        logger.success("MySQL tables are ready")
    except Exception as err:
        logger.error(f"Create MySQL Table Error: {err}")


def ensure_knowledge_file_schema():
    inspector = inspect(engine)
    if "knowledge_file" not in inspector.get_table_names():
        return

    existing_columns = {column["name"] for column in inspector.get_columns("knowledge_file")}
    alter_statements = []

    if "error_message" not in existing_columns:
        alter_statements.append("ALTER TABLE knowledge_file ADD COLUMN error_message TEXT NULL")
    if "parse_engine" not in existing_columns:
        alter_statements.append("ALTER TABLE knowledge_file ADD COLUMN parse_engine VARCHAR(64) NOT NULL DEFAULT ''")
    if "parse_mode" not in existing_columns:
        alter_statements.append("ALTER TABLE knowledge_file ADD COLUMN parse_mode VARCHAR(32) NOT NULL DEFAULT 'sync'")
    if "finished_at" not in existing_columns:
        alter_statements.append("ALTER TABLE knowledge_file ADD COLUMN finished_at DATETIME NULL")

    if not alter_statements:
        return

    with engine.begin() as conn:
        for statement in alter_statements:
            conn.execute(text(statement))
    logger.success("Knowledge file schema is ready")


async def init_default_agent():
    try:
        result = await AgentService.get_agent()
        if len(result) == 0:
            logger.info("Initializing default agents in MySQL")
            await insert_tools_to_mysql()
            await insert_llm_to_mysql()
            await insert_agent_to_mysql()
            logger.success("Default agents initialized successfully")
        else:
            logger.info("Default agents already initialized")
    except Exception as err:
        logger.error(f"Failed to initialize default agents: {err}")


async def update_system_mcp_server():
    try:
        mcp_server = await MCPService.get_all_servers(SystemUser)
        if len(mcp_server):
            await update_mcp_server_into_mysql(True)
        else:
            await update_mcp_server_into_mysql(False)
    except Exception as err:
        logger.error(f"Failed to initialize system MCP server: {err}")


async def insert_agent_to_mysql():
    llm = await LLMService.get_one_llm()
    tools = await ToolService.get_tools_data()
    for tool in tools:
        tool["name"] = tool["display_name"] + "助手"
        await AgentDao.create_agent(
            AgentTable(
                **ToolTable(**tool).model_dump(exclude={"user_id", "tool_id"}),
                tool_ids=[tool["tool_id"]],
                user_id=SystemUser,
                is_custom=False,
                llm_id=llm.get("llm_id"),
            )
        )


async def insert_llm_to_mysql():
    api_key = app_settings.multi_models.conversation_model.api_key
    base_url = app_settings.multi_models.conversation_model.base_url
    model = app_settings.multi_models.conversation_model.model_name
    provider = get_provider_from_model(model)

    await LLMService.create_llm(
        user_id=SystemUser,
        model=model,
        llm_type="LLM",
        api_key=api_key,
        base_url=base_url,
        provider=provider,
    )


async def insert_tools_to_mysql():
    tools = await load_default_tool()
    for tool in tools:
        await ToolService.create_default_tool(
            ToolTable(
                **tool,
                user_id=SystemUser,
                is_user_defined=False,
            )
        )


async def update_mcp_server_into_mysql(has_mcp_server: bool):
    if has_mcp_server:
        if await MCPService.mcp_server_need_update():
            servers = await MCPService.get_all_servers(AdminUser)
            logger.info("Updating MCP Server to the latest version in the database")
        else:
            return
    else:
        servers = await load_system_mcp_server()

    servers_info = []
    for server in servers:
        servers_info.append(
            {
                "type": server["type"],
                "url": server["url"],
                "server_name": server["server_name"],
            }
        )

    mcp_manager = MCPManager(convert_mcp_config(servers_info))
    servers_params = await mcp_manager.show_mcp_tools()

    async def get_config_from_server_name(server_name):
        for server in servers:
            if server["server_name"] == server_name:
                return server
        return None

    async def get_tools_name_from_params(tools_params: dict):
        tools_name = []
        for tool in tools_params:
            tools_name.append(tool["name"])
        return tools_name

    for key, params in servers_params.items():
        server = await get_config_from_server_name(key)
        tools_name = await get_tools_name_from_params(params)

        structured_agent = StructuredResponseAgent(MCPResponseFormat)
        structured_response = structured_agent.get_structured_response(
            McpAsToolPrompt.format(tools_info=json.dumps(params, indent=4))
        )

        if has_mcp_server:
            update_values = {
                "tools": tools_name,
                "params": params,
                "mcp_as_tool_name": structured_response.mcp_as_tool_name,
                "description": structured_response.description,
            }
            await MCPService.update_mcp_server(
                server_id=server["mcp_server_id"],
                update_data=update_values,
            )
        else:
            await MCPService.create_mcp_server(
                server_name=key,
                user_id=SystemUser,
                user_name="Admin",
                url=server["url"],
                type=server["type"],
                config=server["config"],
                tools=tools_name,
                params=params,
                config_enabled=server["config_enabled"],
                logo_url=server["logo_url"],
                mcp_as_tool_name=structured_response.mcp_as_tool_name,
                description=structured_response.description,
            )


async def upload_user_avatars_storage():
    if not storage_client.list_files_in_folder("icons/user"):
        user_avatars = await load_user_avatars()
        for avatar_url in user_avatars["avatars"]:
            async with httpx.AsyncClient() as client:
                response = await client.get(avatar_url)
                image_data = response.content

            file_name = avatar_url.split("/")[-1]
            object_name = f"icons/user/{file_name}"
            storage_client.upload_file(object_name, image_data)


async def load_default_tool():
    with open("./agentchat/config/tool.json", "r", encoding="utf-8") as f:
        result = json.load(f)
    return result


async def load_system_mcp_server():
    with open("./agentchat/config/mcp_server.json", "r", encoding="utf-8") as f:
        result = json.load(f)
    return result


async def load_user_avatars():
    with open("./agentchat/config/avatars.json", "r", encoding="utf-8") as f:
        result = json.load(f)
    return result
