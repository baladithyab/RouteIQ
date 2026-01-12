"""
Tests for LLMRouter strategy integration.
"""

import os
import json
import tempfile


class TestLLMRouterStrategies:
    """Test LLMRouter strategy wrappers."""

    def test_strategy_list_defined(self):
        """Test that strategy list is defined."""
        from litellm_llmrouter.strategies import LLMROUTER_STRATEGIES

        assert len(LLMROUTER_STRATEGIES) > 0
        assert "llmrouter-knn" in LLMROUTER_STRATEGIES
        assert "llmrouter-mlp" in LLMROUTER_STRATEGIES
        assert "llmrouter-custom" in LLMROUTER_STRATEGIES

    def test_strategy_family_init(self):
        """Test LLMRouterStrategyFamily initialization."""
        from litellm_llmrouter.strategies import LLMRouterStrategyFamily

        # Create a temporary LLM data file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(
                {
                    "gpt-4": {"provider": "openai"},
                    "claude-3": {"provider": "anthropic"},
                },
                f,
            )
            llm_data_path = f.name

        try:
            strategy = LLMRouterStrategyFamily(
                strategy_name="llmrouter-knn",
                llm_data_path=llm_data_path,
                hot_reload=False,
            )

            assert strategy.strategy_name == "llmrouter-knn"
            assert strategy.hot_reload is False
            assert len(strategy._llm_data) == 2
        finally:
            os.unlink(llm_data_path)

    def test_should_reload_disabled(self):
        """Test reload check when disabled."""
        from litellm_llmrouter.strategies import LLMRouterStrategyFamily

        strategy = LLMRouterStrategyFamily(
            strategy_name="llmrouter-knn", hot_reload=False
        )

        assert strategy._should_reload() is False


class TestConfigLoader:
    """Test configuration loading utilities."""

    def test_imports(self):
        """Test that all exports are importable."""
        from litellm_llmrouter import (
            register_llmrouter_strategies,
            LLMROUTER_STRATEGIES,
            download_config_from_s3,
        )

        assert callable(register_llmrouter_strategies)
        assert callable(download_config_from_s3)
        assert isinstance(LLMROUTER_STRATEGIES, list)

    def test_version_defined(self):
        """Test that version is defined."""
        from litellm_llmrouter import __version__

        assert __version__ is not None
        assert isinstance(__version__, str)


class TestLLMDataLoading:
    """Test LLM candidates data loading."""

    def test_load_llm_data_from_file(self):
        """Test loading LLM data from JSON file."""
        from litellm_llmrouter.strategies import LLMRouterStrategyFamily

        test_data = {
            "model-a": {
                "provider": "provider-a",
                "capabilities": ["reasoning"],
                "quality_score": 0.9,
            },
            "model-b": {
                "provider": "provider-b",
                "capabilities": ["coding"],
                "quality_score": 0.85,
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_data, f)
            llm_data_path = f.name

        try:
            strategy = LLMRouterStrategyFamily(
                strategy_name="llmrouter-knn", llm_data_path=llm_data_path
            )

            assert "model-a" in strategy._llm_data
            assert "model-b" in strategy._llm_data
            assert strategy._llm_data["model-a"]["quality_score"] == 0.9
        finally:
            os.unlink(llm_data_path)

    def test_load_llm_data_missing_file(self):
        """Test handling of missing LLM data file."""
        from litellm_llmrouter.strategies import LLMRouterStrategyFamily

        strategy = LLMRouterStrategyFamily(
            strategy_name="llmrouter-knn", llm_data_path="/nonexistent/path.json"
        )

        # Should return empty dict on error
        assert strategy._llm_data == {}


class TestA2AGateway:
    """Test A2A Gateway functionality."""

    def test_a2a_gateway_init(self):
        """Test A2AGateway initialization."""
        from litellm_llmrouter.a2a_gateway import A2AGateway

        gateway = A2AGateway()
        assert gateway.agents == {}

    def test_a2a_agent_registration(self):
        """Test agent registration when enabled."""
        from litellm_llmrouter.a2a_gateway import A2AGateway, A2AAgent

        os.environ["A2A_GATEWAY_ENABLED"] = "true"
        try:
            gateway = A2AGateway()
            agent = A2AAgent(
                agent_id="test-agent",
                name="Test Agent",
                description="A test agent",
                url="http://localhost:8000",
                capabilities=["chat", "code"],
            )
            gateway.register_agent(agent)

            assert "test-agent" in gateway.agents
            assert gateway.get_agent("test-agent") == agent
            assert len(gateway.list_agents()) == 1
        finally:
            del os.environ["A2A_GATEWAY_ENABLED"]

    def test_a2a_discover_agents(self):
        """Test agent discovery by capability."""
        from litellm_llmrouter.a2a_gateway import A2AGateway, A2AAgent

        os.environ["A2A_GATEWAY_ENABLED"] = "true"
        try:
            gateway = A2AGateway()
            agent1 = A2AAgent(
                agent_id="agent1",
                name="Agent 1",
                description="Agent with chat",
                url="http://localhost:8001",
                capabilities=["chat"],
            )
            agent2 = A2AAgent(
                agent_id="agent2",
                name="Agent 2",
                description="Agent with code",
                url="http://localhost:8002",
                capabilities=["code"],
            )
            gateway.register_agent(agent1)
            gateway.register_agent(agent2)

            chat_agents = gateway.discover_agents("chat")
            assert len(chat_agents) == 1
            assert chat_agents[0].agent_id == "agent1"
        finally:
            del os.environ["A2A_GATEWAY_ENABLED"]


class TestMCPGateway:
    """Test MCP Gateway functionality."""

    def test_mcp_gateway_init(self):
        """Test MCPGateway initialization."""
        from litellm_llmrouter.mcp_gateway import MCPGateway

        gateway = MCPGateway()
        assert gateway.servers == {}

    def test_mcp_server_registration(self):
        """Test MCP server registration when enabled."""
        from litellm_llmrouter.mcp_gateway import MCPGateway, MCPServer, MCPTransport

        os.environ["MCP_GATEWAY_ENABLED"] = "true"
        try:
            gateway = MCPGateway()
            server = MCPServer(
                server_id="test-server",
                name="Test MCP Server",
                url="http://localhost:9000/mcp",
                transport=MCPTransport.STREAMABLE_HTTP,
                tools=["search", "fetch"],
            )
            gateway.register_server(server)

            assert "test-server" in gateway.servers
            assert gateway.get_server("test-server") == server
            assert len(gateway.list_servers()) == 1
        finally:
            del os.environ["MCP_GATEWAY_ENABLED"]

    def test_mcp_list_tools(self):
        """Test listing tools from all servers."""
        from litellm_llmrouter.mcp_gateway import MCPGateway, MCPServer

        os.environ["MCP_GATEWAY_ENABLED"] = "true"
        try:
            gateway = MCPGateway()
            server = MCPServer(
                server_id="server1",
                name="Server 1",
                url="http://localhost:9001",
                tools=["tool_a", "tool_b"],
            )
            gateway.register_server(server)

            tools = gateway.list_tools()
            assert len(tools) == 2
            assert tools[0]["tool"] == "tool_a"
        finally:
            del os.environ["MCP_GATEWAY_ENABLED"]


class TestConfigSync:
    """Test ConfigSyncManager functionality."""

    def test_config_sync_manager_init(self):
        """Test ConfigSyncManager initialization."""
        from litellm_llmrouter.config_sync import ConfigSyncManager

        manager = ConfigSyncManager()
        assert manager.sync_interval == 60
        assert manager._reload_count == 0

    def test_config_sync_status(self):
        """Test getting sync status."""
        from litellm_llmrouter.config_sync import ConfigSyncManager

        manager = ConfigSyncManager(sync_interval_seconds=120)
        status = manager.get_status()

        assert status["sync_interval_seconds"] == 120
        assert status["reload_count"] == 0
        assert "local_config_path" in status


class TestHotReload:
    """Test HotReloadManager functionality."""

    def test_hot_reload_manager_init(self):
        """Test HotReloadManager initialization."""
        from litellm_llmrouter.hot_reload import HotReloadManager

        manager = HotReloadManager()
        assert manager._router_reload_callbacks == {}

    def test_hot_reload_register_callback(self):
        """Test registering reload callbacks."""
        from litellm_llmrouter.hot_reload import HotReloadManager

        manager = HotReloadManager()
        callback_called = [False]

        def test_callback():
            callback_called[0] = True

        manager.register_router_reload_callback("test-strategy", test_callback)
        assert "test-strategy" in manager._router_reload_callbacks

    def test_hot_reload_router(self):
        """Test reloading a router."""
        from litellm_llmrouter.hot_reload import HotReloadManager

        manager = HotReloadManager()
        reload_count = [0]

        def test_callback():
            reload_count[0] += 1

        manager.register_router_reload_callback("test-strategy", test_callback)
        result = manager.reload_router("test-strategy")

        assert result["status"] == "success"
        assert "test-strategy" in result["reloaded"]
        assert reload_count[0] == 1
