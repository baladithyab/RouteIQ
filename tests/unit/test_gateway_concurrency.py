"""
Concurrency Tests for A2A and MCP Gateways
============================================

Tests thread safety of:
1. Singleton initialization (get_a2a_gateway, get_mcp_gateway)
2. Registry mutations (register/unregister operations)
3. Read operations returning consistent snapshots
4. No race conditions under concurrent access

Uses ThreadPoolExecutor to simulate high-concurrency FastAPI/uvicorn environment.
"""

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from types import MappingProxyType

import pytest


class TestA2AGatewayConcurrency:
    """Test A2A Gateway thread safety under concurrent access."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Reset gateway singleton and enable for testing."""
        # Reset singleton before each test
        from litellm_llmrouter.a2a_gateway import reset_a2a_gateway

        reset_a2a_gateway()
        os.environ["A2A_GATEWAY_ENABLED"] = "true"
        yield
        # Cleanup
        reset_a2a_gateway()
        os.environ.pop("A2A_GATEWAY_ENABLED", None)

    def test_singleton_concurrent_initialization(self):
        """
        Test that concurrent calls to get_a2a_gateway() return the same instance.
        
        Simulates multiple requests hitting the gateway simultaneously at startup.
        """
        from litellm_llmrouter.a2a_gateway import get_a2a_gateway, reset_a2a_gateway

        reset_a2a_gateway()  # Ensure fresh state

        instances = []
        num_threads = 50

        def get_instance():
            return get_a2a_gateway()

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(get_instance) for _ in range(num_threads)]
            for future in as_completed(futures):
                instances.append(future.result())

        # All instances should be the same object
        first_instance = instances[0]
        assert all(
            inst is first_instance for inst in instances
        ), "Singleton violated: different instances returned"

    def test_concurrent_register_unregister(self):
        """
        Test concurrent register and unregister operations don't cause race conditions.
        
        This hammers the registry with simultaneous add/remove operations.
        """
        from litellm_llmrouter.a2a_gateway import A2AAgent, get_a2a_gateway

        gateway = get_a2a_gateway()
        num_operations = 100
        errors = []

        def register_agent(idx: int):
            try:
                agent = A2AAgent(
                    agent_id=f"agent-{idx}",
                    name=f"Agent {idx}",
                    description=f"Test agent {idx}",
                    url=f"https://agent{idx}.example.com",
                    capabilities=["chat", "code"],
                )
                gateway.register_agent(agent)
            except Exception as e:
                errors.append(f"Register error {idx}: {e}")

        def unregister_agent(idx: int):
            try:
                gateway.unregister_agent(f"agent-{idx}")
            except Exception as e:
                errors.append(f"Unregister error {idx}: {e}")

        with ThreadPoolExecutor(max_workers=20) as executor:
            # Submit interleaved register/unregister operations
            futures = []
            for i in range(num_operations):
                futures.append(executor.submit(register_agent, i))
                # Some unregisters for agents that may or may not exist
                if i > 10:
                    futures.append(executor.submit(unregister_agent, i - 10))

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    errors.append(f"Future error: {e}")

        assert not errors, f"Concurrency errors: {errors}"

    def test_concurrent_list_while_mutating(self):
        """
        Test that list_agents returns consistent snapshots while mutations occur.
        
        Verifies no RuntimeError from dictionary changed during iteration.
        """
        from litellm_llmrouter.a2a_gateway import A2AAgent, get_a2a_gateway

        gateway = get_a2a_gateway()
        num_iterations = 50
        errors = []
        list_results = []

        def register_agent(idx: int):
            agent = A2AAgent(
                agent_id=f"list-agent-{idx}",
                name=f"List Agent {idx}",
                description=f"Test agent for list test {idx}",
                url=f"https://listagent{idx}.example.com",
                capabilities=["chat"],
            )
            try:
                gateway.register_agent(agent)
            except Exception as e:
                errors.append(f"Register error: {e}")

        def list_agents():
            try:
                agents = gateway.list_agents()
                # Iterate to ensure snapshot is stable
                count = 0
                for agent in agents:
                    count += 1
                    _ = agent.name
                list_results.append(count)
            except Exception as e:
                errors.append(f"List error: {e}")

        with ThreadPoolExecutor(max_workers=30) as executor:
            futures = []
            for i in range(num_iterations):
                futures.append(executor.submit(register_agent, i))
                futures.append(executor.submit(list_agents))
                futures.append(executor.submit(list_agents))

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    errors.append(f"Future error: {e}")

        assert not errors, f"Concurrency errors: {errors}"
        assert len(list_results) > 0, "No list results captured"

    def test_concurrent_discover_agents(self):
        """
        Test concurrent discover_agents with capability filtering.
        """
        from litellm_llmrouter.a2a_gateway import A2AAgent, get_a2a_gateway

        gateway = get_a2a_gateway()

        # Pre-register some agents with different capabilities
        capabilities_sets = [
            ["chat", "code"],
            ["chat"],
            ["code", "search"],
            ["search"],
        ]
        for i in range(20):
            agent = A2AAgent(
                agent_id=f"discover-agent-{i}",
                name=f"Discover Agent {i}",
                description=f"Agent for discovery test {i}",
                url=f"https://discoveragent{i}.example.com",
                capabilities=capabilities_sets[i % len(capabilities_sets)],
            )
            gateway.register_agent(agent)

        errors = []
        results = []

        def discover(capability: str):
            try:
                agents = gateway.discover_agents(capability=capability)
                results.append((capability, len(agents)))
            except Exception as e:
                errors.append(f"Discover error: {e}")

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = []
            for _ in range(50):
                for cap in ["chat", "code", "search"]:
                    futures.append(executor.submit(discover, cap))

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    errors.append(f"Future error: {e}")

        assert not errors, f"Concurrency errors: {errors}"
        assert len(results) == 150, f"Expected 150 results, got {len(results)}"

    def test_immutable_snapshot(self):
        """
        Test that get_agents_snapshot returns an immutable MappingProxyType.
        """
        from litellm_llmrouter.a2a_gateway import A2AAgent, get_a2a_gateway

        gateway = get_a2a_gateway()

        agent = A2AAgent(
            agent_id="snap-agent-1",
            name="Snapshot Agent",
            description="Agent for snapshot test",
            url="https://snapagent.example.com",
            capabilities=["chat"],
        )
        gateway.register_agent(agent)

        snapshot = gateway.get_agents_snapshot()
        assert isinstance(snapshot, MappingProxyType), "Snapshot should be immutable"
        assert "snap-agent-1" in snapshot

        # Verify immutability - should raise TypeError
        with pytest.raises(TypeError):
            snapshot["new-agent"] = agent  # type: ignore

    def test_get_agent_under_concurrent_unregister(self):
        """
        Test that get_agent returns None safely if agent is unregistered concurrently.
        """
        from litellm_llmrouter.a2a_gateway import A2AAgent, get_a2a_gateway

        gateway = get_a2a_gateway()

        # Register agents
        for i in range(50):
            agent = A2AAgent(
                agent_id=f"get-agent-{i}",
                name=f"Get Agent {i}",
                description=f"Agent for get test {i}",
                url=f"https://getagent{i}.example.com",
                capabilities=["chat"],
            )
            gateway.register_agent(agent)

        errors = []
        results = []

        def get_agent(idx: int):
            try:
                agent = gateway.get_agent(f"get-agent-{idx}")
                results.append(("get", idx, agent is not None))
            except Exception as e:
                errors.append(f"Get error: {e}")

        def unregister_agent(idx: int):
            try:
                gateway.unregister_agent(f"get-agent-{idx}")
                results.append(("unregister", idx, True))
            except Exception as e:
                errors.append(f"Unregister error: {e}")

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = []
            for i in range(50):
                futures.append(executor.submit(get_agent, i))
                futures.append(executor.submit(unregister_agent, i))
                futures.append(executor.submit(get_agent, i))

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    errors.append(f"Future error: {e}")

        assert not errors, f"Concurrency errors: {errors}"


class TestMCPGatewayConcurrency:
    """Test MCP Gateway thread safety under concurrent access."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Reset gateway singleton and enable for testing."""
        from litellm_llmrouter.mcp_gateway import reset_mcp_gateway

        reset_mcp_gateway()
        os.environ["MCP_GATEWAY_ENABLED"] = "true"
        # Disable HA sync for simpler testing
        os.environ["MCP_HA_SYNC_ENABLED"] = "false"
        yield
        reset_mcp_gateway()
        os.environ.pop("MCP_GATEWAY_ENABLED", None)
        os.environ.pop("MCP_HA_SYNC_ENABLED", None)

    def test_singleton_concurrent_initialization(self):
        """
        Test that concurrent calls to get_mcp_gateway() return the same instance.
        """
        from litellm_llmrouter.mcp_gateway import get_mcp_gateway, reset_mcp_gateway

        reset_mcp_gateway()

        instances = []
        num_threads = 50

        def get_instance():
            return get_mcp_gateway()

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(get_instance) for _ in range(num_threads)]
            for future in as_completed(futures):
                instances.append(future.result())

        first_instance = instances[0]
        assert all(
            inst is first_instance for inst in instances
        ), "Singleton violated: different instances returned"

    def test_concurrent_register_unregister(self):
        """
        Test concurrent register and unregister operations don't cause race conditions.
        """
        from litellm_llmrouter.mcp_gateway import MCPServer, MCPTransport, get_mcp_gateway

        gateway = get_mcp_gateway()
        num_operations = 100
        errors = []

        def register_server(idx: int):
            try:
                server = MCPServer(
                    server_id=f"server-{idx}",
                    name=f"Server {idx}",
                    url=f"https://server{idx}.example.com",
                    transport=MCPTransport.STREAMABLE_HTTP,
                    tools=[f"tool-{idx}"],
                    resources=[f"resource-{idx}"],
                )
                gateway.register_server(server)
            except Exception as e:
                errors.append(f"Register error {idx}: {e}")

        def unregister_server(idx: int):
            try:
                gateway.unregister_server(f"server-{idx}")
            except Exception as e:
                errors.append(f"Unregister error {idx}: {e}")

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = []
            for i in range(num_operations):
                futures.append(executor.submit(register_server, i))
                if i > 10:
                    futures.append(executor.submit(unregister_server, i - 10))

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    errors.append(f"Future error: {e}")

        assert not errors, f"Concurrency errors: {errors}"

    def test_concurrent_list_while_mutating(self):
        """
        Test that list_servers returns consistent snapshots while mutations occur.
        """
        from litellm_llmrouter.mcp_gateway import MCPServer, MCPTransport, get_mcp_gateway

        gateway = get_mcp_gateway()
        num_iterations = 50
        errors = []
        list_results = []

        def register_server(idx: int):
            server = MCPServer(
                server_id=f"list-server-{idx}",
                name=f"List Server {idx}",
                url=f"https://listserver{idx}.example.com",
                transport=MCPTransport.STREAMABLE_HTTP,
                tools=[f"tool-{idx}"],
            )
            try:
                gateway.register_server(server)
            except Exception as e:
                errors.append(f"Register error: {e}")

        def list_servers():
            try:
                servers = gateway.list_servers()
                count = 0
                for server in servers:
                    count += 1
                    _ = server.name
                list_results.append(count)
            except Exception as e:
                errors.append(f"List error: {e}")

        with ThreadPoolExecutor(max_workers=30) as executor:
            futures = []
            for i in range(num_iterations):
                futures.append(executor.submit(register_server, i))
                futures.append(executor.submit(list_servers))
                futures.append(executor.submit(list_servers))

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    errors.append(f"Future error: {e}")

        assert not errors, f"Concurrency errors: {errors}"
        assert len(list_results) > 0, "No list results captured"

    def test_concurrent_list_tools(self):
        """
        Test concurrent list_tools calls with concurrent mutations.
        """
        from litellm_llmrouter.mcp_gateway import MCPServer, MCPTransport, get_mcp_gateway

        gateway = get_mcp_gateway()

        # Pre-register some servers
        for i in range(20):
            server = MCPServer(
                server_id=f"tools-server-{i}",
                name=f"Tools Server {i}",
                url=f"https://toolsserver{i}.example.com",
                transport=MCPTransport.STREAMABLE_HTTP,
                tools=[f"tool-a-{i}", f"tool-b-{i}"],
            )
            gateway.register_server(server)

        errors = []
        results = []

        def list_tools():
            try:
                tools = gateway.list_tools()
                results.append(len(tools))
            except Exception as e:
                errors.append(f"List tools error: {e}")

        def list_resources():
            try:
                resources = gateway.list_resources()
                results.append(len(resources))
            except Exception as e:
                errors.append(f"List resources error: {e}")

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = []
            for _ in range(100):
                futures.append(executor.submit(list_tools))
                futures.append(executor.submit(list_resources))

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    errors.append(f"Future error: {e}")

        assert not errors, f"Concurrency errors: {errors}"
        assert len(results) == 200, f"Expected 200 results, got {len(results)}"

    def test_concurrent_tool_lookup(self):
        """
        Test concurrent find_server_for_tool and get_tool calls.
        """
        from litellm_llmrouter.mcp_gateway import (
            MCPServer,
            MCPToolDefinition,
            MCPTransport,
            get_mcp_gateway,
        )

        gateway = get_mcp_gateway()

        # Register servers with tools
        for i in range(20):
            server = MCPServer(
                server_id=f"lookup-server-{i}",
                name=f"Lookup Server {i}",
                url=f"https://lookupserver{i}.example.com",
                transport=MCPTransport.STREAMABLE_HTTP,
                tools=[f"lookup-tool-{i}"],
            )
            gateway.register_server(server)
            # Also register detailed tool definition
            tool_def = MCPToolDefinition(
                name=f"lookup-tool-{i}",
                description=f"Tool {i} for lookup test",
                input_schema={"type": "object", "properties": {"arg": {"type": "string"}}},
            )
            gateway.register_tool_definition(f"lookup-server-{i}", tool_def)

        errors = []
        results = []

        def find_tool(idx: int):
            try:
                server = gateway.find_server_for_tool(f"lookup-tool-{idx}")
                results.append(("find", idx, server is not None))
            except Exception as e:
                errors.append(f"Find tool error: {e}")

        def get_tool(idx: int):
            try:
                tool = gateway.get_tool(f"lookup-tool-{idx}")
                results.append(("get", idx, tool is not None))
            except Exception as e:
                errors.append(f"Get tool error: {e}")

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = []
            for _ in range(5):
                for i in range(20):
                    futures.append(executor.submit(find_tool, i))
                    futures.append(executor.submit(get_tool, i))

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    errors.append(f"Future error: {e}")

        assert not errors, f"Concurrency errors: {errors}"
        assert len(results) == 200, f"Expected 200 results, got {len(results)}"

    def test_immutable_snapshot(self):
        """
        Test that get_servers_snapshot returns an immutable MappingProxyType.
        """
        from litellm_llmrouter.mcp_gateway import MCPServer, MCPTransport, get_mcp_gateway

        gateway = get_mcp_gateway()

        server = MCPServer(
            server_id="snap-server-1",
            name="Snapshot Server",
            url="https://snapserver.example.com",
            transport=MCPTransport.STREAMABLE_HTTP,
            tools=["snap-tool"],
        )
        gateway.register_server(server)

        snapshot = gateway.get_servers_snapshot()
        assert isinstance(snapshot, MappingProxyType), "Snapshot should be immutable"
        assert "snap-server-1" in snapshot

        # Verify immutability - should raise TypeError
        with pytest.raises(TypeError):
            snapshot["new-server"] = server  # type: ignore

    def test_concurrent_register_tool_definition(self):
        """
        Test concurrent registration of tool definitions.
        """
        from litellm_llmrouter.mcp_gateway import (
            MCPServer,
            MCPToolDefinition,
            MCPTransport,
            get_mcp_gateway,
        )

        gateway = get_mcp_gateway()

        # Register a single server
        server = MCPServer(
            server_id="tooldef-server",
            name="Tool Def Server",
            url="https://tooldefserver.example.com",
            transport=MCPTransport.STREAMABLE_HTTP,
            tools=[],
        )
        gateway.register_server(server)

        errors = []
        successes = []

        def register_tool_def(idx: int):
            try:
                tool_def = MCPToolDefinition(
                    name=f"concurrent-tool-{idx}",
                    description=f"Concurrent tool {idx}",
                    input_schema={"type": "object"},
                )
                result = gateway.register_tool_definition("tooldef-server", tool_def)
                successes.append(result)
            except Exception as e:
                errors.append(f"Register tool def error: {e}")

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(register_tool_def, i) for i in range(50)]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    errors.append(f"Future error: {e}")

        assert not errors, f"Concurrency errors: {errors}"
        assert all(successes), "All tool definitions should register successfully"
        assert len(successes) == 50, f"Expected 50 successes, got {len(successes)}"

    def test_concurrent_get_registry(self):
        """
        Test concurrent get_registry calls with access group filtering.
        """
        from litellm_llmrouter.mcp_gateway import MCPServer, MCPTransport, get_mcp_gateway

        gateway = get_mcp_gateway()

        # Register servers with different access groups
        access_groups_sets = [
            ["group-a", "group-b"],
            ["group-a"],
            ["group-b", "group-c"],
            ["group-c"],
        ]
        for i in range(20):
            server = MCPServer(
                server_id=f"registry-server-{i}",
                name=f"Registry Server {i}",
                url=f"https://registryserver{i}.example.com",
                transport=MCPTransport.STREAMABLE_HTTP,
                tools=[f"registry-tool-{i}"],
                metadata={"access_groups": access_groups_sets[i % len(access_groups_sets)]},
            )
            gateway.register_server(server)

        errors = []
        results = []

        def get_registry(groups: list[str] | None):
            try:
                registry = gateway.get_registry(access_groups=groups)
                results.append((groups, registry["server_count"]))
            except Exception as e:
                errors.append(f"Get registry error: {e}")

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = []
            for _ in range(20):
                futures.append(executor.submit(get_registry, None))
                futures.append(executor.submit(get_registry, ["group-a"]))
                futures.append(executor.submit(get_registry, ["group-b"]))
                futures.append(executor.submit(get_registry, ["group-c"]))
                futures.append(executor.submit(get_registry, ["group-a", "group-b"]))

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    errors.append(f"Future error: {e}")

        assert not errors, f"Concurrency errors: {errors}"
        assert len(results) == 100, f"Expected 100 results, got {len(results)}"

    def test_concurrent_list_access_groups(self):
        """
        Test concurrent list_access_groups calls.
        """
        from litellm_llmrouter.mcp_gateway import MCPServer, MCPTransport, get_mcp_gateway

        gateway = get_mcp_gateway()

        # Register servers with access groups
        for i in range(10):
            server = MCPServer(
                server_id=f"access-server-{i}",
                name=f"Access Server {i}",
                url=f"https://accessserver{i}.example.com",
                transport=MCPTransport.STREAMABLE_HTTP,
                tools=[],
                metadata={"access_groups": [f"group-{i % 3}"]},
            )
            gateway.register_server(server)

        errors = []
        results = []

        def list_access_groups():
            try:
                groups = gateway.list_access_groups()
                results.append(len(groups))
            except Exception as e:
                errors.append(f"List access groups error: {e}")

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(list_access_groups) for _ in range(100)]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    errors.append(f"Future error: {e}")

        assert not errors, f"Concurrency errors: {errors}"
        assert len(results) == 100, f"Expected 100 results, got {len(results)}"
        # All results should show 3 access groups
        assert all(r == 3 for r in results), f"Inconsistent results: {set(results)}"


class TestConcurrentSingletonReset:
    """
    Test that singleton reset is safe but warns about concurrent usage.
    
    These tests verify the reset functions work correctly for testing purposes.
    """

    def test_a2a_reset_clears_singleton(self):
        """Test that reset_a2a_gateway clears the singleton."""
        from litellm_llmrouter.a2a_gateway import (
            get_a2a_gateway,
            reset_a2a_gateway,
        )

        os.environ["A2A_GATEWAY_ENABLED"] = "true"
        try:
            gateway1 = get_a2a_gateway()
            reset_a2a_gateway()
            gateway2 = get_a2a_gateway()

            assert gateway1 is not gateway2, "Reset should create a new instance"
        finally:
            reset_a2a_gateway()
            os.environ.pop("A2A_GATEWAY_ENABLED", None)

    def test_mcp_reset_clears_singleton(self):
        """Test that reset_mcp_gateway clears the singleton."""
        from litellm_llmrouter.mcp_gateway import (
            get_mcp_gateway,
            reset_mcp_gateway,
        )

        os.environ["MCP_GATEWAY_ENABLED"] = "true"
        os.environ["MCP_HA_SYNC_ENABLED"] = "false"
        try:
            gateway1 = get_mcp_gateway()
            reset_mcp_gateway()
            gateway2 = get_mcp_gateway()

            assert gateway1 is not gateway2, "Reset should create a new instance"
        finally:
            reset_mcp_gateway()
            os.environ.pop("MCP_GATEWAY_ENABLED", None)
            os.environ.pop("MCP_HA_SYNC_ENABLED", None)


class TestStressConditions:
    """
    Stress tests for high-contention scenarios.
    """

    def test_high_contention_a2a_register_same_agent(self):
        """
        Test multiple threads registering the same agent ID concurrently.
        
        The last write should win, but no exceptions should occur.
        """
        from litellm_llmrouter.a2a_gateway import (
            A2AAgent,
            get_a2a_gateway,
            reset_a2a_gateway,
        )

        reset_a2a_gateway()
        os.environ["A2A_GATEWAY_ENABLED"] = "true"

        try:
            gateway = get_a2a_gateway()
            errors = []

            def register_agent(version: int):
                try:
                    agent = A2AAgent(
                        agent_id="contested-agent",
                        name=f"Contested Agent v{version}",
                        description=f"Version {version}",
                        url="https://contested.example.com",
                        capabilities=["chat"],
                    )
                    gateway.register_agent(agent)
                except Exception as e:
                    errors.append(f"Register error: {e}")

            with ThreadPoolExecutor(max_workers=20) as executor:
                futures = [executor.submit(register_agent, i) for i in range(100)]
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        errors.append(f"Future error: {e}")

            assert not errors, f"Concurrency errors: {errors}"
            # Should have exactly one agent
            assert len(gateway.list_agents()) == 1

        finally:
            reset_a2a_gateway()
            os.environ.pop("A2A_GATEWAY_ENABLED", None)

    def test_high_contention_mcp_register_same_server(self):
        """
        Test multiple threads registering the same server ID concurrently.
        """
        from litellm_llmrouter.mcp_gateway import (
            MCPServer,
            MCPTransport,
            get_mcp_gateway,
            reset_mcp_gateway,
        )

        reset_mcp_gateway()
        os.environ["MCP_GATEWAY_ENABLED"] = "true"
        os.environ["MCP_HA_SYNC_ENABLED"] = "false"

        try:
            gateway = get_mcp_gateway()
            errors = []

            def register_server(version: int):
                try:
                    server = MCPServer(
                        server_id="contested-server",
                        name=f"Contested Server v{version}",
                        url="https://contested.example.com",
                        transport=MCPTransport.STREAMABLE_HTTP,
                        tools=[f"tool-v{version}"],
                    )
                    gateway.register_server(server)
                except Exception as e:
                    errors.append(f"Register error: {e}")

            with ThreadPoolExecutor(max_workers=20) as executor:
                futures = [executor.submit(register_server, i) for i in range(100)]
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        errors.append(f"Future error: {e}")

            assert not errors, f"Concurrency errors: {errors}"
            # Should have exactly one server
            assert len(gateway.list_servers()) == 1

        finally:
            reset_mcp_gateway()
            os.environ.pop("MCP_GATEWAY_ENABLED", None)
            os.environ.pop("MCP_HA_SYNC_ENABLED", None)

    def test_rapid_register_unregister_cycle(self):
        """
        Test rapid register/unregister cycles for the same IDs.
        """
        from litellm_llmrouter.a2a_gateway import (
            A2AAgent,
            get_a2a_gateway,
            reset_a2a_gateway,
        )

        reset_a2a_gateway()
        os.environ["A2A_GATEWAY_ENABLED"] = "true"

        try:
            gateway = get_a2a_gateway()
            errors = []

            def cycle(agent_id: str, iterations: int):
                for i in range(iterations):
                    try:
                        agent = A2AAgent(
                            agent_id=agent_id,
                            name=f"Cycle Agent {agent_id}",
                            description="Rapid cycling",
                            url="https://cycle.example.com",
                            capabilities=["chat"],
                        )
                        gateway.register_agent(agent)
                        gateway.unregister_agent(agent_id)
                    except Exception as e:
                        errors.append(f"Cycle error for {agent_id} iter {i}: {e}")

            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = []
                for i in range(10):
                    futures.append(executor.submit(cycle, f"cycle-agent-{i}", 50))

                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        errors.append(f"Future error: {e}")

            assert not errors, f"Concurrency errors: {errors}"

        finally:
            reset_a2a_gateway()
            os.environ.pop("A2A_GATEWAY_ENABLED", None)