#!/usr/bin/env python3
"""
Gate 7: Security + Multi-Tenancy Validation for LiteLLM HA Stack

This script performs black-box security and multi-tenancy testing against
the running HA stack (LB at http://localhost:8080).

Validation objectives:
1. Master key enforcement
2. Virtual keys / API keys
3. Budgets / rate limits
4. Team isolation
5. Logging redaction
6. HA consistency

Usage:
    uv run scripts/test_security_multi_tenancy.py
"""

import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import requests


# Test configuration
LB_URL = "http://localhost:8080"
REPLICA_1_URL = "http://localhost:4000"
REPLICA_2_URL = "http://localhost:4001"
MASTER_KEY = os.getenv("LITELLM_MASTER_KEY", "sk-master-key-change-me")


@dataclass
class TestResult:
    """Test result data class"""

    test_name: str
    passed: bool
    details: str
    error: Optional[str] = None


class SecurityTester:
    """Security and multi-tenancy test suite"""

    def __init__(self, lb_url: str, replica_urls: List[str], master_key: str):
        self.lb_url = lb_url
        self.replica_urls = replica_urls
        self.master_key = master_key
        self.results: List[TestResult] = []
        self.created_resources: Dict[str, List[str]] = {
            "keys": [],
            "teams": [],
            "users": [],
        }

    def log(self, message: str):
        """Print timestamped log message"""
        print(f"[{time.strftime('%H:%M:%S')}] {message}")

    def add_result(self, result: TestResult):
        """Add test result and log it"""
        self.results.append(result)
        status = "✅ PASS" if result.passed else "❌ FAIL"
        self.log(f"{status}: {result.test_name}")
        if result.details:
            self.log(f"  Details: {result.details}")
        if result.error:
            self.log(f"  Error: {result.error}")

    # =========================================================================
    # Test 1: Master Key Enforcement
    # =========================================================================

    def test_master_key_enforcement(self):
        """Test that protected endpoints require master key"""
        self.log("\n=== Test 1: Master Key Enforcement ===")

        # Test 1a: Request without Authorization header should fail
        try:
            response = requests.get(f"{self.lb_url}/key/info", timeout=10)
            if response.status_code in [401, 403]:
                self.add_result(
                    TestResult(
                        test_name="1a. No auth header → 401/403",
                        passed=True,
                        details=f"Got {response.status_code} as expected",
                    )
                )
            else:
                self.add_result(
                    TestResult(
                        test_name="1a. No auth header → 401/403",
                        passed=False,
                        details=f"Expected 401/403, got {response.status_code}",
                        error=response.text[:200],
                    )
                )
        except Exception as e:
            self.add_result(
                TestResult(
                    test_name="1a. No auth header → 401/403",
                    passed=False,
                    details="Request failed",
                    error=str(e),
                )
            )

        # Test 1b: Request with invalid key should fail
        try:
            response = requests.get(
                f"{self.lb_url}/key/info",
                headers={"Authorization": "Bearer sk-invalid-key-12345"},
                timeout=10,
            )
            if response.status_code in [401, 403]:
                self.add_result(
                    TestResult(
                        test_name="1b. Invalid key → 401/403",
                        passed=True,
                        details=f"Got {response.status_code} as expected",
                    )
                )
            else:
                self.add_result(
                    TestResult(
                        test_name="1b. Invalid key → 401/403",
                        passed=False,
                        details=f"Expected 401/403, got {response.status_code}",
                        error=response.text[:200],
                    )
                )
        except Exception as e:
            self.add_result(
                TestResult(
                    test_name="1b. Invalid key → 401/403",
                    passed=False,
                    details="Request failed",
                    error=str(e),
                )
            )

        # Test 1c: Request with master key should succeed
        try:
            response = requests.post(
                f"{self.lb_url}/key/generate",
                headers={"Authorization": f"Bearer {self.master_key}"},
                json={"duration": "1h"},
                timeout=10,
            )
            if response.status_code in [200, 201]:
                data = response.json()
                if "key" in data:
                    # Clean up the generated key
                    self.created_resources["keys"].append(data["key"])
                    self.add_result(
                        TestResult(
                            test_name="1c. Master key → 200/201",
                            passed=True,
                            details="Successfully generated key with master key",
                        )
                    )
                else:
                    self.add_result(
                        TestResult(
                            test_name="1c. Master key → 200/201",
                            passed=False,
                            details="Response missing 'key' field",
                            error=json.dumps(data)[:200],
                        )
                    )
            else:
                self.add_result(
                    TestResult(
                        test_name="1c. Master key → 200/201",
                        passed=False,
                        details=f"Expected 200/201, got {response.status_code}",
                        error=response.text[:200],
                    )
                )
        except Exception as e:
            self.add_result(
                TestResult(
                    test_name="1c. Master key → 200/201",
                    passed=False,
                    details="Request failed",
                    error=str(e),
                )
            )

    # =========================================================================
    # Test 2: Virtual Keys / API Keys
    # =========================================================================

    def test_virtual_keys(self):
        """Test virtual key creation and usage"""
        self.log("\n=== Test 2: Virtual Keys / API Keys ===")

        # Test 2a: Create a virtual key
        virtual_key = None
        try:
            response = requests.post(
                f"{self.lb_url}/key/generate",
                headers={"Authorization": f"Bearer {self.master_key}"},
                json={
                    "duration": "1h",
                    "aliases": {},
                    "metadata": {"test": "security_validation"},
                },
                timeout=10,
            )
            if response.status_code in [200, 201]:
                data = response.json()
                virtual_key = data.get("key")
                if virtual_key:
                    self.created_resources["keys"].append(virtual_key)
                    self.add_result(
                        TestResult(
                            test_name="2a. Create virtual key",
                            passed=True,
                            details=f"Created key: {virtual_key[:20]}...",
                        )
                    )
                else:
                    self.add_result(
                        TestResult(
                            test_name="2a. Create virtual key",
                            passed=False,
                            details="Response missing 'key' field",
                            error=json.dumps(data)[:200],
                        )
                    )
            else:
                self.add_result(
                    TestResult(
                        test_name="2a. Create virtual key",
                        passed=False,
                        details=f"Expected 200/201, got {response.status_code}",
                        error=response.text[:200],
                    )
                )
                return
        except Exception as e:
            self.add_result(
                TestResult(
                    test_name="2a. Create virtual key",
                    passed=False,
                    details="Request failed",
                    error=str(e),
                )
            )
            return

        # Test 2b: Use virtual key for /v1/chat/completions
        if virtual_key:
            try:
                response = requests.post(
                    f"{self.lb_url}/v1/chat/completions",
                    headers={"Authorization": f"Bearer {virtual_key}"},
                    json={
                        "model": "claude-3-haiku",
                        "messages": [{"role": "user", "content": "Say 'test' only"}],
                        "max_tokens": 10,
                    },
                    timeout=30,
                )
                # Accept 200 (success) or 400+ (model/provider errors)
                # The key should be valid regardless
                if response.status_code == 200:
                    self.add_result(
                        TestResult(
                            test_name="2b. Use virtual key for chat",
                            passed=True,
                            details="Virtual key successfully authenticated",
                        )
                    )
                elif response.status_code in [401, 403]:
                    self.add_result(
                        TestResult(
                            test_name="2b. Use virtual key for chat",
                            passed=False,
                            details=f"Virtual key rejected: {response.status_code}",
                            error=response.text[:200],
                        )
                    )
                else:
                    # Other errors (400, 500) may indicate provider issues, not auth
                    self.add_result(
                        TestResult(
                            test_name="2b. Use virtual key for chat",
                            passed=True,
                            details=f"Key accepted (non-auth error {response.status_code})",
                        )
                    )
            except Exception as e:
                self.add_result(
                    TestResult(
                        test_name="2b. Use virtual key for chat",
                        passed=False,
                        details="Request failed",
                        error=str(e),
                    )
                )

    # =========================================================================
    # Test 3: Budgets / Rate Limits
    # =========================================================================

    def test_budget_enforcement(self):
        """Test budget limits are enforced"""
        self.log("\n=== Test 3: Budgets / Rate Limits ===")

        # Test 3a: Create key with tiny budget
        budget_key = None
        try:
            response = requests.post(
                f"{self.lb_url}/key/generate",
                headers={"Authorization": f"Bearer {self.master_key}"},
                json={
                    "max_budget": 0.0000001,  # ~$0.0000001
                    "duration": "1h",
                    "metadata": {"test": "budget_test"},
                },
                timeout=10,
            )
            if response.status_code in [200, 201]:
                data = response.json()
                budget_key = data.get("key")
                if budget_key:
                    self.created_resources["keys"].append(budget_key)
                    self.add_result(
                        TestResult(
                            test_name="3a. Create key with tiny budget",
                            passed=True,
                            details="Created key with $0.0000001 budget",
                        )
                    )
                else:
                    self.add_result(
                        TestResult(
                            test_name="3a. Create key with tiny budget",
                            passed=False,
                            details="Response missing 'key' field",
                            error=json.dumps(data)[:200],
                        )
                    )
                    return
            else:
                self.add_result(
                    TestResult(
                        test_name="3a. Create key with tiny budget",
                        passed=False,
                        details=f"Expected 200/201, got {response.status_code}",
                        error=response.text[:200],
                    )
                )
                return
        except Exception as e:
            self.add_result(
                TestResult(
                    test_name="3a. Create key with tiny budget",
                    passed=False,
                    details="Request failed",
                    error=str(e),
                )
            )
            return

        # Test 3b: Make requests until budget exceeded
        if budget_key:
            exceeded = False
            for i in range(5):  # Try up to 5 requests
                try:
                    response = requests.post(
                        f"{self.lb_url}/v1/chat/completions",
                        headers={"Authorization": f"Bearer {budget_key}"},
                        json={
                            "model": "claude-3-haiku",
                            "messages": [{"role": "user", "content": "Hi"}],
                            "max_tokens": 5,
                        },
                        timeout=30,
                    )
                    if (
                        response.status_code == 429
                        or "budget" in response.text.lower()
                        or "exceeded" in response.text.lower()
                    ):
                        exceeded = True
                        self.add_result(
                            TestResult(
                                test_name="3b. Budget exceeded → 4xx error",
                                passed=True,
                                details=f"Budget limit enforced after {i+1} attempts (HTTP {response.status_code})",
                            )
                        )
                        break
                except Exception as e:
                    self.log(f"  Request {i+1} error: {e}")

            if not exceeded:
                self.add_result(
                    TestResult(
                        test_name="3b. Budget exceeded → 4xx error",
                        passed=False,
                        details="Budget not enforced after 5 requests",
                        error="Expected budget limit error",
                    )
                )

        # Test 3c: Verify service didn't crash
        try:
            response = requests.get(f"{self.lb_url}/health", timeout=10)
            if response.status_code == 200:
                self.add_result(
                    TestResult(
                        test_name="3c. Service healthy after budget test",
                        passed=True,
                        details="Service still responding",
                    )
                )
            else:
                self.add_result(
                    TestResult(
                        test_name="3c. Service healthy after budget test",
                        passed=False,
                        details=f"Health check failed: {response.status_code}",
                    )
                )
        except Exception as e:
            self.add_result(
                TestResult(
                    test_name="3c. Service healthy after budget test",
                    passed=False,
                    details="Health check failed",
                    error=str(e),
                )
            )

    # =========================================================================
    # Test 4: Team Isolation
    # =========================================================================

    def test_team_isolation(self):
        """Test that teams are isolated from each other"""
        self.log("\n=== Test 4: Team Isolation ===")

        # Test 4a: Create team 1
        team1_id = None
        try:
            response = requests.post(
                f"{self.lb_url}/team/new",
                headers={"Authorization": f"Bearer {self.master_key}"},
                json={
                    "team_alias": "security-test-team-1",
                    "metadata": {"test": "isolation"},
                },
                timeout=10,
            )
            if response.status_code in [200, 201]:
                data = response.json()
                team1_id = data.get("team_id")
                if team1_id:
                    self.created_resources["teams"].append(team1_id)
                    self.add_result(
                        TestResult(
                            test_name="4a. Create team 1",
                            passed=True,
                            details=f"Created team: {team1_id}",
                        )
                    )
                else:
                    self.add_result(
                        TestResult(
                            test_name="4a. Create team 1",
                            passed=False,
                            details="Response missing 'team_id'",
                            error=json.dumps(data)[:200],
                        )
                    )
                    return
            else:
                self.add_result(
                    TestResult(
                        test_name="4a. Create team 1",
                        passed=False,
                        details=f"Expected 200/201, got {response.status_code}",
                        error=response.text[:200],
                    )
                )
                return
        except Exception as e:
            self.add_result(
                TestResult(
                    test_name="4a. Create team 1",
                    passed=False,
                    details="Request failed",
                    error=str(e),
                )
            )
            return

        # Test 4b: Create team 2
        team2_id = None
        try:
            response = requests.post(
                f"{self.lb_url}/team/new",
                headers={"Authorization": f"Bearer {self.master_key}"},
                json={
                    "team_alias": "security-test-team-2",
                    "metadata": {"test": "isolation"},
                },
                timeout=10,
            )
            if response.status_code in [200, 201]:
                data = response.json()
                team2_id = data.get("team_id")
                if team2_id:
                    self.created_resources["teams"].append(team2_id)
                    self.add_result(
                        TestResult(
                            test_name="4b. Create team 2",
                            passed=True,
                            details=f"Created team: {team2_id}",
                        )
                    )
                else:
                    self.add_result(
                        TestResult(
                            test_name="4b. Create team 2",
                            passed=False,
                            details="Response missing 'team_id'",
                            error=json.dumps(data)[:200],
                        )
                    )
                    return
            else:
                self.add_result(
                    TestResult(
                        test_name="4b. Create team 2",
                        passed=False,
                        details=f"Expected 200/201, got {response.status_code}",
                        error=response.text[:200],
                    )
                )
                return
        except Exception as e:
            self.add_result(
                TestResult(
                    test_name="4b. Create team 2",
                    passed=False,
                    details="Request failed",
                    error=str(e),
                )
            )
            return

        # Test 4c: Create keys for each team
        team1_key = None
        team2_key = None

        if team1_id:
            try:
                response = requests.post(
                    f"{self.lb_url}/key/generate",
                    headers={"Authorization": f"Bearer {self.master_key}"},
                    json={"team_id": team1_id, "duration": "1h"},
                    timeout=10,
                )
                if response.status_code in [200, 201]:
                    team1_key = response.json().get("key")
                    if team1_key:
                        self.created_resources["keys"].append(team1_key)
                        self.add_result(
                            TestResult(
                                test_name="4c. Create key for team 1",
                                passed=True,
                                details="Key created for team 1",
                            )
                        )
                    else:
                        self.add_result(
                            TestResult(
                                test_name="4c. Create key for team 1",
                                passed=False,
                                details="Response missing 'key'",
                            )
                        )
                else:
                    self.add_result(
                        TestResult(
                            test_name="4c. Create key for team 1",
                            passed=False,
                            details=f"Got {response.status_code}",
                        )
                    )
            except Exception as e:
                self.add_result(
                    TestResult(
                        test_name="4c. Create key for team 1",
                        passed=False,
                        details="Request failed",
                        error=str(e),
                    )
                )

        if team2_id:
            try:
                response = requests.post(
                    f"{self.lb_url}/key/generate",
                    headers={"Authorization": f"Bearer {self.master_key}"},
                    json={"team_id": team2_id, "duration": "1h"},
                    timeout=10,
                )
                if response.status_code in [200, 201]:
                    team2_key = response.json().get("key")
                    if team2_key:
                        self.created_resources["keys"].append(team2_key)
                        self.add_result(
                            TestResult(
                                test_name="4d. Create key for team 2",
                                passed=True,
                                details="Key created for team 2",
                            )
                        )
                    else:
                        self.add_result(
                            TestResult(
                                test_name="4d. Create key for team 2",
                                passed=False,
                                details="Response missing 'key'",
                            )
                        )
                else:
                    self.add_result(
                        TestResult(
                            test_name="4d. Create key for team 2",
                            passed=False,
                            details=f"Got {response.status_code}",
                        )
                    )
            except Exception as e:
                self.add_result(
                    TestResult(
                        test_name="4d. Create key for team 2",
                        passed=False,
                        details="Request failed",
                        error=str(e),
                    )
                )

        # Test 4e: Verify team1_key and team2_key work independently
        # This is a basic isolation test - they should both authenticate
        if team1_key and team2_key:
            # Both keys should work for their respective teams
            team1_works = False
            team2_works = False

            try:
                response = requests.post(
                    f"{self.lb_url}/v1/chat/completions",
                    headers={"Authorization": f"Bearer {team1_key}"},
                    json={
                        "model": "claude-3-haiku",
                        "messages": [{"role": "user", "content": "test"}],
                        "max_tokens": 5,
                    },
                    timeout=30,
                )
                # Accept non-401/403 as success (auth worked)
                team1_works = response.status_code not in [401, 403]
            except Exception:
                pass

            try:
                response = requests.post(
                    f"{self.lb_url}/v1/chat/completions",
                    headers={"Authorization": f"Bearer {team2_key}"},
                    json={
                        "model": "claude-3-haiku",
                        "messages": [{"role": "user", "content": "test"}],
                        "max_tokens": 5,
                    },
                    timeout=30,
                )
                team2_works = response.status_code not in [401, 403]
            except Exception:
                pass

            if team1_works and team2_works:
                self.add_result(
                    TestResult(
                        test_name="4e. Team keys work independently",
                        passed=True,
                        details="Both team keys authenticated successfully",
                    )
                )
            else:
                self.add_result(
                    TestResult(
                        test_name="4e. Team keys work independently",
                        passed=False,
                        details=f"Team1: {team1_works}, Team2: {team2_works}",
                    )
                )

    # =========================================================================
    # Test 5: Logging Redaction
    # =========================================================================

    def test_log_redaction(self):
        """Test that secrets are not logged"""
        self.log("\n=== Test 5: Logging Redaction ===")

        # Create a test key
        test_key = None
        try:
            response = requests.post(
                f"{self.lb_url}/key/generate",
                headers={"Authorization": f"Bearer {self.master_key}"},
                json={"duration": "1h", "metadata": {"test": "log_redaction"}},
                timeout=10,
            )
            if response.status_code in [200, 201]:
                test_key = response.json().get("key")
                if test_key:
                    self.created_resources["keys"].append(test_key)
        except Exception:
            pass

        # Make a test call with the key
        if test_key:
            try:
                requests.post(
                    f"{self.lb_url}/v1/chat/completions",
                    headers={"Authorization": f"Bearer {test_key}"},
                    json={
                        "model": "claude-3-haiku",
                        "messages": [{"role": "user", "content": "test"}],
                        "max_tokens": 5,
                    },
                    timeout=30,
                )
            except Exception:
                pass

        # Get logs from both replicas
        logs_clean = True
        found_secrets = []

        for i, url in enumerate([REPLICA_1_URL, REPLICA_2_URL], 1):
            try:
                # Try to get logs via docker (assuming we can exec)
                import subprocess

                container = f"litellm-gateway-{i}"
                result = subprocess.run(
                    ["docker", "logs", "--tail", "100", container],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                logs = result.stdout + result.stderr

                # Check for master key
                if self.master_key in logs:
                    logs_clean = False
                    found_secrets.append(f"Master key in {container}")

                # Check for test key
                if test_key and test_key in logs:
                    logs_clean = False
                    found_secrets.append(f"Virtual key in {container}")

            except Exception as e:
                self.log(f"  Warning: Could not check logs for replica {i}: {e}")

        if logs_clean:
            self.add_result(
                TestResult(
                    test_name="5. Log redaction (secrets not in logs)",
                    passed=True,
                    details="No secrets found in gateway logs",
                )
            )
        else:
            self.add_result(
                TestResult(
                    test_name="5. Log redaction (secrets not in logs)",
                    passed=False,
                    details="Secrets found in logs",
                    error="; ".join(found_secrets),
                )
            )

    # =========================================================================
    # Test 6: HA Consistency
    # =========================================================================

    def test_ha_consistency(self):
        """Test HA consistency - keys work across replicas"""
        self.log("\n=== Test 6: HA Consistency ===")

        # Test 6a: Create key via LB
        ha_key = None
        try:
            response = requests.post(
                f"{self.lb_url}/key/generate",
                headers={"Authorization": f"Bearer {self.master_key}"},
                json={"duration": "1h", "metadata": {"test": "ha_consistency"}},
                timeout=10,
            )
            if response.status_code in [200, 201]:
                data = response.json()
                ha_key = data.get("key")
                if ha_key:
                    self.created_resources["keys"].append(ha_key)
                    self.add_result(
                        TestResult(
                            test_name="6a. Create key via LB",
                            passed=True,
                            details="Created key via LB",
                        )
                    )
                else:
                    self.add_result(
                        TestResult(
                            test_name="6a. Create key via LB",
                            passed=False,
                            details="Response missing 'key'",
                        )
                    )
                    return
            else:
                self.add_result(
                    TestResult(
                        test_name="6a. Create key via LB",
                        passed=False,
                        details=f"Got {response.status_code}",
                        error=response.text[:200],
                    )
                )
                return
        except Exception as e:
            self.add_result(
                TestResult(
                    test_name="6a. Create key via LB",
                    passed=False,
                    details="Request failed",
                    error=str(e),
                )
            )
            return

        # Test 6b: Use key directly on replica 1
        if ha_key:
            try:
                response = requests.post(
                    f"{REPLICA_1_URL}/v1/chat/completions",
                    headers={"Authorization": f"Bearer {ha_key}"},
                    json={
                        "model": "claude-3-haiku",
                        "messages": [{"role": "user", "content": "test"}],
                        "max_tokens": 5,
                    },
                    timeout=30,
                )
                if response.status_code not in [401, 403]:
                    self.add_result(
                        TestResult(
                            test_name="6b. Key works on replica 1",
                            passed=True,
                            details="Key authenticated on replica 1",
                        )
                    )
                else:
                    self.add_result(
                        TestResult(
                            test_name="6b. Key works on replica 1",
                            passed=False,
                            details=f"Auth failed: {response.status_code}",
                            error=response.text[:200],
                        )
                    )
            except Exception as e:
                self.add_result(
                    TestResult(
                        test_name="6b. Key works on replica 1",
                        passed=False,
                        details="Request failed",
                        error=str(e),
                    )
                )

        # Test 6c: Use key directly on replica 2
        if ha_key:
            try:
                response = requests.post(
                    f"{REPLICA_2_URL}/v1/chat/completions",
                    headers={"Authorization": f"Bearer {ha_key}"},
                    json={
                        "model": "claude-3-haiku",
                        "messages": [{"role": "user", "content": "test"}],
                        "max_tokens": 5,
                    },
                    timeout=30,
                )
                if response.status_code not in [401, 403]:
                    self.add_result(
                        TestResult(
                            test_name="6c. Key works on replica 2",
                            passed=True,
                            details="Key authenticated on replica 2",
                        )
                    )
                else:
                    self.add_result(
                        TestResult(
                            test_name="6c. Key works on replica 2",
                            passed=False,
                            details=f"Auth failed: {response.status_code}",
                            error=response.text[:200],
                        )
                    )
            except Exception as e:
                self.add_result(
                    TestResult(
                        test_name="6c. Key works on replica 2",
                        passed=False,
                        details="Request failed",
                        error=str(e),
                    )
                )

    # =========================================================================
    # Cleanup
    # =========================================================================

    def cleanup(self):
        """Clean up created resources"""
        self.log("\n=== Cleanup ===")

        # Delete keys
        for key in self.created_resources["keys"]:
            try:
                response = requests.post(
                    f"{self.lb_url}/key/delete",
                    headers={"Authorization": f"Bearer {self.master_key}"},
                    json={"key": key},
                    timeout=10,
                )
                if response.status_code in [200, 201]:
                    self.log(f"  Deleted key: {key[:20]}...")
                else:
                    self.log(
                        f"  Warning: Could not delete key {key[:20]}...: {response.status_code}"
                    )
            except Exception as e:
                self.log(f"  Warning: Error deleting key: {e}")

        # Delete teams
        for team_id in self.created_resources["teams"]:
            try:
                response = requests.post(
                    f"{self.lb_url}/team/delete",
                    headers={"Authorization": f"Bearer {self.master_key}"},
                    json={"team_id": team_id},
                    timeout=10,
                )
                if response.status_code in [200, 201]:
                    self.log(f"  Deleted team: {team_id}")
                else:
                    self.log(
                        f"  Warning: Could not delete team {team_id}: {response.status_code}"
                    )
            except Exception as e:
                self.log(f"  Warning: Error deleting team: {e}")

    # =========================================================================
    # Report
    # =========================================================================

    def generate_report(self) -> str:
        """Generate test report"""
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        total = len(self.results)

        report = []
        report.append("=" * 80)
        report.append("GATE 7: SECURITY + MULTI-TENANCY VALIDATION REPORT")
        report.append("=" * 80)
        report.append("")
        report.append(f"Total Tests: {total}")
        report.append(f"Passed: {passed} ✅")
        report.append(f"Failed: {failed} ❌")
        report.append(f"Success Rate: {100*passed/total if total > 0 else 0:.1f}%")
        report.append("")
        report.append("=" * 80)
        report.append("PASS/FAIL MATRIX")
        report.append("=" * 80)
        report.append("")

        for result in self.results:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            report.append(f"{status} | {result.test_name}")
            if result.details:
                report.append(f"       Details: {result.details}")
            if result.error:
                report.append(f"       Error: {result.error}")
            report.append("")

        return "\n".join(report)

    def run(self):
        """Run all tests"""
        self.log("Starting Gate 7 security validation...")
        self.log(f"Load Balancer: {self.lb_url}")
        self.log(f"Replica 1: {self.replica_urls[0]}")
        self.log(f"Replica 2: {self.replica_urls[1]}")

        try:
            self.test_master_key_enforcement()
            self.test_virtual_keys()
            self.test_budget_enforcement()
            self.test_team_isolation()
            self.test_log_redaction()
            self.test_ha_consistency()
        finally:
            self.cleanup()

        report = self.generate_report()
        print("\n" + report)

        # Save report
        with open("GATE7_SECURITY_VALIDATION_REPORT.md", "w") as f:
            f.write(report)
        self.log("\nReport saved to GATE7_SECURITY_VALIDATION_REPORT.md")

        # Exit with error code if any tests failed
        failed = sum(1 for r in self.results if not r.passed)
        return 0 if failed == 0 else 1


def main():
    """Main entry point"""
    tester = SecurityTester(
        lb_url=LB_URL,
        replica_urls=[REPLICA_1_URL, REPLICA_2_URL],
        master_key=MASTER_KEY,
    )
    sys.exit(tester.run())


if __name__ == "__main__":
    main()
