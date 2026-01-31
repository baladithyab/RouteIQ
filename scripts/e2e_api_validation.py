#!/usr/bin/env python3
"""
E2E API Validation Script for RouteIQ Gateway.

Standalone script to validate API endpoints against a running gateway.
Produces a detailed validation report with evidence.

Usage:
    python scripts/e2e_api_validation.py
    
Environment Variables:
    GATEWAY_URL: Gateway base URL (default: http://localhost:4010)
    ADMIN_KEY: Admin API key (default: sk-test-master-key)
"""

import os
import sys
import json
import requests
from typing import Dict, List, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict, field


@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    passed: bool
    status_code: int =  0
    response_body: Dict = None
    error: str = None
    details: str = None


@dataclass
class ValidationReport:
    """Overall validation report."""
    timestamp: str
    gateway_url: str
    tests_run: int = 0
    tests_passed: int = 0
    tests_failed: int = 0
    results: List[TestResult] = field(default_factory=list)


class E2EValidator:
    """E2E API validator."""
    
    def __init__(self, gateway_url: str, admin_key: str, timeout: int = 5):
        self.gateway_url = gateway_url.rstrip('/')
        self.admin_key = admin_key
        self.timeout = timeout
        self.report = ValidationReport(
            timestamp=datetime.utcnow().isoformat() + 'Z',
            gateway_url=gateway_url
        )
    
    def run_all_tests(self):
        """Run all E2E validation tests."""
        print(f"=== E2E API Validation ===")
        print(f"Gateway: {self.gateway_url}")
        print(f"Timestamp: {self.report.timestamp}")
        print()
        
        # Test suites
        self.test_authz_boundary()
        self.test_ssrf_protection()
        self.test_skills_discovery()
        self.test_health_endpoints()
        self.test_mcp_endpoints()
        
        # Calculate totals
        self.report.tests_run = len(self.report.results)
        self.report.tests_passed = sum(1 for r in self.report.results if r.passed)
        self.report.tests_failed = self.report.tests_run - self.report.tests_passed
        
        return self.report
    
    def add_result(self, result: TestResult):
        """Add test result and print summary."""
        self.report.results.append(result)
        status = "✅ PASS" if result.passed else "❌ FAIL"
        print(f"{status}: {result.name}")
        if result.details:
            print(f"  {result.details}")
        if not result.passed and result.error:
            print(f"  Error: {result.error}")
        print()
    
    def test_authz_boundary(self):
        """Test admin vs invalid key authorization."""
        print("## 1. Admin vs Invalid Key AuthZ Boundary")
        print()
        
        # Test 1: Invalid key should be rejected on /router/reload
        try:
            response = requests.post(
                f"{self.gateway_url}/router/reload",
                headers={"Authorization": "Bearer sk-invalid-key-12345"},
                timeout=self.timeout
            )
            passed = response.status_code in [401, 403]
            self.add_result(TestResult(
                name="POST /router/reload with invalid key",
                passed=passed,
                status_code=response.status_code,
                response_body=response.json() if response.text else {},
                details=f"Expected 401/403, got {response.status_code}"
            ))
        except Exception as e:
            self.add_result(TestResult(
                name="POST /router/reload with invalid key",
                passed=False,
                error=str(e)
            ))
        
        # Test 2: Admin key should be accepted on /router/reload
        try:
            response = requests.post(
                f"{self.gateway_url}/router/reload",
                headers={"Authorization": f"Bearer {self.admin_key}"},
                timeout=self.timeout
            )
            passed = response.status_code == 200
            self.add_result(TestResult(
                name="POST /router/reload with admin key",
                passed=passed,
                status_code=response.status_code,
                response_body=response.json() if response.text else {},
                details=f"Expected 200, got {response.status_code}"
            ))
        except Exception as e:
            self.add_result(TestResult(
                name="POST /router/reload with admin key",
                passed=False,
                error=str(e)
            ))
    
    def test_ssrf_protection(self):
        """Test SSRF/outbound URL protection."""
        print("## 2. SSRF/Outbound URL Protection")
        print()
        
        private_urls = [
            ("http://127.0.0.1:9999/mcp", "loopback"),
            ("http://10.0.0.1:8080/mcp", "private-10"),
            ("http://192.168.1.1:8080/mcp", "private-192"),
        ]
        
        for private_url, label in private_urls:
            try:
                response = requests.post(
                    f"{self.gateway_url}/llmrouter/mcp/servers",
                    headers={
                        "Authorization": f"Bearer {self.admin_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "server_id": f"test-ssrf-{label}",
                        "name": f"SSRF Test {label}",
                        "url": private_url,
                        "transport": "http"
                    },
                    timeout=self.timeout
                )
                # Should be blocked
                passed = response.status_code in [400, 403, 422]
                
                response_text = response.text.lower() if response.text else ""
                has_ssrf_message = any(keyword in response_text for keyword in [
                    "ssrf", "private", "blocked", "not allowed", "invalid"
                ])
                
                try:
                    body = response.json() if response.text and 'json' in response.headers.get('content-type', '') else {"raw": response.text[:200]}
                except:
                    body = {"raw": response.text[:200]}
                
                self.add_result(TestResult(
                    name=f"MCP registration blocks private IP ({label})",
                    passed=passed and has_ssrf_message,
                    status_code=response.status_code,
                    response_body=body,
                    details=f"URL: {private_url}, blocked={passed}, has_ssrf_msg={has_ssrf_message}"
                ))
            except Exception as e:
                self.add_result(TestResult(
                    name=f"MCP registration blocks private IP ({label})",
                    passed=False,
                    error=str(e)
                ))
    
    def test_skills_discovery(self):
        """Test skills discovery endpoints."""
        print("## 3. Skills Discovery Endpoints")
        print()
        
        endpoints = [
            ("/v1/skills", "LiteLLM native skills endpoint"),
            ("/.well-known/skills/index.json", "Skills discovery plugin"),
        ]
        
        for path, description in endpoints:
            try:
                response = requests.get(
                    f"{self.gateway_url}{path}",
                    headers={"Authorization": f"Bearer {self.admin_key}"},
                    timeout=self.timeout
                )
                # 200 (implemented) or 404/501 (not implemented) are both acceptable
                passed = response.status_code in [200, 404, 501]
                
                try:
                    body = response.json() if response.text and 'json' in response.headers.get('content-type', '') else {"raw": response.text[:200]}
                except:
                    body = {"raw": response.text[:200]}
                
                self.add_result(TestResult(
                    name=f"GET {path}",
                    passed=passed,
                    status_code=response.status_code,
                    response_body=body,
                    details=f"{description} - acceptable status={passed}"
                ))
            except Exception as e:
                self.add_result(TestResult(
                    name=f"GET {path}",
                    passed=False,
                    error=str(e)
                ))
    
    def test_health_endpoints(self):
        """Test health check endpoints."""
        print("## 4. Health Endpoints")
        print()
        
        # Test liveness probe
        try:
            response = requests.get(
                f"{self.gateway_url}/_health/live",
                timeout=self.timeout
            )
            passed = response.status_code == 200
            data = response.json() if response.text else {}
            has_status = "status" in data and data.get("status") == "alive"
            
            self.add_result(TestResult(
                name="GET /_health/live (liveness probe)",
                passed=passed and has_status,
                status_code=response.status_code,
                response_body=data,
                details=f"status_code={response.status_code}, has_alive_status={has_status}"
            ))
        except Exception as e:
            self.add_result(TestResult(
                name= "GET /_health/live (liveness probe)",
                passed=False,
                error=str(e)
            ))
        
        # Test readiness probe
        try:
            response = requests.get(
                f"{self.gateway_url}/_health/ready",
                timeout=self.timeout
            )
            passed = response.status_code in [200, 503]
            data = response.json() if response.text else {}
            
            self.add_result(TestResult(
                name="GET /_health/ready (readiness probe)",
                passed=passed,
                status_code=response.status_code,
                response_body=data,
                details=f"status_code={response.status_code} (200/503 acceptable)"
            ))
        except Exception as e:
            self.add_result(TestResult(
                name="GET /_health/ready (readiness probe)",
                passed=False,
                error=str(e)
            ))
    
    def test_mcp_endpoints(self):
        """Test MCP gateway endpoints."""
        print("## 5. MCP Endpoints")
        print()
        
        # Test list servers
        try:
            response = requests.get(
                f"{self.gateway_url}/llmrouter/mcp/servers",
                headers={"Authorization": f"Bearer {self.admin_key}"},
                timeout=self.timeout
            )
            passed = response.status_code == 200
            data = response.json() if response.text else {}
            has_servers = "servers" in data and isinstance(data.get("servers"), list)
            
            self.add_result(TestResult(
                name="GET /llmrouter/mcp/servers",
                passed=passed and has_servers,
                status_code=response.status_code,
                response_body=data,
                details=f"Returns server list: {has_servers}"
            ))
        except Exception as e:
            self.add_result(TestResult(
                name="GET /llmrouter/mcp/servers",
                passed=False,
                error=str(e)
            ))
        
        # Test list tools
        try:
            response = requests.get(
                f"{self.gateway_url}/llmrouter/mcp/tools",
                headers={"Authorization": f"Bearer {self.admin_key}"},
                timeout=self.timeout
            )
            passed = response.status_code == 200
            data = response.json() if response.text else {}
            has_tools = "tools" in data
            
            self.add_result(TestResult(
                name="GET /llmrouter/mcp/tools",
                passed=passed and has_tools,
                status_code=response.status_code,
                response_body=data,
                details=f"Returns tools list: {has_tools}"
            ))
        except Exception as e:
            self.add_result(TestResult(
                name="GET /llmrouter/mcp/tools",
                passed=False,
                error=str(e)
            ))
    
    def print_summary(self):
        """Print validation summary."""
        print("=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)
        print(f"Gateway URL: {self.report.gateway_url}")
        print(f"Timestamp: {self.report.timestamp}")
        print()
        print(f"Total Tests: {self.report.tests_run}")
        print(f"Passed: {self.report.tests_passed} ✅")
        print(f"Failed: {self.report.tests_failed} ❌")
        print()
        
        if self.report.tests_failed > 0:
            print("Failed Tests:")
            for result in self.report.results:
                if not result.passed:
                    print(f"  - {result.name}")
                    if result.error:
                        print(f"    Error: {result.error}")
            print()
        
        pass_rate = (self.report.tests_passed / self.report.tests_run * 100) if self.report.tests_run > 0 else 0
        print(f"Pass Rate: {pass_rate:.1f}%")
        
        if pass_rate >= 80:
            print("\n✅ OVERALL STATUS: PASS")
        else:
            print("\n❌ OVERALL STATUS: FAIL")
    
    def save_report(self, filepath: str = "E2E_API_VALIDATION_REPORT.json"):
        """Save detailed JSON report."""
        report_dict = asdict(self.report)
        with open(filepath, 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)
        print(f"\nDetailed report saved to: {filepath}")


def main():
    """Main entry point."""
    gateway_url = os.getenv("GATEWAY_URL", "http://localhost:4010")
    admin_key = os.getenv("ADMIN_KEY", "sk-test-master-key")
    
    validator = E2EValidator(gateway_url, admin_key)
    
    try:
        validator.run_all_tests()
        validator.print_summary()
        validator.save_report()
        
        # Exit with error code if any tests failed
        sys.exit(0 if validator.report.tests_failed == 0 else 1)
        
    except KeyboardInterrupt:
        print("\n\nValidation interrupted by user")
        sys.exit(2)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(3)


if __name__ == "__main__":
    main()
