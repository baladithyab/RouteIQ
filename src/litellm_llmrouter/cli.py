"""RouteIQ Gateway CLI.

Provides a unified command-line interface for the RouteIQ Gateway.

Usage::

    routeiq start [--config CONFIG] [--port PORT] [--workers N]
    routeiq validate-config [--config CONFIG]
    routeiq version
    routeiq probe-services
"""

import argparse
import sys

__all__ = [
    "main",
]


def main():
    """Entry point for the ``routeiq`` CLI."""
    parser = argparse.ArgumentParser(
        prog="routeiq",
        description="RouteIQ Gateway \u2014 Cloud-Native AI Gateway with Intelligent Routing",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # routeiq start
    start_parser = subparsers.add_parser("start", help="Start the gateway")
    start_parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Config file path (default: config/config.yaml)",
    )
    start_parser.add_argument(
        "--port",
        type=int,
        default=4000,
        help="HTTP port (default: 4000)",
    )
    start_parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Worker count (default: 1)",
    )

    # routeiq validate-config
    validate_parser = subparsers.add_parser(
        "validate-config", help="Validate configuration without starting"
    )
    validate_parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Config file path (default: config/config.yaml)",
    )

    # routeiq version
    subparsers.add_parser("version", help="Show version")

    # routeiq probe-services
    subparsers.add_parser("probe-services", help="Probe external service connectivity")

    args = parser.parse_args()

    if args.command == "start":
        _cmd_start(args)
    elif args.command == "validate-config":
        _cmd_validate_config(args)
    elif args.command == "version":
        _cmd_version()
    elif args.command == "probe-services":
        _cmd_probe_services()
    else:
        parser.print_help()
        sys.exit(1)


def _cmd_start(args: argparse.Namespace) -> None:
    """Start the gateway by delegating to startup.main()."""
    import os

    os.environ.setdefault("LITELLM_CONFIG_PATH", args.config)
    os.environ["LITELLM_PORT"] = str(args.port)
    os.environ["ROUTEIQ_WORKERS"] = str(args.workers)

    from litellm_llmrouter.startup import main as startup_main

    # Don't mutate sys.argv — pass config via env vars that startup.main() reads:
    #   LITELLM_CONFIG_PATH (--config), LITELLM_PORT (--port), ROUTEIQ_WORKERS (--workers)
    startup_main()


def _cmd_validate_config(args: argparse.Namespace) -> None:
    """Validate config without starting the gateway."""
    import os

    os.environ.setdefault("LITELLM_CONFIG_PATH", args.config)

    print(f"Validating config: {args.config}")

    # 1. Check the file exists
    if not os.path.isfile(args.config):
        print(f"ERROR: Config file not found: {args.config}")
        sys.exit(1)

    # 2. Try to parse the YAML
    try:
        import yaml

        with open(args.config) as f:
            config = yaml.safe_load(f)
        if not isinstance(config, dict):
            print("ERROR: Config file is not a valid YAML mapping")
            sys.exit(1)
        print(f"  YAML syntax: OK ({len(config)} top-level keys)")
    except Exception as e:
        print(f"ERROR: YAML parse error: {e}")
        sys.exit(1)

    # 3. Validate environment variables
    try:
        from litellm_llmrouter.env_validation import validate_environment

        result = validate_environment()
        if result.errors:
            print(f"  Environment: {len(result.errors)} error(s)")
            for err in result.errors:
                print(f"    ERROR: {err}")
        if result.warnings:
            print(f"  Environment: {len(result.warnings)} warning(s)")
            for warn in result.warnings:
                print(f"    WARN:  {warn}")
        if not result.errors and not result.warnings:
            print("  Environment: OK")
        sys.exit(1 if result.errors else 0)
    except ImportError:
        print("  Environment validation: skipped (module not available)")
        sys.exit(0)


def _cmd_version() -> None:
    """Print the RouteIQ version."""
    try:
        from importlib.metadata import version

        v = version("routeiq")
    except Exception:
        v = "0.0.0-dev"
    print(f"RouteIQ Gateway v{v}")


def _cmd_probe_services() -> None:
    """Probe external services and print the status table."""
    import asyncio

    try:
        from litellm_llmrouter.service_discovery import (
            probe_all_services,
            get_feature_availability,
            format_service_status_table,
        )

        async def _probe():
            services = await probe_all_services()
            features = get_feature_availability(services)
            return format_service_status_table(services, features)

        table = asyncio.run(_probe())
        print(table)
    except ImportError as e:
        print(f"Service discovery not available: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Service probe failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
