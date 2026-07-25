"""
Failure-Aware Search Agents — ADK orchestrator.

Wires search → verification → composer via SequentialAgent (or mock pipeline).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any, Tuple

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.composer import MockComposerAgent, build_composer_agent
from agents.prompts_loader import load_prompt
from agents.search_agent import MockSearchAgent, build_search_agent
from agents.verification_agent import MockVerificationAgent, build_verification_agent
from shared.hardware import get_runtime_info
from shared.inference_config import load_inference_config

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _mcp_server_path() -> str:
    return str(Path(__file__).resolve().parents[1] / "mcp_server" / "search_mcp_server.py")


async def build_mcp_tools() -> Tuple[Any, Any]:
    """
    Connect to MCP search server via ADK MCPToolset.

    Returns (toolset_or_None, toolset_or_None). The second element is the
    same toolset (caller calls toolset.close() when done), or None on failure.
    """
    try:
        from mcp import StdioServerParameters
        from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
        from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset

        toolset = MCPToolset(
            connection_params=StdioConnectionParams(
                server_params=StdioServerParameters(
                    command=sys.executable,
                    args=[_mcp_server_path()],
                ),
            ),
            tool_filter=[
                "web_search",
                "wikipedia_lookup",
                "coherence_score",
                "search_quality_check",
            ],
        )
        return toolset, toolset
    except ImportError:
        logger.warning("ADK MCP tools unavailable — agents will use shared.search_tools directly")
        return None, None
    except Exception as exc:
        logger.warning("MCP connection failed: %s — using mock tools", exc)
        return None, None


async def build_agent(use_adk: bool = False):
    """Build orchestrator: mock (default), ollama, or ADK SequentialAgent."""
    cfg = load_inference_config()
    runtime = get_runtime_info()
    backend = cfg.get("backend", "mock").lower()
    if backend == "ollama":
        gpu = runtime.get("gpu", {})
        if not gpu.get("available", False) or not gpu.get("enough", False):
            if runtime.get("kaggle", False):
                logger.warning(
                    "Kaggle environment detected but no suitable GPU is available. "
                    "Falling back to mock backend in this kernel. "
                    "Enable a GPU accelerator or use a GPU-backed Kaggle session for Ollama/real inference."
                )
            else:
                logger.warning(
                    "No suitable GPU detected locally. "
                    "Falling back to mock backend. "
                    "To use Ollama/real GPU inference, run on a GPU-enabled host or Kaggle."
                )
            backend = "mock"
            cfg["backend"] = "mock"

    tools, toolset = await build_mcp_tools()
    tool_list = [tools] if tools else []

    if use_adk or backend == "adk":
        search = build_search_agent(tools=tool_list)
        verify = build_verification_agent(tools=tool_list)
        composer = build_composer_agent()

        try:
            from google.adk.agents import SequentialAgent

            orchestrator = SequentialAgent(
                name="failure_aware_orchestrator",
                sub_agents=[search, verify, composer],
                description="Routes query through search, verification, and response composition.",
            )
            return orchestrator, toolset
        except ImportError:
            logger.info("google-adk not installed — falling back to mock pipeline")

    if backend == "ollama":
        from agents.ollama_pipeline import build_ollama_pipeline

        logger.info(
            "Using Ollama live pipeline (model=%s). DPO use_dpo=%s (not wired to Ollama yet).",
            cfg.get("ollama_model"),
            cfg.get("use_dpo"),
        )
        search, verify, composer = build_ollama_pipeline(cfg)
        return MockOrchestrator(search, verify, composer, backend="ollama"), toolset

    logger.info(
        "Using heuristic mock pipeline (no LLM). "
        "Set INFERENCE_BACKEND=ollama for live Ollama agents, or use generate_rollouts for DPO data."
    )
    search = MockSearchAgent(instruction=load_prompt("search_agent_prompt.txt"))
    verify = MockVerificationAgent(instruction=load_prompt("verify_agent_prompt.txt"))
    composer = MockComposerAgent(instruction=load_prompt("composer_prompt.txt"))
    return MockOrchestrator(search, verify, composer, backend="mock"), toolset


class MockOrchestrator:
    """Sequential mock pipeline for local testing."""

    name = "failure_aware_orchestrator"

    def __init__(self, search, verify, composer, backend: str = "mock") -> None:
        self.search = search
        self.verify = verify
        self.composer = composer
        self.backend = backend

    async def run(self, query: str, search_mode: str = "general") -> dict:
        search_out = await self.search.run(query, search_mode=search_mode)
        verify_out = await self.verify.run(
            search_out.get("search_draft", ""),
            query,
        )
        final = await self.composer.run(
            query,
            search_out.get("search_draft", ""),
            verify_out.get("verification_verdict", "{}"),
            search_out=search_out,
            verify_out=verify_out,
        )
        return {
            "query": query,
            "inference_backend": self.backend,
            "search": search_out,
            "verification": verify_out,
            "final": final,
        }


async def run_query(query: str, use_adk: bool = False, search_mode: str = "general") -> dict:
    cfg = load_inference_config()
    if cfg.get("backend") == "adk":
        use_adk = True

    agent, toolset = await build_agent(use_adk=use_adk)
    try:
        if isinstance(agent, MockOrchestrator):
            return await agent.run(query, search_mode=search_mode)
        from google.adk.runners import Runner

        # For ADK, we could pass mode via context, but we mock it here.
        runner = Runner(agent=agent)
        result = await runner.run_async(user_message=f"[{search_mode} mode] {query}")
        return {"result": result, "inference_backend": "adk"}
    finally:
        if toolset is not None:
            try:
                close_result = toolset.close()
                if asyncio.iscoroutine(close_result):
                    await close_result
            except Exception:
                pass


async def main_async(query: str) -> None:
    result = await run_query(query)
    print(result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run failure-aware search orchestrator")
    parser.add_argument("--query", required=True, help="User question")
    parser.add_argument("--adk", action="store_true", help="Use Google ADK runner (requires API key)")
    args = parser.parse_args()

    if not args.adk and not os.environ.get("GOOGLE_API_KEY"):
        logger.info("GOOGLE_API_KEY not set — running mock orchestrator")

    asyncio.run(main_async(args.query))


if __name__ == "__main__":
    main()
