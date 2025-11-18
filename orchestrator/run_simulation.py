from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
from typing import Optional

import oasis
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from oasis.social_platform.platform import Platform
from oasis.social_platform.channel import Channel
from oasis.social_platform.typing import RecsysType

from generation.emission_policy import EmissionPolicy
from orchestrator.agent_factory import build_agent_graph_from_csv
from orchestrator.manifest_loader import Manifest, load_manifest
from orchestrator.sidecar_logger import SidecarLogger
from orchestrator.expect_registry import ExpectRegistry
from orchestrator.interceptor_channel import InterceptorChannel
from orchestrator.scheduler import MultiLabelScheduler, MultiLabelTargets


async def run(
    manifest_path: Path,
    personas_csv: Path,
    db_path: Path,
    steps: int,
) -> None:
    manifest: Manifest = load_manifest(manifest_path)
    run_seed = manifest.run_seed

    sidecar = SidecarLogger(path=db_path.parent / "sidecar.jsonl")

    # Configure emission policy (extendable for subclasses later)
    policy = EmissionPolicy(
        run_seed=run_seed,
        post_label_mode_probs=manifest.post_label_mode_probs,
        label_to_tokens=None,
    )
    expect_registry = ExpectRegistry()
    targets_cfg = manifest.multi_label_targets
    scheduler = MultiLabelScheduler(
        MultiLabelTargets(
            target_rate=targets_cfg["target_rate"],
            min_rate=targets_cfg["min_rate"],
            max_rate=targets_cfg["max_rate"],
        )
    )

    # Build model (expects OPENAI_API_KEY in env if using OpenAI)
    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O_MINI,
    )

    # Build platform with an explicit channel so env doesn't create its own
    base_channel = Channel()
    channel = InterceptorChannel(base_channel, expect_registry)
    platform = Platform(
        db_path=str(db_path),
        channel=channel,
        sandbox_clock=oasis.Clock(magnification_factor=60),
        start_time=None,
        recsys_type=RecsysType.TWHIN,
        refresh_rec_post_count=3,
        max_rec_post_len=5,
        following_post_count=2,
        show_score=False,
        allow_self_rating=False,
    )

    agent_graph = await build_agent_graph_from_csv(
        personas_csv=personas_csv,
        model=model,
        channel=channel,
        available_actions=None,
        emission_policy=policy,
        sidecar_logger=sidecar,
        run_seed=run_seed,
        # ExtendedSocialAgent will use this to publish expectations
        # and we intercept on the channel to ensure tokens exist.
        # (build_agent_graph_from_csv forwards kwargs to ExtendedSocialAgent)
        # The factory currently does not accept extra kwargs; we pass via
        # ExtendedSocialAgent instantiation below. (Handled inside factory)
        expect_registry=expect_registry,
        scheduler=scheduler,
    )

    env = oasis.make(
        agent_graph=agent_graph,
        platform=platform,
        database_path=str(db_path),
    )

    await env.reset()

    # Drive steps with all agents using LLMAction
    for _ in range(max(0, int(steps))):
        actions = {agent: oasis.LLMAction() for _, agent in env.agent_graph.get_agents()}
        await env.step(actions)

    await env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run production simulation with ExtendedSocialAgent.")
    parser.add_argument("--manifest", type=str, required=True, help="Path to manifest YAML.")
    parser.add_argument("--personas-csv", type=str, required=True, help="Path to personas CSV.")
    parser.add_argument("--db", type=str, required=True, help="Path to output sqlite DB.")
    parser.add_argument("--steps", type=int, default=10, help="Number of steps to run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(os.path.abspath(args.manifest))
    personas_csv = Path(os.path.abspath(args.personas_csv))
    db_path = Path(os.path.abspath(args.db))
    asyncio.run(run(manifest_path, personas_csv, db_path, args.steps))


if __name__ == "__main__":
    main()


