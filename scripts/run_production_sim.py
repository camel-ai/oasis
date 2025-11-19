#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import csv
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from camel.models import BaseModelBackend
from camel.types import ModelType

import oasis
from generation.emission_policy import EmissionPolicy
from oasis.clock.clock import Clock
from oasis.social_platform.channel import Channel
from oasis.social_platform.platform import Platform
from oasis.social_platform.typing import RecsysType
from orchestrator.agent_factory import build_agent_graph_from_csv
from orchestrator.expect_registry import ExpectRegistry
from orchestrator.interceptor_channel import InterceptorChannel
from orchestrator.manifest_loader import load_manifest
from orchestrator.model_provider import (LLMProviderSettings,
                                         create_model_backend)
from orchestrator.scheduler import MultiLabelScheduler, MultiLabelTargets
from orchestrator.sidecar_logger import SidecarLogger

#
# LLM selection (single, modular configuration block)
#
# Examples:
#   Use xAI Grok:
#     LLM_SETTINGS = LLMProviderSettings(provider="xai", model_name="grok-4-fast-non-reasoning", api_key="<key>", base_url="https://api.x.ai/v1")
#   Use Gemini:
#     LLM_SETTINGS = LLMProviderSettings(provider="gemini", model_name="gemini-2.5-flash", api_key="<key>")
#   Use OpenAI:
#     LLM_SETTINGS = LLMProviderSettings(provider="openai", model_name=ModelType.GPT_4O_MINI.value, api_key=os.getenv("OPENAI_API_KEY", ""))
#
LLM_SETTINGS: LLMProviderSettings = LLMProviderSettings(
    provider="xai",
    model_name="grok-4-fast-non-reasoning",
    api_key=(os.getenv("XAI_API_KEY", "").strip() or ""),
    base_url="https://api.x.ai/v1",
    timeout_seconds=3600.0,
)


def _validate_personas_and_edges(manifest, personas_csv: Path,
                                 edges_path: Path | None) -> None:
    r"""Validate that personas and edges are consistent with the manifest.

    This guards against running a production simulation with mismatched
    persona counts or an out-of-date follow graph.
    """
    if not personas_csv.exists():
        raise SystemExit(
            f"[production] Personas CSV not found: {personas_csv}. "
            "Generate personas before running the simulation."
        )
    # Count persona rows (excluding header).
    with personas_csv.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        _ = next(reader, None)
        persona_rows = sum(1 for _ in reader)
    if persona_rows <= 0:
        raise SystemExit(
            f"[production] Personas CSV {personas_csv} contains no rows.")

    # If manifest has a population block, ensure totals match.
    population_cfg = getattr(manifest, "data", {}).get("population", {})
    if isinstance(population_cfg, dict) and population_cfg:
        expected_total = sum(int(v) for v in population_cfg.values())
        if expected_total > 0 and persona_rows != expected_total:
            raise SystemExit(
                "[production] Personas CSV row count does not match manifest population. "
                f"CSV rows={persona_rows}, manifest population sum={expected_total}. "
                "Regenerate personas for this manifest."
            )

    if edges_path is None:
        # Caller chose to run without a seeded follow graph.
        return

    if not edges_path.exists():
        raise SystemExit(
            f"[production] Edges CSV not found: {edges_path}. "
            "Generate the follow graph with scripts/build_graph.py or set PROD_EDGES_CSV."
        )

    max_id = -1
    with edges_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                follower = int(r["follower_id"])
                followee = int(r["followee_id"])
            except Exception:
                continue
            if follower > max_id:
                max_id = follower
            if followee > max_id:
                max_id = followee

    if max_id >= persona_rows:
        raise SystemExit(
            "[production] Edges CSV references user_id outside the personas index range. "
            f"Max referenced user_id={max_id}, personas rows={persona_rows}. "
            "Regenerate the graph after updating personas."
        )


def _follow_table_empty(env) -> bool:
    try:
        cur = env.platform.db_cursor
        cur.execute("SELECT COUNT(*) FROM follow")
        cnt = cur.fetchone()[0]
        return int(cnt) == 0
    except Exception:
        return True


def _seed_initial_follows_from_csv(env, edges_csv: Path) -> int:
    """Seed initial follow edges into the SQLite DB and AgentGraph."""
    if not edges_csv.exists():
        return 0
    rows = []
    upd_followings = []
    upd_followers = []
    try:
        current_time = env.platform.sandbox_clock.get_time_step()
    except Exception:
        current_time = 0
    with edges_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                follower = int(r["follower_id"])
                followee = int(r["followee_id"])
            except Exception:
                continue
            if follower == followee:
                continue
            rows.append((follower, followee, current_time))
            upd_followings.append((follower,))
            upd_followers.append((followee,))
    if not rows:
        return 0
    # insert follows
    env.platform.pl_utils._execute_many_db_command(
        "INSERT INTO follow (follower_id, followee_id, created_at) VALUES (?, ?, ?)",
        rows,
        commit=True,
    )
    # update counters
    env.platform.pl_utils._execute_many_db_command(
        "UPDATE user SET num_followings = num_followings + 1 WHERE user_id = ?",
        upd_followings,
        commit=True,
    )
    env.platform.pl_utils._execute_many_db_command(
        "UPDATE user SET num_followers = num_followers + 1 WHERE user_id = ?",
        upd_followers,
        commit=True,
    )
    # sync in-memory graph
    for follower, followee, _ in rows:
        try:
            env.agent_graph.add_edge(follower, followee)
        except Exception:
            pass
    return len(rows)


async def run(manifest_path: Path, personas_csv: Path, db_path: Path,
              steps: int, edges_csv: Path | None,
              warmup_steps: int) -> None:
    manifest = load_manifest(manifest_path)
    run_seed = manifest.run_seed

    # Choose edges path with env override, then validate inputs.
    override = os.getenv("PROD_EDGES_CSV")
    edges_path = Path(override) if override else edges_csv
    _validate_personas_and_edges(manifest, personas_csv, edges_path)

    sidecar = SidecarLogger(path=db_path.parent / "sidecar.jsonl")

    policy = EmissionPolicy(
        run_seed=run_seed,
        post_label_mode_probs=manifest.post_label_mode_probs,
        label_to_tokens=None,  # defaults cover the 6-parent-family setup
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

    # Define the model for the agents (LLMAction only; no manual actions)
    model = create_model_backend(LLM_SETTINGS)

    # Setup platform and channel
    base_channel = Channel()
    channel = InterceptorChannel(base_channel, expect_registry)
    platform = Platform(
        db_path=str(db_path),
        channel=channel,
        sandbox_clock=Clock(k=60),
        start_time=None,
        recsys_type=RecsysType.TWHIN,
        refresh_rec_post_count=3,
        max_rec_post_len=5,
        following_post_count=2,
        show_score=False,
        allow_self_rating=False,
    )

    # Restrict available actions for Twitter simulation
    allowed_actions = [
        RecsysType  # placeholder to keep import grouped (no-op)
    ]
    # Define allowed ActionTypes explicitly
    from oasis.social_platform.typing import ActionType as _AT  # local alias
    allowed_actions = [
        _AT.REFRESH,
        _AT.SEARCH_USER,
        _AT.SEARCH_POSTS,
        _AT.CREATE_POST,
        _AT.LIKE_POST,
        _AT.FOLLOW,
        _AT.UNFOLLOW,
        _AT.REPOST,
        _AT.QUOTE_POST,
        # _AT.UPDATE_REC_TABLE,
        _AT.CREATE_COMMENT,
        _AT.LIKE_COMMENT,
        _AT.DO_NOTHING,
    ]

    # Build agent graph using ExtendedSocialAgent with persona class metadata
    agent_graph = await build_agent_graph_from_csv(
        personas_csv=personas_csv,
        model=model,
        channel=channel,
        available_actions=allowed_actions,
        emission_policy=policy,
        sidecar_logger=sidecar,
        run_seed=run_seed,
        expect_registry=expect_registry,
        scheduler=scheduler,
        guidance_config=manifest.guidance_config,
    )

    # Create environment
    env = oasis.make(
        agent_graph=agent_graph,
        platform=platform,
        database_path=str(db_path),
    )

    await env.reset()

    # Seed initial follow graph if provided and table is empty
    try:
        if _follow_table_empty(env):
            if edges_path:
                seeded = _seed_initial_follows_from_csv(env, edges_path)
                if seeded:
                    print(
                        f"[production] Seeded {seeded} follow edges from {edges_path}"
                    )
        else:
            print("[production] Follow table non-empty; skipping follow seed.")
    except Exception as e:
        print(f"[production] Follow seeding failed: {e}")

    # Warmup: run a few LLMAction steps to populate initial content before main loop
    for _ in range(max(0, int(warmup_steps))):
        actions = {agent: oasis.LLMAction() for _, agent in env.agent_graph.get_agents()}
        await env.step(actions)

    # LLMAction for all agents across steps
    for _ in range(max(0, int(steps))):
        actions = {agent: oasis.LLMAction() for _, agent in env.agent_graph.get_agents()}
        await env.step(actions)

    await env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run production simulation with ExtendedSocialAgent (LLMAction only)."
    )
    parser.add_argument("--manifest", type=str, required=True, help="Path to manifest YAML.")
    parser.add_argument("--personas-csv", type=str, required=True, help="Path to personas CSV with class columns.")
    parser.add_argument("--db", type=str, required=True, help="Path to output sqlite DB.")
    parser.add_argument("--steps", type=int, default=10, help="Number of steps to run.")
    parser.add_argument("--edges-csv", type=str, default="", help="Path to edges CSV (follower_id,followee_id). Env PROD_EDGES_CSV overrides.")
    parser.add_argument("--warmup-steps", type=int, default=1, help="Warmup LLMAction steps before main loop.")
    parser.add_argument(
        "--fresh-db",
        action="store_true",
        help="If set, delete the existing DB file before running.",
    )
    parser.add_argument(
        "--unique-db",
        action="store_true",
        help="If set, append a timestamp to the DB filename to create a new DB per run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(os.path.abspath(args.manifest))
    personas_csv = Path(os.path.abspath(args.personas_csv))
    db_path = Path(os.path.abspath(args.db))
    # Ensure parent directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # Choose DB handling strategy
    if args.unique_db:
        # Append timestamp before suffix
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        db_path = db_path.with_name(f"{db_path.stem}_{ts}{db_path.suffix}")
    elif args.fresh_db and db_path.exists():
        try:
            os.remove(db_path)
        except Exception as e:
            print(f"[production] Failed to remove existing DB: {e}")
    edges_csv = Path(os.path.abspath(args.edges_csv)) if args.edges_csv else None
    asyncio.run(run(manifest_path, personas_csv, db_path, args.steps, edges_csv, args.warmup_steps))


if __name__ == "__main__":
    main()


