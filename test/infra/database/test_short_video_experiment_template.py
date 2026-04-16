# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
from __future__ import annotations

import os
import shutil
import tempfile

import pytest

from examples.experiment.short_video_negative_feedback.run_experiment import (
    run_condition,
)


@pytest.mark.asyncio
async def test_negative_feedback_experiment_separates_baseline_and_treatment():
    output_dir = tempfile.mkdtemp(prefix="oasis-short-video-exp-")
    try:
        baseline = await run_condition(
            os.path.join(output_dir, "baseline.db"),
            heavy_negative_feedback=False,
        )
        treatment = await run_condition(
            os.path.join(output_dir, "treatment.db"),
            heavy_negative_feedback=True,
        )

        baseline_comedy = next(
            row for row in baseline["videos"] if row["category"] == "comedy")
        treatment_comedy = next(
            row for row in treatment["videos"] if row["category"] == "comedy")
        baseline_lifestyle = next(
            row for row in baseline["videos"] if row["category"] == "lifestyle")
        treatment_lifestyle = next(
            row for row in treatment["videos"] if row["category"] == "lifestyle")

        assert baseline_comedy["traffic_pool_level"] == 1
        assert treatment_comedy["traffic_pool_level"] == 0
        assert baseline_comedy["negative_count"] < treatment_comedy["negative_count"]
        assert baseline_lifestyle["traffic_pool_level"] == 0
        assert treatment_lifestyle["traffic_pool_level"] == 1
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)
