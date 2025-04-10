# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
from dataclasses import dataclass
from typing import Any, Dict, List

from oasis.social_platform.typing import ActionType


@dataclass
class SingleAction:
    agent_id: int
    action: ActionType
    args: Dict[str, Any]
    r"""Some predefined social platform actions that need to be executed by
    certain agents.

    Args:
        agent_id: The ID of the agent that will perform the action.
        action: The action to perform.
        args: The arguments to pass to the action. For details of each args in
            each action, please refer to
            `https://github.com/camel-ai/oasis/blob/main/oasis/social_agent/agent_action.py`.
    """


@dataclass
class EnvAction:
    activate_agents: List[int] | None = None
    intervention: List[SingleAction] | None = None
    r"""All actions in a single timestep to perform in the Oasis environment.

    Args:
        activate_agents: The list of agent ids that will be activated.
        intervention: The interventions to perform.
    """
