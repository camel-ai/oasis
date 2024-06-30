from dataclasses import dataclass
from typing import Any

from social_simulation.social_platform.typing import ActionType


@dataclass
class UserInfo:
    name: str | None = None
    description: str | None = None
    profile: dict[str, Any] | None = None
    is_controllable: bool = False

    def to_system_message(self) -> str:
        if self.name is not None:
            name_string = f"You are a twitter user with name {self.name}."
        else:
            name_string = "You are a twitter user without specifying a name."
        if self.profile['other_info']['user_profile'] is not None:
            user_profile = self.profile['other_info']['user_profile']
            description_string = f"Your have profile: {user_profile}."
        else:
            description_string = "Your don't specify any profile."
        # Allowed actions except for the SIGN_UP
        valid_actions = [
            f"action_{action_type.value}" for action_type in list(ActionType)
            if action_type != ActionType.SIGNUP
        ]
        valid_action_string = ", ".join(valid_actions)
        action_string = (f"Remember each time you need to choose one twitter "
                         f"action from following list: {valid_action_string}.")
        return f"{name_string} {description_string} {action_string}"
