## Description

Describe your changes in detail (optional if the linked issue already contains a detailed description of the changes).

## Checklist

Go over all the following points, and put an `x` in all the boxes that apply.

- [ ] I have read the [CONTRIBUTION](https://github.com/camel-ai/oasis/blob/master/CONTRIBUTING.md) guide (**required**)
- [ ] I have linked this PR to an issue using the Development section on the right sidebar or by adding `Fixes #issue-number` in the PR description (**required**)
- [ ] I have checked if any dependencies need to be added or updated in `pyproject.toml`
- [ ] I have updated the tests accordingly (*required for a bug fix or a new feature*)
- [ ] I have updated the documentation if needed:
- [ ] I have added examples if this is a new feature

**Note:** If you are developing a new action for `SocialAgent`, please review the checklist below and mark all applicable items with an `x`. If you're not adding a new action, you can skip this section.

- [ ] I have added the new action to `ActionType` in [`typing.py`](https://github.com/camel-ai/oasis/blob/main/oasis/social_platform/typing.py).
- [ ] I have added a corresponding test or a similar function, as shown in [`test_user_create_post.py`](https://github.com/camel-ai/oasis/blob/main/test/infra/database/test_user_create_post.py).
- [ ] I have included the new `ActionType` in both [`test_action_docstring.py`](https://github.com/camel-ai/oasis/blob/main/test/agent/test_action_docstring.py) and [`test_twitter_user_agent_all_actions.py`](https://github.com/camel-ai/oasis/blob/main/test/agent/test_twitter_user_agent_all_actions.py).
- [ ] I have documented the new action in [`actions.mdx`](https://github.com/camel-ai/oasis/blob/main/docs/key_modules/actions.mdx); the Mintlify GitHub app will deploy the changes automatically.

If you are unsure about any of these, don't hesitate to ask. We are here to help!
