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
import pytest

from oasis.social_platform.platform import Platform


@pytest.fixture
def platform(tmp_path):
    instance = Platform(str(tmp_path / "quote_reports.db"),
                        refresh_rec_post_count=10)
    yield instance
    instance.db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["refresh", "search_posts"])
@pytest.mark.parametrize("quote_only", [True, False])
@pytest.mark.parametrize("original_reports, quote_reports", [(0, 2), (2, 0)])
async def test_quote_post_report_counts(platform, action, quote_only,
                                        original_reports, quote_reports):
    await platform.sign_up(1, ("original_author", "Original Author", ""))
    await platform.sign_up(2, ("quote_author", "Quote Author", ""))
    original = await platform.create_post(1, "Original content")
    original_id = original["post_id"]
    quote = await platform.quote_post(2, (original_id, "Quote content"))
    quote_id = quote["post_id"]
    platform.db.executemany(
        "UPDATE post SET num_reports = ? WHERE post_id = ?",
        [(original_reports, original_id), (quote_reports, quote_id)])

    if action == "refresh":
        post_ids = [quote_id] if quote_only else [original_id, quote_id]
        platform.db.executemany(
            "INSERT INTO rec (user_id, post_id) VALUES (?, ?)",
            [(1, post_id) for post_id in post_ids])
        platform.db.commit()
        response = await platform.refresh(1)
    else:
        platform.db.commit()
        query = str(quote_id) if quote_only else "Original content"
        response = await platform.search_posts(1, query)

    assert response["success"], response
    posts = {post["post_id"]: post for post in response["posts"]}
    expected_counts = {quote_id: quote_reports}
    if not quote_only:
        expected_counts[original_id] = original_reports
    assert set(posts) == set(expected_counts)
    for post_id, report_count in expected_counts.items():
        assert posts[post_id]["num_reports"] == report_count
        warning = (f"[Warning: This post has been reported "
                   f"{report_count} times]")
        if report_count >= platform.report_threshold:
            assert posts[post_id]["content"].startswith(warning)
        else:
            assert not posts[post_id]["content"].startswith("[Warning:")
