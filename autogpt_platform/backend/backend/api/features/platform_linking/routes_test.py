"""Tests for platform bot linking API routes."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException


@asynccontextmanager
async def _fake_transaction():
    """Stub for backend.data.db.transaction used in route tests.

    The real transaction() opens a Prisma tx which binds asyncio primitives to
    the running event loop. Tests run in their own loops via pytest-asyncio, so
    we swap this in to avoid cross-loop errors.
    """
    yield MagicMock()


from backend.api.features.platform_linking.auth import check_bot_api_key
from backend.api.features.platform_linking.models import (
    BotChatRequest,
    ConfirmLinkResponse,
    CreateLinkTokenRequest,
    CreateUserLinkTokenRequest,
    DeleteLinkResponse,
    LinkTokenStatusResponse,
    LinkType,
    Platform,
    ResolveResponse,
    ResolveServerRequest,
    ResolveUserRequest,
)


class TestPlatformEnum:
    def test_all_platforms_exist(self):
        assert Platform.DISCORD.value == "DISCORD"
        assert Platform.TELEGRAM.value == "TELEGRAM"
        assert Platform.SLACK.value == "SLACK"
        assert Platform.TEAMS.value == "TEAMS"
        assert Platform.WHATSAPP.value == "WHATSAPP"
        assert Platform.GITHUB.value == "GITHUB"
        assert Platform.LINEAR.value == "LINEAR"


class TestBotApiKeyAuth:
    @patch.dict("os.environ", {"PLATFORM_BOT_API_KEY": ""}, clear=False)
    @patch("backend.api.features.platform_linking.auth.Settings")
    def test_no_key_configured_allows_when_auth_disabled(self, mock_settings_cls):
        from backend.api.features.platform_linking.auth import _auth_enabled

        _auth_enabled.cache_clear()
        mock_settings_cls.return_value.config.enable_auth = False
        check_bot_api_key(None)

    @patch.dict("os.environ", {"PLATFORM_BOT_API_KEY": ""}, clear=False)
    @patch("backend.api.features.platform_linking.auth.Settings")
    def test_no_key_configured_rejects_when_auth_enabled(self, mock_settings_cls):
        from backend.api.features.platform_linking.auth import _auth_enabled

        _auth_enabled.cache_clear()
        mock_settings_cls.return_value.config.enable_auth = True
        with pytest.raises(HTTPException) as exc_info:
            check_bot_api_key(None)
        assert exc_info.value.status_code == 503

    @patch.dict("os.environ", {"PLATFORM_BOT_API_KEY": "secret123"}, clear=False)
    def test_valid_key(self):
        check_bot_api_key("secret123")

    @patch.dict("os.environ", {"PLATFORM_BOT_API_KEY": "secret123"}, clear=False)
    def test_invalid_key_rejected(self):
        with pytest.raises(HTTPException) as exc_info:
            check_bot_api_key("wrong")
        assert exc_info.value.status_code == 401

    @patch.dict("os.environ", {"PLATFORM_BOT_API_KEY": "secret123"}, clear=False)
    def test_missing_key_rejected(self):
        with pytest.raises(HTTPException) as exc_info:
            check_bot_api_key(None)
        assert exc_info.value.status_code == 401


class TestCreateLinkTokenRequest:
    def test_valid_request(self):
        req = CreateLinkTokenRequest(
            platform=Platform.DISCORD,
            platform_server_id="1126875755960336515",
            platform_user_id="353922987235213313",
            platform_username="Bently",
            server_name="My Discord Server",
        )
        assert req.platform == Platform.DISCORD
        assert req.platform_server_id == "1126875755960336515"
        assert req.platform_user_id == "353922987235213313"
        assert req.server_name == "My Discord Server"

    def test_minimal_request(self):
        req = CreateLinkTokenRequest(
            platform=Platform.TELEGRAM,
            platform_server_id="-100123456789",
            platform_user_id="987654321",
        )
        assert req.server_name is None
        assert req.platform_username is None

    def test_empty_server_id_rejected(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            CreateLinkTokenRequest(
                platform=Platform.DISCORD,
                platform_server_id="",
                platform_user_id="123",
            )

    def test_too_long_server_id_rejected(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            CreateLinkTokenRequest(
                platform=Platform.DISCORD,
                platform_server_id="x" * 256,
                platform_user_id="123",
            )

    def test_invalid_platform_rejected(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            CreateLinkTokenRequest.model_validate(
                {
                    "platform": "INVALID",
                    "platform_server_id": "123",
                    "platform_user_id": "456",
                }
            )


class TestResolveServerRequest:
    def test_valid_request(self):
        req = ResolveServerRequest(
            platform=Platform.DISCORD,
            platform_server_id="1126875755960336515",
        )
        assert req.platform == Platform.DISCORD
        assert req.platform_server_id == "1126875755960336515"

    def test_empty_server_id_rejected(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ResolveServerRequest(
                platform=Platform.SLACK,
                platform_server_id="",
            )


class TestBotChatRequest:
    def test_server_context(self):
        req = BotChatRequest(
            platform=Platform.DISCORD,
            platform_server_id="1126875755960336515",
            platform_user_id="353922987235213313",
            message="Hello CoPilot!",
        )
        assert req.platform == Platform.DISCORD
        assert req.platform_server_id == "1126875755960336515"
        assert req.session_id is None

    def test_dm_context_omits_server_id(self):
        req = BotChatRequest(
            platform=Platform.DISCORD,
            platform_user_id="353922987235213313",
            message="Hello in DMs!",
        )
        assert req.platform_server_id is None

    def test_with_session_id(self):
        req = BotChatRequest(
            platform=Platform.DISCORD,
            platform_server_id="guild_123",
            platform_user_id="user_456",
            message="follow up",
            session_id="session-uuid-here",
        )
        assert req.session_id == "session-uuid-here"

    def test_empty_message_rejected(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            BotChatRequest(
                platform=Platform.DISCORD,
                platform_server_id="guild_123",
                platform_user_id="user_456",
                message="",
            )

    def test_empty_string_server_id_rejected(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            BotChatRequest(
                platform=Platform.DISCORD,
                platform_server_id="",
                platform_user_id="user_456",
                message="hi",
            )


class TestResponseModels:
    def test_link_token_status_pending(self):
        resp = LinkTokenStatusResponse(status="pending")
        assert resp.status == "pending"

    def test_link_token_status_linked(self):
        resp = LinkTokenStatusResponse(status="linked")
        assert resp.status == "linked"

    def test_link_token_status_expired(self):
        resp = LinkTokenStatusResponse(status="expired")
        assert resp.status == "expired"

    def test_resolve_linked(self):
        resp = ResolveResponse(linked=True)
        assert resp.linked is True

    def test_resolve_not_linked(self):
        resp = ResolveResponse(linked=False)
        assert resp.linked is False

    def test_confirm_link_response(self):
        resp = ConfirmLinkResponse(
            success=True,
            platform="DISCORD",
            platform_server_id="1126875755960336515",
            server_name="My Server",
        )
        assert resp.success is True
        assert resp.server_name == "My Server"

    def test_delete_link_response(self):
        resp = DeleteLinkResponse(success=True)
        assert resp.success is True


class TestResolveEndpoint:
    """Endpoint-level tests using mocked Prisma."""

    @pytest.mark.asyncio
    @patch.dict("os.environ", {"PLATFORM_BOT_API_KEY": "testkey"}, clear=False)
    async def test_resolve_linked_server(self):
        from backend.api.features.platform_linking.routes import resolve_platform_server

        mock_link = MagicMock()
        mock_link.userId = "autogpt-user-123"

        with patch(
            "backend.api.features.platform_linking.routes.find_server_link",
            new=AsyncMock(return_value=mock_link),
        ):
            result = await resolve_platform_server(
                ResolveServerRequest(
                    platform=Platform.DISCORD,
                    platform_server_id="guild_123",
                ),
                x_bot_api_key="testkey",
            )

        assert result.linked is True

    @pytest.mark.asyncio
    @patch.dict("os.environ", {"PLATFORM_BOT_API_KEY": "testkey"}, clear=False)
    async def test_resolve_unlinked_server(self):
        from backend.api.features.platform_linking.routes import resolve_platform_server

        with patch(
            "backend.api.features.platform_linking.routes.find_server_link",
            new=AsyncMock(return_value=None),
        ):
            result = await resolve_platform_server(
                ResolveServerRequest(
                    platform=Platform.DISCORD,
                    platform_server_id="guild_unknown",
                ),
                x_bot_api_key="testkey",
            )

        assert result.linked is False

    @pytest.mark.asyncio
    @patch.dict("os.environ", {"PLATFORM_BOT_API_KEY": "testkey"}, clear=False)
    async def test_resolve_rejects_wrong_api_key(self):
        from backend.api.features.platform_linking.routes import resolve_platform_server

        with pytest.raises(HTTPException) as exc_info:
            await resolve_platform_server(
                ResolveServerRequest(
                    platform=Platform.DISCORD,
                    platform_server_id="guild_123",
                ),
                x_bot_api_key="wrong_key",
            )
        assert exc_info.value.status_code == 401


class TestCreateLinkTokenEndpoint:
    @pytest.mark.asyncio
    @patch.dict("os.environ", {"PLATFORM_BOT_API_KEY": "testkey"}, clear=False)
    async def test_create_token_for_unlinked_server(self):
        from backend.api.features.platform_linking.routes import create_link_token

        with (
            patch(
                "backend.api.features.platform_linking.routes.find_server_link",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "backend.api.features.platform_linking.routes.transaction",
                new=_fake_transaction,
            ),
            patch(
                "backend.api.features.platform_linking.routes.PlatformLinkToken"
            ) as mock_token_model,
        ):
            mock_token_model.prisma.return_value.update_many = AsyncMock(return_value=0)
            mock_token_model.prisma.return_value.create = AsyncMock(
                return_value=MagicMock()
            )

            result = await create_link_token(
                CreateLinkTokenRequest(
                    platform=Platform.DISCORD,
                    platform_server_id="guild_123",
                    platform_user_id="user_456",
                    server_name="Test Server",
                ),
                x_bot_api_key="testkey",
            )

        assert result.token
        assert "guild_123" not in result.token  # token is random
        assert result.token in result.link_url
        assert "?platform=DISCORD" in result.link_url

    @pytest.mark.asyncio
    @patch.dict("os.environ", {"PLATFORM_BOT_API_KEY": "testkey"}, clear=False)
    async def test_create_token_409_if_already_linked(self):
        from backend.api.features.platform_linking.routes import create_link_token

        with patch(
            "backend.api.features.platform_linking.routes.find_server_link",
            new=AsyncMock(return_value=MagicMock()),  # server is already linked
        ):
            with pytest.raises(HTTPException) as exc_info:
                await create_link_token(
                    CreateLinkTokenRequest(
                        platform=Platform.DISCORD,
                        platform_server_id="guild_already_linked",
                        platform_user_id="user_456",
                    ),
                    x_bot_api_key="testkey",
                )

        assert exc_info.value.status_code == 409


class TestCreateUserLinkTokenEndpoint:
    @pytest.mark.asyncio
    @patch.dict("os.environ", {"PLATFORM_BOT_API_KEY": "testkey"}, clear=False)
    async def test_create_user_token_for_unlinked_user(self):
        from backend.api.features.platform_linking.routes import create_user_link_token

        with (
            patch(
                "backend.api.features.platform_linking.routes.find_user_link",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "backend.api.features.platform_linking.routes.transaction",
                new=_fake_transaction,
            ),
            patch(
                "backend.api.features.platform_linking.routes.PlatformLinkToken"
            ) as mock_token_model,
        ):
            mock_token_model.prisma.return_value.update_many = AsyncMock(return_value=0)
            mock_token_model.prisma.return_value.create = AsyncMock(
                return_value=MagicMock()
            )

            result = await create_user_link_token(
                CreateUserLinkTokenRequest(
                    platform=Platform.DISCORD,
                    platform_user_id="user_456",
                    platform_username="Bently",
                ),
                x_bot_api_key="testkey",
            )

        assert result.token
        assert result.token in result.link_url
        assert "?platform=DISCORD" in result.link_url

    @pytest.mark.asyncio
    @patch.dict("os.environ", {"PLATFORM_BOT_API_KEY": "testkey"}, clear=False)
    async def test_create_user_token_409_if_already_linked(self):
        from backend.api.features.platform_linking.routes import create_user_link_token

        with patch(
            "backend.api.features.platform_linking.routes.find_user_link",
            new=AsyncMock(return_value=MagicMock()),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await create_user_link_token(
                    CreateUserLinkTokenRequest(
                        platform=Platform.DISCORD,
                        platform_user_id="user_already_linked",
                    ),
                    x_bot_api_key="testkey",
                )

        assert exc_info.value.status_code == 409


class TestResolveUserEndpoint:
    @pytest.mark.asyncio
    @patch.dict("os.environ", {"PLATFORM_BOT_API_KEY": "testkey"}, clear=False)
    async def test_resolve_linked_user(self):
        from backend.api.features.platform_linking.routes import resolve_platform_user

        with patch(
            "backend.api.features.platform_linking.routes.find_user_link",
            new=AsyncMock(return_value=MagicMock(userId="autogpt-user-xyz")),
        ):
            result = await resolve_platform_user(
                ResolveUserRequest(
                    platform=Platform.DISCORD,
                    platform_user_id="user_456",
                ),
                x_bot_api_key="testkey",
            )

        assert result.linked is True

    @pytest.mark.asyncio
    @patch.dict("os.environ", {"PLATFORM_BOT_API_KEY": "testkey"}, clear=False)
    async def test_resolve_unlinked_user(self):
        from backend.api.features.platform_linking.routes import resolve_platform_user

        with patch(
            "backend.api.features.platform_linking.routes.find_user_link",
            new=AsyncMock(return_value=None),
        ):
            result = await resolve_platform_user(
                ResolveUserRequest(
                    platform=Platform.DISCORD,
                    platform_user_id="user_unknown",
                ),
                x_bot_api_key="testkey",
            )

        assert result.linked is False


class TestGetLinkTokenStatusEndpoint:
    """Tests for /tokens/{token}/status — pending / linked / expired / 404."""

    @pytest.mark.asyncio
    @patch.dict("os.environ", {"PLATFORM_BOT_API_KEY": "testkey"}, clear=False)
    async def test_not_found(self):
        from backend.api.features.platform_linking.routes import get_link_token_status

        with patch(
            "backend.api.features.platform_linking.routes.PlatformLinkToken"
        ) as mock_model:
            mock_model.prisma.return_value.find_unique = AsyncMock(return_value=None)
            with pytest.raises(HTTPException) as exc_info:
                await get_link_token_status(token="abc123", x_bot_api_key="testkey")
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    @patch.dict("os.environ", {"PLATFORM_BOT_API_KEY": "testkey"}, clear=False)
    async def test_pending(self):
        from datetime import datetime, timedelta, timezone

        from backend.api.features.platform_linking.routes import get_link_token_status

        future = datetime.now(timezone.utc) + timedelta(minutes=10)
        fake_token = MagicMock(usedAt=None, expiresAt=future)

        with patch(
            "backend.api.features.platform_linking.routes.PlatformLinkToken"
        ) as mock_model:
            mock_model.prisma.return_value.find_unique = AsyncMock(
                return_value=fake_token
            )
            result = await get_link_token_status(
                token="abc123", x_bot_api_key="testkey"
            )

        assert result.status == "pending"

    @pytest.mark.asyncio
    @patch.dict("os.environ", {"PLATFORM_BOT_API_KEY": "testkey"}, clear=False)
    async def test_expired_by_time(self):
        from datetime import datetime, timedelta, timezone

        from backend.api.features.platform_linking.routes import get_link_token_status

        past = datetime.now(timezone.utc) - timedelta(minutes=10)
        fake_token = MagicMock(usedAt=None, expiresAt=past)

        with patch(
            "backend.api.features.platform_linking.routes.PlatformLinkToken"
        ) as mock_model:
            mock_model.prisma.return_value.find_unique = AsyncMock(
                return_value=fake_token
            )
            result = await get_link_token_status(
                token="abc123", x_bot_api_key="testkey"
            )

        assert result.status == "expired"

    @pytest.mark.asyncio
    @patch.dict("os.environ", {"PLATFORM_BOT_API_KEY": "testkey"}, clear=False)
    async def test_used_with_user_link_reports_linked(self):
        from datetime import datetime, timezone

        from backend.api.features.platform_linking.routes import get_link_token_status

        fake_token = MagicMock(
            usedAt=datetime.now(timezone.utc),
            linkType=LinkType.USER.value,
            platform="DISCORD",
            platformUserId="user_456",
        )

        with (
            patch(
                "backend.api.features.platform_linking.routes.PlatformLinkToken"
            ) as mock_model,
            patch(
                "backend.api.features.platform_linking.routes.find_user_link",
                new=AsyncMock(return_value=MagicMock()),
            ),
        ):
            mock_model.prisma.return_value.find_unique = AsyncMock(
                return_value=fake_token
            )
            result = await get_link_token_status(
                token="abc123", x_bot_api_key="testkey"
            )

        assert result.status == "linked"

    @pytest.mark.asyncio
    @patch.dict("os.environ", {"PLATFORM_BOT_API_KEY": "testkey"}, clear=False)
    async def test_used_without_link_reports_expired(self):
        """A superseded token has usedAt set but no actual link row."""
        from datetime import datetime, timezone

        from backend.api.features.platform_linking.routes import get_link_token_status

        fake_token = MagicMock(
            usedAt=datetime.now(timezone.utc),
            linkType=LinkType.SERVER.value,
            platform="DISCORD",
            platformServerId="guild_123",
        )

        with (
            patch(
                "backend.api.features.platform_linking.routes.PlatformLinkToken"
            ) as mock_model,
            patch(
                "backend.api.features.platform_linking.routes.find_server_link",
                new=AsyncMock(return_value=None),
            ),
        ):
            mock_model.prisma.return_value.find_unique = AsyncMock(
                return_value=fake_token
            )
            result = await get_link_token_status(
                token="abc123", x_bot_api_key="testkey"
            )

        assert result.status == "expired"


class TestGetLinkTokenInfoEndpoint:
    """Tests for /tokens/{token}/info — no-auth display endpoint."""

    @pytest.mark.asyncio
    async def test_not_found(self):
        from backend.api.features.platform_linking.routes import get_link_token_info

        with patch(
            "backend.api.features.platform_linking.routes.PlatformLinkToken"
        ) as mock_model:
            mock_model.prisma.return_value.find_unique = AsyncMock(return_value=None)
            with pytest.raises(HTTPException) as exc_info:
                await get_link_token_info(token="abc123")
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_used_returns_404(self):
        from datetime import datetime, timezone

        from backend.api.features.platform_linking.routes import get_link_token_info

        fake_token = MagicMock(usedAt=datetime.now(timezone.utc))

        with patch(
            "backend.api.features.platform_linking.routes.PlatformLinkToken"
        ) as mock_model:
            mock_model.prisma.return_value.find_unique = AsyncMock(
                return_value=fake_token
            )
            with pytest.raises(HTTPException) as exc_info:
                await get_link_token_info(token="abc123")
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_expired_returns_410(self):
        from datetime import datetime, timedelta, timezone

        from backend.api.features.platform_linking.routes import get_link_token_info

        past = datetime.now(timezone.utc) - timedelta(minutes=5)
        fake_token = MagicMock(usedAt=None, expiresAt=past)

        with patch(
            "backend.api.features.platform_linking.routes.PlatformLinkToken"
        ) as mock_model:
            mock_model.prisma.return_value.find_unique = AsyncMock(
                return_value=fake_token
            )
            with pytest.raises(HTTPException) as exc_info:
                await get_link_token_info(token="abc123")
        assert exc_info.value.status_code == 410

    @pytest.mark.asyncio
    async def test_success_returns_display_info(self):
        from datetime import datetime, timedelta, timezone

        from backend.api.features.platform_linking.routes import get_link_token_info

        future = datetime.now(timezone.utc) + timedelta(minutes=10)
        fake_token = MagicMock(
            usedAt=None,
            expiresAt=future,
            platform="DISCORD",
            linkType=LinkType.SERVER.value,
            serverName="My Server",
        )

        with patch(
            "backend.api.features.platform_linking.routes.PlatformLinkToken"
        ) as mock_model:
            mock_model.prisma.return_value.find_unique = AsyncMock(
                return_value=fake_token
            )
            result = await get_link_token_info(token="abc123")

        assert result.platform == "DISCORD"
        assert result.link_type == LinkType.SERVER
        assert result.server_name == "My Server"


class TestConfirmLinkTokenEndpoint:
    """Tests for /tokens/{token}/confirm — the JWT-authed server-link confirmation."""

    @pytest.mark.asyncio
    async def test_not_found(self):
        from backend.api.features.platform_linking.routes import confirm_link_token

        with patch(
            "backend.api.features.platform_linking.routes.PlatformLinkToken"
        ) as mock_model:
            mock_model.prisma.return_value.find_unique = AsyncMock(return_value=None)
            with pytest.raises(HTTPException) as exc_info:
                await confirm_link_token(token="abc", user_id="user-1")
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_wrong_link_type_rejected(self):
        from backend.api.features.platform_linking.routes import confirm_link_token

        fake_token = MagicMock(linkType=LinkType.USER.value)
        with patch(
            "backend.api.features.platform_linking.routes.PlatformLinkToken"
        ) as mock_model:
            mock_model.prisma.return_value.find_unique = AsyncMock(
                return_value=fake_token
            )
            with pytest.raises(HTTPException) as exc_info:
                await confirm_link_token(token="abc", user_id="user-1")
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_already_used(self):
        from datetime import datetime, timezone

        from backend.api.features.platform_linking.routes import confirm_link_token

        fake_token = MagicMock(
            linkType=LinkType.SERVER.value,
            usedAt=datetime.now(timezone.utc),
        )
        with patch(
            "backend.api.features.platform_linking.routes.PlatformLinkToken"
        ) as mock_model:
            mock_model.prisma.return_value.find_unique = AsyncMock(
                return_value=fake_token
            )
            with pytest.raises(HTTPException) as exc_info:
                await confirm_link_token(token="abc", user_id="user-1")
        assert exc_info.value.status_code == 410

    @pytest.mark.asyncio
    async def test_expired(self):
        from datetime import datetime, timedelta, timezone

        from backend.api.features.platform_linking.routes import confirm_link_token

        fake_token = MagicMock(
            linkType=LinkType.SERVER.value,
            usedAt=None,
            expiresAt=datetime.now(timezone.utc) - timedelta(minutes=5),
        )
        with patch(
            "backend.api.features.platform_linking.routes.PlatformLinkToken"
        ) as mock_model:
            mock_model.prisma.return_value.find_unique = AsyncMock(
                return_value=fake_token
            )
            with pytest.raises(HTTPException) as exc_info:
                await confirm_link_token(token="abc", user_id="user-1")
        assert exc_info.value.status_code == 410

    @pytest.mark.asyncio
    async def test_already_linked_to_same_user(self):
        from datetime import datetime, timedelta, timezone

        from backend.api.features.platform_linking.routes import confirm_link_token

        fake_token = MagicMock(
            linkType=LinkType.SERVER.value,
            usedAt=None,
            expiresAt=datetime.now(timezone.utc) + timedelta(minutes=10),
            platform="DISCORD",
            platformServerId="guild_123",
        )
        existing = MagicMock(userId="user-1")

        with (
            patch(
                "backend.api.features.platform_linking.routes.PlatformLinkToken"
            ) as mock_model,
            patch(
                "backend.api.features.platform_linking.routes.find_server_link",
                new=AsyncMock(return_value=existing),
            ),
        ):
            mock_model.prisma.return_value.find_unique = AsyncMock(
                return_value=fake_token
            )
            with pytest.raises(HTTPException) as exc_info:
                await confirm_link_token(token="abc", user_id="user-1")
        assert exc_info.value.status_code == 409
        assert "your account" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_already_linked_to_other_user(self):
        from datetime import datetime, timedelta, timezone

        from backend.api.features.platform_linking.routes import confirm_link_token

        fake_token = MagicMock(
            linkType=LinkType.SERVER.value,
            usedAt=None,
            expiresAt=datetime.now(timezone.utc) + timedelta(minutes=10),
            platform="DISCORD",
            platformServerId="guild_123",
        )
        existing = MagicMock(userId="other-user")

        with (
            patch(
                "backend.api.features.platform_linking.routes.PlatformLinkToken"
            ) as mock_model,
            patch(
                "backend.api.features.platform_linking.routes.find_server_link",
                new=AsyncMock(return_value=existing),
            ),
        ):
            mock_model.prisma.return_value.find_unique = AsyncMock(
                return_value=fake_token
            )
            with pytest.raises(HTTPException) as exc_info:
                await confirm_link_token(token="abc", user_id="user-1")
        assert exc_info.value.status_code == 409
        assert "another" in exc_info.value.detail


class TestConfirmUserLinkTokenEndpoint:
    """Tests for /user-tokens/{token}/confirm — DM link confirmation."""

    @pytest.mark.asyncio
    async def test_not_found(self):
        from backend.api.features.platform_linking.routes import confirm_user_link_token

        with patch(
            "backend.api.features.platform_linking.routes.PlatformLinkToken"
        ) as mock_model:
            mock_model.prisma.return_value.find_unique = AsyncMock(return_value=None)
            with pytest.raises(HTTPException) as exc_info:
                await confirm_user_link_token(token="abc", user_id="user-1")
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_wrong_link_type_rejected(self):
        from backend.api.features.platform_linking.routes import confirm_user_link_token

        fake_token = MagicMock(linkType=LinkType.SERVER.value)
        with patch(
            "backend.api.features.platform_linking.routes.PlatformLinkToken"
        ) as mock_model:
            mock_model.prisma.return_value.find_unique = AsyncMock(
                return_value=fake_token
            )
            with pytest.raises(HTTPException) as exc_info:
                await confirm_user_link_token(token="abc", user_id="user-1")
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_expired(self):
        from datetime import datetime, timedelta, timezone

        from backend.api.features.platform_linking.routes import confirm_user_link_token

        fake_token = MagicMock(
            linkType=LinkType.USER.value,
            usedAt=None,
            expiresAt=datetime.now(timezone.utc) - timedelta(minutes=5),
        )
        with patch(
            "backend.api.features.platform_linking.routes.PlatformLinkToken"
        ) as mock_model:
            mock_model.prisma.return_value.find_unique = AsyncMock(
                return_value=fake_token
            )
            with pytest.raises(HTTPException) as exc_info:
                await confirm_user_link_token(token="abc", user_id="user-1")
        assert exc_info.value.status_code == 410

    @pytest.mark.asyncio
    async def test_already_linked_to_other_user(self):
        from datetime import datetime, timedelta, timezone

        from backend.api.features.platform_linking.routes import confirm_user_link_token

        fake_token = MagicMock(
            linkType=LinkType.USER.value,
            usedAt=None,
            expiresAt=datetime.now(timezone.utc) + timedelta(minutes=10),
            platform="DISCORD",
            platformUserId="user_456",
        )
        existing = MagicMock(userId="other-user")

        with (
            patch(
                "backend.api.features.platform_linking.routes.PlatformLinkToken"
            ) as mock_model,
            patch(
                "backend.api.features.platform_linking.routes.find_user_link",
                new=AsyncMock(return_value=existing),
            ),
        ):
            mock_model.prisma.return_value.find_unique = AsyncMock(
                return_value=fake_token
            )
            with pytest.raises(HTTPException) as exc_info:
                await confirm_user_link_token(token="abc", user_id="user-1")
        assert exc_info.value.status_code == 409
