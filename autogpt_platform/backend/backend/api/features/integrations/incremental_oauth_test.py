"""Tests for incremental OAuth authorization (scope upgrade)."""

from unittest.mock import AsyncMock, MagicMock, patch

import fastapi
import fastapi.testclient
import pytest
from pydantic import SecretStr

from backend.api.features.integrations.router import router
from backend.data.model import APIKeyCredentials, OAuth2Credentials, OAuthState

app = fastapi.FastAPI()
app.include_router(router)
client = fastapi.testclient.TestClient(app)

TEST_USER_ID = "test-user-id"


def _make_google_oauth2_cred(
    cred_id: str = "google-cred-1",
    scopes: list[str] | None = None,
    username: str = "alice@gmail.com",
    title: str = "My Google",
) -> OAuth2Credentials:
    return OAuth2Credentials(
        id=cred_id,
        provider="google",
        title=title,
        access_token=SecretStr("ya29.access-token"),
        refresh_token=SecretStr("1//refresh-token"),
        scopes=scopes or ["https://www.googleapis.com/auth/gmail.readonly"],
        username=username,
        access_token_expires_at=9999999999,
    )


def _make_github_oauth2_cred(
    cred_id: str = "github-cred-1",
    scopes: list[str] | None = None,
    username: str = "alice",
    title: str = "My GitHub",
) -> OAuth2Credentials:
    return OAuth2Credentials(
        id=cred_id,
        provider="github",
        title=title,
        access_token=SecretStr("ghp_access_token"),
        refresh_token=SecretStr("ghp_refresh_token"),
        scopes=scopes or ["repo"],
        username=username,
    )


@pytest.fixture(autouse=True)
def setup_auth(mock_jwt_user):
    from autogpt_libs.auth.jwt_utils import get_jwt_payload

    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    yield
    app.dependency_overrides.clear()


# ==================== OAuthState model tests ==================== #


class TestOAuthStateCredentialId:
    """OAuthState model should support a credential_id field for upgrades."""

    def test_oauth_state_accepts_credential_id(self):
        state = OAuthState(
            token="abc",
            provider="google",
            expires_at=9999999999,
            scopes=["openid"],
            credential_id="existing-cred-id",
        )
        assert state.credential_id == "existing-cred-id"

    def test_oauth_state_defaults_credential_id_none(self):
        state = OAuthState(
            token="abc",
            provider="google",
            expires_at=9999999999,
            scopes=["openid"],
        )
        assert state.credential_id is None


# ==================== Login endpoint tests ==================== #


class TestIncrementalOAuthLogin:
    """Tests for the login endpoint with credential_id parameter."""

    def test_login_with_credential_id_stores_in_state(self):
        """Login with credential_id should pass it through to store_state_token."""
        existing = _make_google_oauth2_cred()
        handler = MagicMock()
        handler.get_login_url.return_value = "https://accounts.google.com/auth"

        with (
            patch(
                "backend.api.features.integrations.router._get_provider_oauth_handler",
                return_value=handler,
            ),
            patch("backend.api.features.integrations.router.creds_manager") as mock_mgr,
        ):
            mock_mgr.store.get_creds_by_id = AsyncMock(return_value=existing)
            mock_mgr.store.store_state_token = AsyncMock(
                return_value=("state-token", "code-challenge")
            )

            resp = client.get(
                "/google/login",
                params={
                    "scopes": "https://www.googleapis.com/auth/calendar.readonly",
                    "credential_id": "google-cred-1",
                },
            )

        assert resp.status_code == 200
        # Verify store_state_token was called with credential_id
        call_kwargs = mock_mgr.store.store_state_token.call_args
        assert call_kwargs.kwargs.get("credential_id") == "google-cred-1" or (
            len(call_kwargs.args) > 3 and call_kwargs.args[3] == "google-cred-1"
        )

    def test_login_github_unions_scopes_for_upgrade(self):
        """For GitHub, login should request union of existing + new scopes."""
        existing = _make_github_oauth2_cred(scopes=["repo"])
        handler = MagicMock()
        handler.get_login_url.return_value = "https://github.com/login/oauth/authorize"

        with (
            patch(
                "backend.api.features.integrations.router._get_provider_oauth_handler",
                return_value=handler,
            ),
            patch("backend.api.features.integrations.router.creds_manager") as mock_mgr,
        ):
            mock_mgr.store.get_creds_by_id = AsyncMock(return_value=existing)
            mock_mgr.store.store_state_token = AsyncMock(
                return_value=("state-token", "code-challenge")
            )

            resp = client.get(
                "/github/login",
                params={
                    "scopes": "read:org",
                    "credential_id": "github-cred-1",
                },
            )

        assert resp.status_code == 200
        # The scopes passed to get_login_url should be the union
        login_scopes = handler.get_login_url.call_args[0][0]
        assert set(login_scopes) == {"repo", "read:org"}

    def test_login_google_keeps_requested_scopes_only(self):
        """For Google, login should use only the new scopes (include_granted_scopes handles merging)."""
        existing = _make_google_oauth2_cred(
            scopes=["https://www.googleapis.com/auth/gmail.readonly"]
        )
        handler = MagicMock()
        handler.get_login_url.return_value = "https://accounts.google.com/auth"

        with (
            patch(
                "backend.api.features.integrations.router._get_provider_oauth_handler",
                return_value=handler,
            ),
            patch("backend.api.features.integrations.router.creds_manager") as mock_mgr,
        ):
            mock_mgr.store.get_creds_by_id = AsyncMock(return_value=existing)
            mock_mgr.store.store_state_token = AsyncMock(
                return_value=("state-token", "code-challenge")
            )

            resp = client.get(
                "/google/login",
                params={
                    "scopes": "https://www.googleapis.com/auth/calendar.readonly",
                    "credential_id": "google-cred-1",
                },
            )

        assert resp.status_code == 200
        login_scopes = handler.get_login_url.call_args[0][0]
        # Google should NOT union scopes in the login URL
        assert "https://www.googleapis.com/auth/calendar.readonly" in login_scopes
        assert "https://www.googleapis.com/auth/gmail.readonly" not in login_scopes
        # Verify credential_id was passed through to store_state_token
        call_kwargs = mock_mgr.store.store_state_token.call_args
        assert call_kwargs.kwargs.get("credential_id") == "google-cred-1"

    def test_login_credential_not_found_returns_404(self):
        handler = MagicMock()
        with (
            patch(
                "backend.api.features.integrations.router._get_provider_oauth_handler",
                return_value=handler,
            ),
            patch("backend.api.features.integrations.router.creds_manager") as mock_mgr,
        ):
            mock_mgr.store.get_creds_by_id = AsyncMock(return_value=None)

            resp = client.get(
                "/google/login",
                params={
                    "scopes": "openid",
                    "credential_id": "nonexistent",
                },
            )

        assert resp.status_code == 404

    def test_login_credential_provider_mismatch_returns_400(self):
        """credential_id pointing to a Google cred when URL says github -> 400."""
        google_cred = _make_google_oauth2_cred()
        handler = MagicMock()

        with (
            patch(
                "backend.api.features.integrations.router._get_provider_oauth_handler",
                return_value=handler,
            ),
            patch("backend.api.features.integrations.router.creds_manager") as mock_mgr,
        ):
            mock_mgr.store.get_creds_by_id = AsyncMock(return_value=google_cred)

            resp = client.get(
                "/github/login",
                params={
                    "scopes": "repo",
                    "credential_id": "google-cred-1",
                },
            )

        assert resp.status_code == 400

    def test_login_non_oauth2_credential_returns_400(self):
        """credential_id pointing to an API key credential -> 400."""
        api_key_cred = APIKeyCredentials(
            id="apikey-1",
            provider="github",
            title="API Key",
            api_key=SecretStr("ghp_key"),
        )
        handler = MagicMock()

        with (
            patch(
                "backend.api.features.integrations.router._get_provider_oauth_handler",
                return_value=handler,
            ),
            patch("backend.api.features.integrations.router.creds_manager") as mock_mgr,
        ):
            mock_mgr.store.get_creds_by_id = AsyncMock(return_value=api_key_cred)

            resp = client.get(
                "/github/login",
                params={
                    "scopes": "repo",
                    "credential_id": "apikey-1",
                },
            )

        assert resp.status_code == 400


# ==================== Callback endpoint tests ==================== #


class TestIncrementalOAuthCallback:
    """Tests for the callback endpoint when upgrading credentials."""

    def _make_state_with_credential_id(
        self,
        credential_id: str,
        scopes: list[str] | None = None,
        provider: str = "google",
    ) -> OAuthState:
        return OAuthState(
            token="state-token",
            provider=provider,
            expires_at=9999999999,
            scopes=scopes or ["https://www.googleapis.com/auth/calendar.readonly"],
            credential_id=credential_id,
        )

    def test_callback_upgrades_existing_credential(self):
        """When state has credential_id, should update existing credential."""
        existing = _make_google_oauth2_cred(
            scopes=["https://www.googleapis.com/auth/gmail.readonly"]
        )
        new_cred = _make_google_oauth2_cred(
            scopes=[
                "https://www.googleapis.com/auth/gmail.readonly",
                "https://www.googleapis.com/auth/calendar.readonly",
            ]
        )
        state = self._make_state_with_credential_id("google-cred-1")
        handler = MagicMock()
        handler.exchange_code_for_tokens = AsyncMock(return_value=new_cred)
        handler.handle_default_scopes.return_value = state.scopes

        with (
            patch(
                "backend.api.features.integrations.router._get_provider_oauth_handler",
                return_value=handler,
            ),
            patch("backend.api.features.integrations.router.creds_manager") as mock_mgr,
        ):
            mock_mgr.store.verify_state_token = AsyncMock(return_value=state)
            mock_mgr.store.get_creds_by_id = AsyncMock(return_value=existing)
            mock_mgr.update = AsyncMock()
            mock_mgr.create = AsyncMock()

            resp = client.post(
                "/google/callback",
                json={"code": "auth-code", "state_token": "state-token"},
            )

        assert resp.status_code == 200
        # Should call update, not create
        mock_mgr.update.assert_called_once()
        mock_mgr.create.assert_not_called()

    def test_callback_upgrade_merges_scopes(self):
        """Upgraded credential should have union of old + new scopes."""
        existing = _make_google_oauth2_cred(
            scopes=["https://www.googleapis.com/auth/gmail.readonly"]
        )
        new_cred = _make_google_oauth2_cred(
            scopes=[
                "https://www.googleapis.com/auth/gmail.readonly",
                "https://www.googleapis.com/auth/calendar.readonly",
            ]
        )
        state = self._make_state_with_credential_id("google-cred-1")
        handler = MagicMock()
        handler.exchange_code_for_tokens = AsyncMock(return_value=new_cred)
        handler.handle_default_scopes.return_value = state.scopes

        with (
            patch(
                "backend.api.features.integrations.router._get_provider_oauth_handler",
                return_value=handler,
            ),
            patch("backend.api.features.integrations.router.creds_manager") as mock_mgr,
        ):
            mock_mgr.store.verify_state_token = AsyncMock(return_value=state)
            mock_mgr.store.get_creds_by_id = AsyncMock(return_value=existing)
            mock_mgr.update = AsyncMock()

            resp = client.post(
                "/google/callback",
                json={"code": "auth-code", "state_token": "state-token"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert set(data["scopes"]) == {
            "https://www.googleapis.com/auth/gmail.readonly",
            "https://www.googleapis.com/auth/calendar.readonly",
        }

    def test_callback_upgrade_preserves_id_and_title(self):
        """Upgraded credential should keep its original ID and title."""
        existing = _make_google_oauth2_cred(
            cred_id="original-id", title="My Work Google"
        )
        new_cred = _make_google_oauth2_cred(cred_id="new-id-from-exchange")
        state = self._make_state_with_credential_id("original-id")
        handler = MagicMock()
        handler.exchange_code_for_tokens = AsyncMock(return_value=new_cred)
        handler.handle_default_scopes.return_value = state.scopes

        with (
            patch(
                "backend.api.features.integrations.router._get_provider_oauth_handler",
                return_value=handler,
            ),
            patch("backend.api.features.integrations.router.creds_manager") as mock_mgr,
        ):
            mock_mgr.store.verify_state_token = AsyncMock(return_value=state)
            mock_mgr.store.get_creds_by_id = AsyncMock(return_value=existing)
            mock_mgr.update = AsyncMock()

            resp = client.post(
                "/google/callback",
                json={"code": "auth-code", "state_token": "state-token"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "original-id"
        assert data["title"] == "My Work Google"

    def test_callback_upgrade_rejects_username_mismatch(self):
        """Should reject if the new auth returns a different username."""
        existing = _make_google_oauth2_cred(username="alice@gmail.com")
        new_cred = _make_google_oauth2_cred(username="bob@gmail.com")
        state = self._make_state_with_credential_id("google-cred-1")
        handler = MagicMock()
        handler.exchange_code_for_tokens = AsyncMock(return_value=new_cred)
        handler.handle_default_scopes.return_value = state.scopes

        with (
            patch(
                "backend.api.features.integrations.router._get_provider_oauth_handler",
                return_value=handler,
            ),
            patch("backend.api.features.integrations.router.creds_manager") as mock_mgr,
        ):
            mock_mgr.store.verify_state_token = AsyncMock(return_value=state)
            mock_mgr.store.get_creds_by_id = AsyncMock(return_value=existing)

            resp = client.post(
                "/google/callback",
                json={"code": "auth-code", "state_token": "state-token"},
            )

        assert resp.status_code == 400
        assert "username" in resp.json()["detail"].lower()

    def test_callback_implicit_merge_same_provider_username(self):
        """Without credential_id, should auto-merge when same provider+username exists."""
        existing = _make_google_oauth2_cred(
            scopes=["https://www.googleapis.com/auth/gmail.readonly"]
        )
        new_cred = _make_google_oauth2_cred(
            cred_id="new-cred-id",
            scopes=[
                "https://www.googleapis.com/auth/gmail.readonly",
                "https://www.googleapis.com/auth/calendar.readonly",
            ],
            username="alice@gmail.com",
        )
        # State WITHOUT credential_id
        state = OAuthState(
            token="state-token",
            provider="google",
            expires_at=9999999999,
            scopes=["https://www.googleapis.com/auth/calendar.readonly"],
        )
        handler = MagicMock()
        handler.exchange_code_for_tokens = AsyncMock(return_value=new_cred)
        handler.handle_default_scopes.return_value = state.scopes

        with (
            patch(
                "backend.api.features.integrations.router._get_provider_oauth_handler",
                return_value=handler,
            ),
            patch("backend.api.features.integrations.router.creds_manager") as mock_mgr,
        ):
            mock_mgr.store.verify_state_token = AsyncMock(return_value=state)
            mock_mgr.store.get_creds_by_provider = AsyncMock(return_value=[existing])
            mock_mgr.store.get_creds_by_id = AsyncMock(return_value=existing)
            mock_mgr.update = AsyncMock()
            mock_mgr.create = AsyncMock()

            resp = client.post(
                "/google/callback",
                json={"code": "auth-code", "state_token": "state-token"},
            )

        assert resp.status_code == 200
        # Should update the existing credential, not create a new one
        mock_mgr.update.assert_called_once()
        mock_mgr.create.assert_not_called()
        # The returned ID should be the existing credential's ID
        data = resp.json()
        assert data["id"] == "google-cred-1"

    def test_callback_no_implicit_merge_different_username(self):
        """Without credential_id, different username should create new credential."""
        existing = _make_google_oauth2_cred(username="alice@gmail.com")
        new_cred = _make_google_oauth2_cred(
            cred_id="new-cred-id",
            username="bob@gmail.com",
        )
        state = OAuthState(
            token="state-token",
            provider="google",
            expires_at=9999999999,
            scopes=["https://www.googleapis.com/auth/gmail.readonly"],
        )
        handler = MagicMock()
        handler.exchange_code_for_tokens = AsyncMock(return_value=new_cred)
        handler.handle_default_scopes.return_value = state.scopes

        with (
            patch(
                "backend.api.features.integrations.router._get_provider_oauth_handler",
                return_value=handler,
            ),
            patch("backend.api.features.integrations.router.creds_manager") as mock_mgr,
        ):
            mock_mgr.store.verify_state_token = AsyncMock(return_value=state)
            mock_mgr.store.get_creds_by_provider = AsyncMock(return_value=[existing])
            mock_mgr.create = AsyncMock()
            mock_mgr.update = AsyncMock()

            resp = client.post(
                "/google/callback",
                json={"code": "auth-code", "state_token": "state-token"},
            )

        assert resp.status_code == 200
        mock_mgr.create.assert_called_once()
        mock_mgr.update.assert_not_called()
        # Verify the implicit merge lookup was attempted
        mock_mgr.store.get_creds_by_provider.assert_called_once()

    def test_callback_creates_new_when_no_existing(self):
        """Without credential_id and no matching credential, creates new."""
        new_cred = _make_google_oauth2_cred()
        state = OAuthState(
            token="state-token",
            provider="google",
            expires_at=9999999999,
            scopes=["https://www.googleapis.com/auth/gmail.readonly"],
        )
        handler = MagicMock()
        handler.exchange_code_for_tokens = AsyncMock(return_value=new_cred)
        handler.handle_default_scopes.return_value = state.scopes

        with (
            patch(
                "backend.api.features.integrations.router._get_provider_oauth_handler",
                return_value=handler,
            ),
            patch("backend.api.features.integrations.router.creds_manager") as mock_mgr,
        ):
            mock_mgr.store.verify_state_token = AsyncMock(return_value=state)
            mock_mgr.store.get_creds_by_provider = AsyncMock(return_value=[])
            mock_mgr.create = AsyncMock()
            mock_mgr.update = AsyncMock()

            resp = client.post(
                "/google/callback",
                json={"code": "auth-code", "state_token": "state-token"},
            )

        assert resp.status_code == 200
        mock_mgr.create.assert_called_once()
        mock_mgr.update.assert_not_called()
        # Verify the implicit merge lookup was attempted
        mock_mgr.store.get_creds_by_provider.assert_called_once()
