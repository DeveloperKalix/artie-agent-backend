from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


class CreateLinkTokenBody(BaseModel):
    """Identifies the end user in Plaid Link (must be stable per user)."""

    user_id: str = Field(..., min_length=1, description="Your app user id (e.g. Supabase auth user id)")
    completion_redirect_uri: str | None = Field(
        None,
        description=(
            "Pass to activate Plaid Hosted Link. "
            "Plaid serves the UI at a secure.plaid.com URL and redirects here when done "
            "(e.g. 'artie://plaid-complete'). "
            "Response will include 'hosted_link_url' when set."
        ),
    )


class LinkTokenResponse(BaseModel):
    link_token: str
    expiration: str | None = None
    hosted_link_url: str | None = None

    @field_validator("expiration", mode="before")
    @classmethod
    def expiration_to_iso_string(cls, v: Any) -> str | None:
        if v is None:
            return None
        if isinstance(v, datetime):
            return v.isoformat()
        if isinstance(v, str):
            return v
        return str(v)


class CompleteHostedLinkBody(BaseModel):
    """Body for the Hosted Link completion endpoint."""

    link_token: str = Field(..., min_length=1, description="The link_token returned by /plaid/link_token")
    user_id: str = Field(..., min_length=1, description="Same user_id used when creating the link token")


class CompleteHostedLinkResponse(BaseModel):
    success: bool
    status: str  # "complete" | "incomplete" | "no_session"
    item_id: str | None = None
    institution_id: str | None = None
    institution_name: str | None = None


class ExchangePublicTokenBody(BaseModel):
    public_token: str = Field(..., min_length=1)
    user_id: str = Field(..., min_length=1, description="Same user_id used when creating the link token")


class PlaidItemStored(BaseModel):
    item_id: str
    institution_id: str | None = None
    institution_name: str | None = None


class PlaidAccountsResponse(BaseModel):
    """Plaid account objects with ``item_id`` for the linked Item."""

    accounts: list[dict[str, Any]]
