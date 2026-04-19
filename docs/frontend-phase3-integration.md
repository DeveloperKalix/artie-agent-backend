# Artie Backend — Phase 3 Frontend Integration Guide

Audience: the Artie Expo/React-Native frontend AI.
Scope: everything that changed in Phase 3 (agent orchestration, memory, skills,
recommendations, conversation voice pipeline) plus a checklist of screens that
need to be updated or created.

All endpoints below are mounted under both root (`/…`) and `/api/v1/…`. Use
`/api/v1/…` for production calls (mirrors the convention used by the existing
Plaid/SnapTrade endpoints).

All multi-tenant endpoints require the header:

```
X-User-Id: <app user id>
```

The value must match the id used for Plaid/SnapTrade registrations — if you
store a Supabase auth user id today, keep using that.

---

## 1. High-level behavior changes you must account for

### 1.1 Voice upload moved

- **Removed**: `POST /process-voice` (returned `{intent, status}` only).
- **Replacement**: `POST /api/v1/conversations/{conversation_id}/messages/voice`.

The voice endpoint now:

1. Transcribes the audio with Groq Whisper.
2. Persists the user message with `metadata.input = "voice"` and the
   transcript in both `content` and `transcript` fields.
3. Runs the `AgentOrchestrator` (skill detection → agent reply).
4. Returns the **assistant** `Message` (durable, with `id`, `created_at`,
   `metadata`).

If you still have UI calling `/process-voice`, it will 404. Replace it with
the conversation-scoped endpoint.

### 1.2 Conversations replace ad-hoc chat

Every user-facing agent interaction lives inside a `Conversation`:

- The user may have many conversations (like ChatGPT threads).
- Each conversation owns an ordered list of `Message`s.
- Messages have a `role` (`user` | `assistant` | `system` | `tool`),
  `content`, and optional `metadata` the UI should inspect to style the
  message (skill ack vs. normal reply vs. error).

### 1.3 Experience level

Every user has a `UserProfile` with `experience_level ∈ {novice, intermediate,
veteran}`. The recommendation agent tailors tone and detail to it. The UI
should expose this in a settings screen.

### 1.4 /skill command (free-form memory)

Whenever a user message begins with `/skill`, the orchestrator routes it to
the `SkillSystem` instead of the LLM:

- `/skill I prefer dividend stocks over growth` → appends a memory note.
- `/skill set-level veteran` → also updates `experience_level`.

The assistant reply comes back with `metadata.source = "skill"` and
`metadata.kind ∈ {note_appended, level_updated, invalid}`. The UI should
render these distinctly (e.g. a subtle "✓ Remembered" affordance).

### 1.5 On-demand recommendations (non-conversational)

`POST /api/v1/recommendations` returns structured, JSON-validated portfolio
recommendations without touching a conversation. Call it from the home/
dashboard screen on load or pull-to-refresh.

---

## 2. New / changed endpoints (shapes)

Everything is JSON unless stated otherwise.

### 2.1 Conversations

#### `GET /api/v1/conversations`

Headers: `X-User-Id`

Response `200`:

```json
{
  "conversations": [
    {
      "id": "uuid",
      "user_id": "user-123",
      "title": "Morning check-in",
      "created_at": "2026-04-19T04:30:00Z",
      "updated_at": "2026-04-19T04:35:00Z",
      "last_message_at": "2026-04-19T04:35:00Z"
    }
  ]
}
```

Ordered by `last_message_at DESC` (null last).

#### `POST /api/v1/conversations`

Headers: `X-User-Id`
Body:

```json
{ "title": "Morning check-in" }   // title optional, may be null
```

Response `200`: a single `Conversation` (same shape as above, no messages yet).

#### `GET /api/v1/conversations/{id}/messages?limit=50`

Headers: `X-User-Id`

Response `200`:

```json
{
  "messages": [
    {
      "id": "uuid",
      "conversation_id": "uuid",
      "role": "user",
      "content": "/skill I prefer dividends",
      "audio_url": null,
      "transcript": null,
      "metadata": { "input": "text" },
      "created_at": "..."
    },
    {
      "id": "uuid",
      "conversation_id": "uuid",
      "role": "assistant",
      "content": "Got it — I'll remember: \u201cI prefer dividends\u201d",
      "metadata": {
        "source": "skill",
        "kind": "note_appended",
        "note_id": "uuid"
      },
      "created_at": "..."
    }
  ]
}
```

Errors: `404` if the conversation id doesn't exist, `403` if it belongs to
another user. Both are safe to surface as "conversation not found".

#### `POST /api/v1/conversations/{id}/messages`

Append a **text** message. The backend persists the user turn, runs the
orchestrator, persists the assistant reply, and returns the **assistant
message**.

Body:

```json
{ "content": "What's up with TSLA today?" }
```

Response `200`: a `Message` (assistant). Inspect `metadata.source`:

| `metadata.source` | Meaning                                           |
| ----------------- | ------------------------------------------------- |
| `skill`           | Response from `/skill …`. See `metadata.kind`.    |
| `chat`            | Normal agent reply (free-form text).              |
| `error`           | Agent crashed; reply is a fallback apology.       |

After receiving the response, re-fetch messages or optimistically append both
the user turn (you already had its text) and the returned assistant turn.

#### `POST /api/v1/conversations/{id}/messages/voice`

`multipart/form-data` with a single field `file` (audio). Recommended formats:
`m4a`, `mp3`, `wav`, `webm`. Max 25 MB (Whisper's limit).

Returns the assistant `Message`, same shape as the text endpoint. The user
turn is persisted server-side with `metadata.input = "voice"` and the
transcript visible in the `transcript` column — you can fetch it via
`GET /conversations/{id}/messages` after the POST resolves.

Errors:

| Status | Meaning                                         |
| ------ | ----------------------------------------------- |
| `403`  | Conversation belongs to another user            |
| `404`  | Conversation id not found                       |
| `422`  | Transcription returned empty                    |
| `502`  | Whisper call failed                             |

### 2.2 Skills / memory

#### `GET /api/v1/skills?limit=50`

Headers: `X-User-Id`

Response `200`:

```json
{
  "notes": [
    {
      "id": "uuid",
      "user_id": "user-123",
      "content": "I prefer dividends",
      "source": "skill",
      "created_at": "..."
    }
  ]
}
```

Ordered newest-first.

#### `POST /api/v1/skills`

Headers: `X-User-Id`
Body:

```json
{ "content": "I prefer dividend stocks over growth" }
```

- If `content` starts with `/skill`, the backend honors subcommands like
  `/skill set-level veteran`.
- Otherwise the whole body is stored as a free-form note.

Response `200`:

```json
{
  "note": { ...MemoryNote... },          // null if kind is invalid/level_updated
  "reply": "Remembered: \u201cI prefer dividends\u201d",
  "kind": "note_appended"                // note_appended | level_updated | invalid
}
```

Render `reply` to the user as a toast or banner.

#### `DELETE /api/v1/skills/{note_id}`

Headers: `X-User-Id`
Response `200`: `{ "deleted": true }`
Response `404`: `{ "detail": "note not found" }`

### 2.3 Profile

#### `GET /api/v1/profile`

Headers: `X-User-Id`

Returns the caller's profile (auto-created as `novice` on first hit):

```json
{
  "user_id": "user-123",
  "experience_level": "novice",
  "onboarded_at": null,
  "created_at": "...",
  "updated_at": "..."
}
```

#### `PATCH /api/v1/profile`

Headers: `X-User-Id`
Body:

```json
{ "experience_level": "intermediate" }
```

Response `200`: updated `UserProfile`.

### 2.4 News (read-only)

Unchanged since Phase 2, but reiterated for completeness.

#### `GET /api/v1/news?tickers=AAPL,BTC-USD&query=earnings&limit=20`

No `X-User-Id` required (public read against the ingested `news_items` table).

Response `200`:

```json
{
  "items": [
    {
      "id": "uuid",
      "url": "https://...",
      "title": "Apple beats earnings",
      "summary": "...",
      "source": "yahoo_rss",
      "tickers": ["AAPL"],
      "published_at": "...",
      "sentiment": null
    }
  ],
  "total": 20
}
```

Priority: `tickers > query > recent`. Supplying both uses `tickers`.

### 2.5 Recommendations (on-demand, structured)

#### `POST /api/v1/recommendations`

Headers: `X-User-Id`
Body: none.

Response `200`:

```json
{
  "user_id": "user-123",
  "generated_at": "2026-04-19T04:40:00Z",
  "recommendations": [
    {
      "ticker": "AAPL",
      "action": "hold",
      "confidence": "medium",
      "explanation": "Earnings beat offsets iPhone softness …",
      "supporting_articles": [
        {
          "id": "uuid",
          "url": "https://...",
          "title": "Apple beats earnings",
          "summary": "...",
          "source": "yahoo_rss",
          "tickers": ["AAPL"],
          "published_at": "...",
          "sentiment": null
        }
      ]
    }
  ],
  "disclaimer": "This is not financial advice. …"
}
```

- `action` ∈ `buy | sell | hold | increase | reduce`.
- `confidence` ∈ `high | medium | low`.
- `supporting_articles` is resolved server-side from the same `news_items`
  pool — never trust any other URLs. If empty, the agent chose not to cite.
- `recommendations` may legitimately be empty (agent sees nothing actionable
  or user has no positions). Always render the `disclaimer`.

Errors: `500` with `detail` string on unexpected failure; it's safe to show
"Couldn't generate recommendations right now. Try again." and keep the last
cached response.

---

## 3. Request / response contracts at a glance

Keep these shapes in one central types module on the frontend. Here they are
in TypeScript:

```ts
export type MessageRole = 'user' | 'assistant' | 'system' | 'tool';

export interface Message {
  id: string;
  conversation_id: string;
  role: MessageRole;
  content: string;
  audio_url?: string | null;
  transcript?: string | null;
  metadata?: Record<string, unknown> | null;
  created_at: string;
}

export interface Conversation {
  id: string;
  user_id: string;
  title?: string | null;
  created_at: string;
  updated_at: string;
  last_message_at?: string | null;
}

export type ExperienceLevel = 'novice' | 'intermediate' | 'veteran';

export interface UserProfile {
  user_id: string;
  experience_level: ExperienceLevel;
  onboarded_at?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
}

export interface MemoryNote {
  id: string;
  user_id: string;
  content: string;
  source: 'skill' | 'system' | 'inferred';
  created_at: string;
}

export interface NewsItem {
  id?: string | null;
  url: string;
  title: string;
  summary?: string | null;
  source: string;
  tickers: string[];
  published_at?: string | null;
  sentiment?: 'positive' | 'negative' | 'neutral' | null;
}

export type RecommendationAction =
  | 'buy' | 'sell' | 'hold' | 'increase' | 'reduce';

export type Confidence = 'high' | 'medium' | 'low';

export interface PortfolioRecommendation {
  ticker: string;
  action: RecommendationAction;
  confidence: Confidence;
  explanation: string;
  supporting_articles: NewsItem[];
}

export interface RecommendationResponse {
  user_id: string;
  generated_at: string;
  recommendations: PortfolioRecommendation[];
  disclaimer: string;
}
```

---

## 4. UI work items

### 4.1 Chat surface

- **Conversations list screen** (sidebar / dedicated tab):
  - Fetch `GET /conversations`.
  - New conversation button → `POST /conversations` with optional title.
  - Selecting a conversation navigates to the conversation screen.
- **Conversation screen**:
  - Fetch `GET /conversations/{id}/messages` on mount.
  - Render messages differently by `role`:
    - `user` → right-aligned bubble.
    - `assistant` → left-aligned bubble.
  - Render assistant messages differently by `metadata.source`:
    - `skill` + `kind=note_appended` → small badge ("Remembered 💾") + muted style.
    - `skill` + `kind=level_updated` → "Experience level updated" banner.
    - `error` → red/warning background.
    - `chat` → default.
  - Compose box:
    - Text submit → `POST /conversations/{id}/messages`, then append the
      user's optimistic bubble and the returned assistant message.
    - Mic button → record audio → `POST /conversations/{id}/messages/voice`
      as `multipart/form-data` with field name `file`.
    - Detect a leading `/skill` in text and style the bubble as a skill turn
      before the response comes back (optional polish).

### 4.2 Settings / profile

- Add a settings screen that reads `GET /profile` and shows a segmented
  control for `experience_level`.
- Changing it calls `PATCH /profile` — the agent will pick it up on the next
  message.

### 4.3 Memory manager

- Dedicated "Memory" screen:
  - `GET /skills` → list of notes.
  - Delete icon per row → `DELETE /skills/{id}`.
  - "Add note" composer → `POST /skills` (raw content, no `/skill` prefix
    needed). The backend accepts either form.

### 4.4 Home / dashboard recommendations widget

- On tab focus or pull-to-refresh call `POST /recommendations`.
- Render each `PortfolioRecommendation` as a card:
  - Header: ticker + colored pill for `action`.
  - Subheader: `confidence` chip.
  - Body: `explanation`.
  - Footer: chips for each `supporting_articles[*]` (tap opens `url` in the
    in-app browser).
- Always show `disclaimer` at the bottom of the list.
- Handle `recommendations.length === 0`: show "Nothing actionable since your
  last check" instead of an empty card.

### 4.5 Remove `/process-voice` usage

Search the frontend for `process-voice` / `processVoice` — any call to that
path now 404s. Replace with a two-step flow:

1. Ensure there's an active conversation (create one lazily the first time).
2. `POST /api/v1/conversations/{id}/messages/voice` with the recording.

The assistant reply comes back in the HTTP response; show it immediately and
also refresh the message list so the server-side user message appears.

---

## 5. Error handling patterns

| HTTP status | Where                                     | Recommended UI                        |
| ----------- | ----------------------------------------- | ------------------------------------- |
| `422`       | Missing `X-User-Id`, empty body, bad file | Toast with the `detail` string        |
| `403`       | Conversation ownership mismatch           | "Conversation not available"          |
| `404`       | Unknown conversation / note               | Navigate back to list + refresh       |
| `502`       | Upstream transcription / LLM failure      | "Something went wrong. Try again."    |
| `500`       | Backend crash                             | Same, with silent error reporting     |

For `POST /recommendations` specifically: keep showing the last successful
response if the request fails. Agents can hiccup; don't wipe the screen.

---

## 6. Suggested order of work

1. Add the TypeScript types above.
2. Wire the conversations/messages screens (fetch + append text message).
3. Swap the mic button to the new voice endpoint. Delete `/process-voice`.
4. Add the settings screen (profile GET/PATCH).
5. Add the memory manager screen.
6. Add the recommendations widget on the home screen.
7. Polish: skill message badges, experience-level segmented control styling,
   recommendation card affordances.

---

## 7. Smoke test checklist (manual)

Run the backend locally, then from the frontend:

1. `POST /api/v1/conversations` with a test user id → get an id back.
2. `POST /api/v1/conversations/{id}/messages` with `{"content": "/skill set-level veteran"}` → expect `metadata.kind = "level_updated"`.
3. `GET /api/v1/profile` → `experience_level === "veteran"`.
4. `POST /api/v1/conversations/{id}/messages` with `{"content": "What should I do about AAPL?"}` → expect a normal chat reply with `metadata.source = "chat"`.
5. `POST /api/v1/recommendations` → expect a `RecommendationResponse`. If the
   user has no SnapTrade positions, `recommendations` may be empty — this is
   correct behavior.
6. `POST /api/v1/conversations/{id}/messages/voice` with a small `.m4a` →
   expect a transcribed user message and an assistant reply.
7. `GET /api/v1/skills` → includes any notes appended via `/skill`.
8. `DELETE /api/v1/skills/{note_id}` → `{ "deleted": true }`.
