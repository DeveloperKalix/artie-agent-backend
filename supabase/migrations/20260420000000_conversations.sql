-- Conversations + Messages for the user-facing agent.
-- Each conversation belongs to an app user and has its own ordered message log.
-- app_user_id is kept as text to mirror snaptrade_users (frontend sends X-User-Id).

create table if not exists public.conversations (
  id              uuid primary key default gen_random_uuid(),
  user_id         text not null,
  title           text,
  created_at      timestamptz not null default now(),
  updated_at      timestamptz not null default now(),
  last_message_at timestamptz
);

create index if not exists conversations_user_recent_idx
  on public.conversations (user_id, last_message_at desc);

create trigger conversations_updated_at
  before update on public.conversations
  for each row execute procedure public.set_updated_at();

create table if not exists public.messages (
  id              uuid primary key default gen_random_uuid(),
  conversation_id uuid not null references public.conversations(id) on delete cascade,
  role            text not null check (role in ('user','assistant','system','tool')),
  content         text not null,
  audio_url       text,
  transcript      text,
  metadata        jsonb,
  created_at      timestamptz not null default now()
);

create index if not exists messages_conversation_time_idx
  on public.messages (conversation_id, created_at);
