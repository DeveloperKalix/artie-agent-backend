-- News items ingested by the NewsAggregatorAgent, plus a per-user cursor
-- for tracking which news has already been surfaced to the recommendation agent.

create table if not exists public.news_items (
  id           uuid primary key default gen_random_uuid(),
  url          text not null unique,
  title        text not null,
  summary      text,
  source       text not null,
  tickers      text[] not null default '{}',
  published_at timestamptz,
  sentiment    text check (sentiment in ('positive','negative','neutral')),
  ingested_at  timestamptz not null default now()
);

create index if not exists news_items_published_idx
  on public.news_items (published_at desc);

create index if not exists news_items_tickers_idx
  on public.news_items using gin (tickers);

create table if not exists public.user_news_cursor (
  user_id      text primary key,
  last_seen_at timestamptz not null default now()
);
