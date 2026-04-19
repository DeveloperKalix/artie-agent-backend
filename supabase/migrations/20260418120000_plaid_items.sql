-- Plaid Item storage (run via Supabase SQL Editor or `supabase db push`).
-- Fixes PostgREST PGRST205: Could not find the table 'public.plaid_items' in the schema cache

create table if not exists public.plaid_items (
  id uuid primary key default gen_random_uuid(),
  user_id text not null,
  item_id text not null unique,
  access_token text not null,
  institution_id text,
  institution_name text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists plaid_items_user_id_idx on public.plaid_items (user_id);

alter table public.plaid_items enable row level security;

-- service_role key bypasses RLS. For anon/authenticated clients, add policies as needed.

notify pgrst, 'reload schema';
