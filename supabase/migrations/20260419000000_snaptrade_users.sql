-- Maps your app's user_id to a SnapTrade user_id + user_secret.
-- The user_secret is sensitive — never expose it to the frontend.
create table if not exists public.snaptrade_users (
  id            uuid primary key default gen_random_uuid(),
  app_user_id   text not null unique,       -- your Supabase auth user id
  snaptrade_user_id text not null unique,   -- UUID we generated for SnapTrade
  user_secret   text not null,             -- SnapTrade-issued per-user secret
  created_at    timestamptz not null default now(),
  updated_at    timestamptz not null default now()
);

-- Keep updated_at current automatically.
create or replace function public.set_updated_at()
returns trigger language plpgsql as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

create trigger snaptrade_users_updated_at
  before update on public.snaptrade_users
  for each row execute procedure public.set_updated_at();
