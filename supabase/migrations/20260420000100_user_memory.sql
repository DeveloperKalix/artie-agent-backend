-- User profile (experience level) + free-form memory notes appended by /skill.
-- user_id is text to mirror the convention used by conversations/snaptrade_users.

create table if not exists public.user_profiles (
  user_id          text primary key,
  experience_level text not null default 'novice'
                   check (experience_level in ('novice','intermediate','veteran')),
  onboarded_at     timestamptz,
  created_at       timestamptz not null default now(),
  updated_at       timestamptz not null default now()
);

create trigger user_profiles_updated_at
  before update on public.user_profiles
  for each row execute procedure public.set_updated_at();

create table if not exists public.user_memory (
  id         uuid primary key default gen_random_uuid(),
  user_id    text not null,
  content    text not null,
  source     text not null default 'skill'
             check (source in ('skill','system','inferred')),
  created_at timestamptz not null default now()
);

create index if not exists user_memory_user_idx
  on public.user_memory (user_id);
