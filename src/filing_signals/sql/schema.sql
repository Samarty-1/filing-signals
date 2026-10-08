-- Raw-ish tables: one per source, loaded idempotently (re-runs upsert).

create table if not exists master_index (
    accession       varchar primary key,
    cik             bigint not null,
    company_name    varchar,
    form            varchar not null,
    date_filed      date not null,
    index_quarter   varchar not null            -- e.g. 2015Q1
);

create table if not exists filers (
    cik                     bigint primary key,
    name                    varchar,
    entity_type             varchar,
    sic                     integer,
    sic_description         varchar,
    state_of_incorporation  varchar,
    fiscal_year_end         varchar,
    current_tickers         varchar,            -- today's tickers: not history
    fetched_at              timestamptz not null
);

create table if not exists filer_former_names (
    cik         bigint not null,
    name        varchar not null,
    valid_from  date,
    valid_to    date,
    primary key (cik, name, valid_from)
);

-- The sample: every candidate drawn, in draw order, with the decision.
create table if not exists universe (
    cik         bigint primary key,
    draw_rank   integer not null,
    seed        bigint not null,
    status      varchar not null,               -- included | excluded
    reason      varchar,
    decided_at  timestamptz not null
);

create table if not exists filings (
    accession           varchar primary key,
    cik                 bigint not null,
    form                varchar not null,
    filing_date         date,
    report_date         date,
    accepted_api_raw    varchar,                -- submissions API value, unreliable
    primary_document    varchar,
    size_bytes          bigint,
    items               varchar,
    first_seen_at       timestamptz not null
);

create table if not exists filing_headers (
    accession       varchar primary key,
    accepted_at     timestamptz,                -- authoritative: from the SGML header
    header_period   date,
    fetched_at      timestamptz not null
);

create table if not exists documents (
    accession       varchar primary key,
    url             varchar not null,
    stored_path     varchar not null,
    bytes_raw       bigint not null,
    bytes_stored    bigint not null,
    sha256          varchar not null,
    fetched_at      timestamptz not null
);

create table if not exists fetch_failures (
    url         varchar primary key,
    accession   varchar,
    error       varchar,
    failed_at   timestamptz not null
);

create table if not exists ingest_runs (
    run_id          varchar primary key,
    started_at      timestamptz not null,
    finished_at     timestamptz,
    params          json,
    requests        bigint,
    retries         bigint,
    bytes           bigint,
    status          varchar
);
