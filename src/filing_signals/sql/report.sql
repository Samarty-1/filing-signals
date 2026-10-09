-- name: sample
-- How the universe was drawn: every candidate decision, by reason.
select status, coalesce(reason, '') as reason, count(*) as filers
from universe
group by all
order by filers desc;

-- name: coverage
-- Annual reports in the window, by filing year: fetched text and failures.
select
    year(filing_date)                                           as filing_year,
    count(*)                                                    as reports,
    count(*) filter (where is_amendment)                        as amendments,
    count(stored_path)                                          as with_text,
    count(distinct cik)                                         as firms
from annual_reports
where accepted_at is not null
group by all
order by filing_year;

-- name: timing
-- When annual reports become public, and what filing_date gets wrong.
select
    count(*)                                                    as reports,
    round(100.0 * avg(dated_next_business_day::int), 1)         as pct_dated_next_business_day,
    round(100.0 * avg((hour(accepted_at_et) >= 16)::int), 1)    as pct_after_close,
    round(100.0 * avg((hour(accepted_at_et) * 60 + minute(accepted_at_et) < 570)::int), 1)
                                                                as pct_before_open
from annual_reports
where accepted_at is not null;

-- name: api_offsets
-- The submissions API's "Z" timestamp minus the header's acceptance time.
-- 0 means the API is honest UTC.
select api_minus_header_hours as hours_off, count(*) as filings,
       round(100.0 * count(*) / sum(count(*)) over (), 1) as pct
from api_timestamp_check
group by all
order by filings desc;

-- name: session_flips
-- Filings whose trading session (pre-open / regular / after-close, on which
-- date) would be misclassified by trusting the API timestamp.
with s as (
    select
        accession,
        accepted_at                                                     as header_utc,
        strptime(accepted_api_raw, '%Y-%m-%dT%H:%M:%S.%gZ')::timestamp at time zone 'UTC'
                                                                        as api_utc
    from api_timestamp_check
),
sessions as (
    select
        accession,
        timezone('America/New_York', header_utc)   as h,
        timezone('America/New_York', api_utc)      as a
    from s
),
labelled as (
    select
        accession,
        h::date || ' ' || case when h::time < time '09:30' then 'pre-open'
                               when h::time < time '16:00' then 'regular'
                               else 'after-close' end      as header_session,
        a::date || ' ' || case when a::time < time '09:30' then 'pre-open'
                               when a::time < time '16:00' then 'regular'
                               else 'after-close' end      as api_session
    from sessions
)
select
    count(*)                                                    as filings,
    count(*) filter (where header_session <> api_session)       as session_flips,
    round(100.0 * avg((header_session <> api_session)::int), 1) as pct_flipped
from labelled;

-- name: amendments
-- How often a fiscal year's annual report is later amended, and how late.
with periods as (
    select cik, fiscal_period_end,
           count(*) - 1                                         as n_amendments,
           date_diff('day', min(accepted_at), max(accepted_at)) as days_to_last_amendment
    from annual_reports
    where accepted_at is not null
    group by all
)
select
    count(*)                                                    as fiscal_periods,
    round(100.0 * avg((n_amendments > 0)::int), 1)              as pct_amended,
    max(n_amendments)                                           as max_amendments,
    median(days_to_last_amendment) filter (where n_amendments > 0) as median_days_to_amendment
from periods;

-- name: period_disagreement
-- API reportDate vs the header's CONFORMED PERIOD OF REPORT.
select
    count(*)                                                            as reports,
    count(*) filter (where f.report_date is null)                       as api_period_missing,
    count(*) filter (where f.report_date <> h.header_period)            as disagree
from filings f
join filing_headers h using (accession);

-- name: storage
select
    count(*)                                    as documents,
    round(sum(bytes_raw) / 1e9, 2)              as raw_gb,
    round(sum(bytes_stored) / 1e9, 2)           as stored_gb,
    round(sum(bytes_raw) / sum(bytes_stored), 1) as compression
from documents;

-- name: sections
-- Section extraction outcome on original 10-Ks (amendments are mostly Part III
-- only and legitimately contain neither section).
select
    s.section,
    s.status,
    count(*)                                                            as reports,
    round(100.0 * count(*) / sum(count(*)) over (partition by s.section), 1) as pct
from sections s
join annual_reports a using (accession)
where not a.is_amendment
group by s.section, s.status
order by s.section, reports desc;

-- name: section_changes
select
    section,
    count(*)                                    as pairs,
    count(distinct cik)                         as firms,
    round(median(sim_cosine), 3)                as median_cosine,
    round(median(sim_jaccard), 3)               as median_jaccard,
    round(median(sim_tfidf), 3)                 as median_tfidf,
    round(100.0 * avg((sim_cosine > 0.9995)::int), 1) as pct_near_verbatim
from section_changes
group by section
order by section;
