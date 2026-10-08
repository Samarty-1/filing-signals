-- Modelled views over the raw tables. Recreated on every connect.

-- One row per annual report version. An original 10-K and each later 10-K/A
-- for the same fiscal period are versions of one report; version_no orders
-- them by when they became public.
create or replace view annual_reports as
with annual as (
    select
        f.accession,
        f.cik,
        f.form,
        f.form like '%/A'                                       as is_amendment,
        -- the header's period of report is authoritative; the API's
        -- reportDate is blank on some older filings
        coalesce(h.header_period, f.report_date)                as fiscal_period_end,
        f.filing_date,
        h.accepted_at,
        timezone('America/New_York', h.accepted_at)             as accepted_at_et,
        f.primary_document,
        d.stored_path,
        d.bytes_raw
    from filings f
    left join filing_headers h using (accession)
    left join documents d using (accession)
    where f.form in ('10-K', '10-K405', '10-KT', '10-KSB',
                     '10-K/A', '10-K405/A', '10-KT/A', '10-KSB/A')
)
select
    *,
    row_number() over (
        partition by cik, fiscal_period_end order by accepted_at, accession
    )                                                           as version_no,
    first_value(accession) over (
        partition by cik, fiscal_period_end order by accepted_at, accession
    )                                                           as original_accession,
    -- EDGAR dates anything accepted after 17:30 ET on the next business day,
    -- so filing_date is not when the text became public
    accepted_at_et::date < filing_date                          as dated_next_business_day
from annual;

-- What an investor could have read at time `as_of`: per company and fiscal
-- period, the latest version already public. Use this, never the raw table,
-- in anything that joins text to returns.
create or replace macro annual_reports_as_of(as_of) as table
    select *
    from annual_reports
    where accepted_at <= as_of
    qualify row_number() over (
        partition by cik, fiscal_period_end order by accepted_at desc, accession desc
    ) = 1;

-- How far the submissions API's acceptanceDateTime ("...Z") sits from the
-- header's acceptance time, per filing. Correct UTC would give 0.
create or replace view api_timestamp_check as
select
    f.accession,
    f.cik,
    f.form,
    h.accepted_at,
    f.accepted_api_raw,
    round(date_diff('second', h.accepted_at,
                    strptime(f.accepted_api_raw, '%Y-%m-%dT%H:%M:%S.%gZ')::timestamp
                        at time zone 'UTC') / 3600.0, 2)        as api_minus_header_hours
from filings f
join filing_headers h using (accession)
where f.accepted_api_raw is not null and h.accepted_at is not null;
