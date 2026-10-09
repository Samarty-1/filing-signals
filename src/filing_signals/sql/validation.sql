-- name: event_auc
-- Do the change measures move when the business changes? Pairs spanning an
-- 8-K Item 2.01 (completed acquisition or disposition) vs pairs that don't.
-- AUC = P(a random event pair is LESS similar than a random no-event pair);
-- 0.5 means no signal. A sanity check on the measures, not a causal claim:
-- firms that do deals differ in other ways too.
with ev as (
    select
        sc.*,
        exists (
            select 1
            from filings f
            join annual_reports p on p.accession = sc.prior_accession
            where f.cik = sc.cik and f.form = '8-K' and f.items like '%2.01%'
              and f.filing_date > p.accepted_at::date
              and f.filing_date < sc.accepted_at::date
        ) as had_deal
    from section_changes sc
),
long as (
    select section, had_deal, 'cosine' as measure, sim_cosine as sim from ev
    union all select section, had_deal, 'jaccard', sim_jaccard from ev
    union all select section, had_deal, 'tfidf', sim_tfidf from ev
),
ranked as (
    -- average rank for ties (15% of risk-factor pairs are verbatim copies,
    -- all tied at similarity 1); rank() alone would bias the AUC
    select *,
           rank() over (partition by section, measure order by sim desc)
             + (count(*) over (partition by section, measure, sim) - 1) / 2.0 as r
    from long
    where sim is not null and not isnan(sim)
)
select
    section,
    measure,
    count(*) filter (where had_deal)                                    as deal_pairs,
    count(*) filter (where not had_deal)                                as other_pairs,
    -- Mann-Whitney U from ranks (descending similarity), as an AUC
    round((sum(r) filter (where had_deal)
           - count(*) filter (where had_deal) * (count(*) filter (where had_deal) + 1) / 2.0)
          / (count(*) filter (where had_deal) * count(*) filter (where not had_deal)), 3)
                                                                        as auc
from ranked
group by section, measure
order by section, auc desc;
