"""Deterministic, source-preserving true-row paraphrases; never world evidence."""
import calendar
import collections
import csv
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent

# Natural names for metrics, deliberately preserving scope and billing basis.
NOUNS = dict(line.split('|', 1) for line in """
price|price
version|release version
free_trial|free trial period
monthly_active_users|monthly user base
languages|language support
launched|launch date
storage|storage allowance
date|event date
units_sold|sales total
store_rating|store rating
employees|employee headcount
median_response_time|median response time
included_seats|included seat allowance
uptime_sla|uptime guarantee
city|host city
paying_customers|paying customer base
context_window|context window
customers|customer total
active_users|active user total
regions|region count
headquarters|headquarters location
lead|lead
uptime_last_quarter|uptime last quarter
free_storage|free storage allowance
head|head
share_of_devices|share of devices
staff|staff headcount
max_upload_size|maximum upload size
data_retention|data retention period
attendees|attendance total
share_of_ad_revenue|share of Hooly's advertising revenue
refund_window|refund window
cloud_provider|cloud provider
integrations|integration count
released|release date
battery_capacity|battery capacity
availability_zones|availability zone count
joined|joining date
venue|venue
sponsors|sponsor count
stable_version|stable version
cto|chief technology officer
programming_languages|programming language support
ticket_price|ticket price
languages_served|language coverage
german_premium_phone_share|share of Germany's premium phone market
base_storage|base storage
hallucination_rate|hallucination rate
founded|founding date
annual_price|annual subscription price
history_retention|history retention period
daily_requests|daily request allowance
length|event length
official_sdks|official SDK count
stable_release_date|stable release date
latest_release_date|latest release date
index_size|search index size
images_indexed|indexed image total
publisher_revenue_share|publisher revenue share
executive_owner|executive owner
countries_served|country coverage
repair_centres|repair centre count
data_centres|data centre count
battery_life|battery life
team_size|team size
median_time_to_recover|median recovery time
cli_version|CLI version
beta_date|beta launch date
general_availability_date|general availability date
release_cadence|release interval
talks|talk count
manufacturing_partner|manufacturing partner
cash|cash balance
supported_ides|supported IDE count
supported_file_types|supported file type count
price_per_user|price per user
primary_region|primary hosting region
advertising_share|advertising share of revenue
signed_in_devices|signed-in device allowance
saved_projects|saved project allowance
reliability_lead|reliability lead
webhook_events|webhook event count
browser_market_share|browser market share
extensions_in_store|extension store total
news_refresh_interval|news refresh interval
beta_testers|beta tester count
weight|weight
max_trade_in_credit|maximum trade-in credit
update_promise|promised upgrade count
series_a_lead|Series A lead investor
offers|job offer total
standard_storage_price|standard storage price
free_egress_per_month|free monthly egress allowance
countries_covered|country coverage
places_listed|listed place total
supported_models|supported model count
storage_per_user|storage allowance per user
desks|desk count
rent|rent
bike_racks|bike rack count
operating_margin|operating margin
home_region|home region
attachment_limit|attachment size limit
support_first_response|support first response time
hosting_regions|hosting region count
webcast_viewers|webcast audience
call_length|call length
context_files|context file allowance
max_output_length|maximum output length
flawbench_score|FlawBench score
registrations|registration total
suggestion_acceptance_rate|suggestion acceptance rate
cold_start_time|cold start time
top_bug_bounty|top bug bounty
yearly_billing_discount|yearly billing discount
trash_retention|trash retention period
photos_stored|stored photo total
update_size|update download size
inner_screen_size|inner screen size
series_a_amount|Series A funding amount
independent_members|independent member count
applicants|applicant total
offer_acceptance|offer acceptance rate
referral_bonus|referral bonus
time_to_hire|time to hire
storage_tiers|storage tier count
mapped_distance|mapped distance
ad_free_price|ad-free subscription price
screen_size|screen size
charging_power|charging power
revenue|revenue
extra_seat_price|extra seat price
guidance|revenue guidance
ceo|chief executive officer
previous_employer|previous employer
opening_keynote|opening keynote speaker
services|service count
simultaneous_editors|simultaneous editor allowance
tab_sleep_delay|tab sleep delay
announced|announcement date
family_group_size|family group size
seats|board seat count
hires|hire total
referral_share|referral share of hires
interview_rounds|interview round count
first_reply|time to first recruiting reply
executive_sponsor|executive sponsor
respondents|respondent count
would_recommend|share of respondents who would recommend Hooly
manager_rating|manager rating
manageable_workload|share of respondents reporting a manageable workload
next_round|next survey round date
sessions|session count
free_credit|free credit
cool_storage_price|cool storage price
archive_storage_price|archive storage price
offline_area_limit|offline map storage limit per area
shipping_android|shipping Android version
active_devices|active device total
attendance|director attendance rate
open_roles|open role count
cost_per_hire|average cost per hire
desks_managed|managed desk count
newest_region|newest region
mailbox_storage|mailbox storage allowance
voice_latency|voice latency
next_meeting|next meeting date
revenue_growth|revenue growth
offices|office count
help_guides|help guide count
planned_general_availability|planned general availability date
minimum_nodes|minimum node count
concurrent_chats|concurrent chat allowance
templates|template count
head_of_workspace|head of Workspace
capacity|capacity
host|host
video_upload_limit|video upload duration limit
support_satisfaction|support satisfaction rate
stable_release|stable release date
cover_screen_size|cover screen size
minimum_seats|minimum seat count
chair|board chair
meetings_per_year|annual meeting count
response_rate|survey response rate
open_for|survey response window
speakers|speaker count
max_participants|maximum participant count
max_call_length|maximum call length
apps_included|included app count
sessions_held|completed board session count
recruiters|recruiter count
open_engineering_roles|open engineering role count
facilities_team|facilities team headcount
lease_end|lease end date
spam_catch_rate|spam catch rate
business_listings|business listing total
pro_price|Pro subscription price
text_limit|text length limit
languages_supported|supported language count
offline_languages|offline language count
rated_accuracy|rated accuracy
devices_keynote_speaker|devices keynote speaker
memory|memory capacity
requests_per_day|daily request volume
offices_run|managed office count
published|publication date
charging_stations_listed|listed charging station count
head_of_product|head of product
keynote_length|keynote length
soc2_auditor|SOC 2 auditor
head_of_security|head of security
audit_retention|audit retention period
livestream_viewers|livestream audience
api_version|API version
customer_countries|customer country count
duration|duration
keynote_speaker|keynote speaker
hinge_rating|hinge durability rating
run_by|survey organiser
footprint|carbon footprint
renewable_electricity|renewable electricity share
devices_recycled|recycled device count
cycling_commuters|share of commuters who cycle
travel_emissions_cut|travel emissions reduction
net_zero_date|net-zero target date
functions|function count
max_cells|maximum cell count
refresh_rate|screen refresh rate
main_camera|main camera resolution
stream_viewers|stream audience
colours|available colour count
requests_per_minute|request limit per minute
opened|opening date
voice_guidance_languages|voice guidance language count
dialects_supported|supported dialect count
growth_opportunities|share of respondents reporting growth opportunities
headcount|employee headcount
soc2_audit_start|SOC 2 audit start date
onboarding_time|onboarding time
maximum_seats|maximum seat count
workshops|workshop count
p99_response_time|p99 response time
waitlist|waitlist total
founders|founder count
incidents_this_year|incident count this year
users|user total
net_promoter_score|net promoter score
water_resistance_depth|water resistance depth
full_charge_time|full charge time
caption_languages|caption language count
emissions_cut_target|emissions reduction target
owner|report owner
features_launched|launched feature count
chair_since|start date of the chair's tenure
data_residency_regions|data residency region count
chairs|board chaired
flawless_seats|Flawless seat count
customers_in_the_eu|share of customers in the EU
training_data_cutoff|training data cutoff
webinars_held|completed webinar count
newsletter_subscribers|newsletter subscriber total
support_tickets_per_week|weekly support ticket total
max_file_size|maximum file size
max_guests|maximum guest count
version_history|version history retention period
""".strip().splitlines())

BOOLS = dict(line.split('|', 1) for line in """
end_to_end_encrypted_backups|encrypts backups end to end
region_outside_europe|has a region outside Europe
single_sign_on|supports single sign-on
voice_mode|offers voice mode
ai_answers|offers AI answers
open_source|is open source
carbon_neutral|is carbon neutral
offline_editing|supports offline editing
trade_in_programme|offers a trade-in programme
raviqa_seat|has a seat for Raviqa Capital
integration_partner_of_flawless|is an integration partner of Flawless
end_to_end_encryption|provides end-to-end encryption
personalised_ads_by_default|enables personalised ads by default
had_major_outage|has had a major outage
image_input|accepts image input
replay_available|has a replay available
satellite_messaging|supports satellite messaging
ai_sidebar|includes an AI sidebar
eu_hosted|is hosted in the EU
is_headquarters|is Hooly's headquarters
is_engineering_hub|is an engineering hub
audit_committee|has an audit committee
data_processing_agreement|offers a data processing agreement
generally_available|is generally available
soc2_certified|is SOC 2 certified
discontinued|is discontinued
raised_series_a|has raised Series A funding
externally_audited|has been externally audited
cloud_recording|offers cloud recording
passkey_sync|supports passkey synchronisation
ad_free_option|offers an ad-free option
memory_saver|includes memory saver
gemini_formula_assist|includes Gemini formula assistance
appointment_booking|offers appointment booking
encryption_at_rest|provides encryption at rest
camera_translation|supports camera translation
copilot_included|includes the copilot
on_device_gemini|supports on-device Gemini
ai_editing|supports AI editing
ar_walking|supports augmented-reality walking directions
livestreamed|is livestreamed
cellular_option|has a cellular option
adaptive_noise_cancelling|supports adaptive noise cancelling
chairs_audit_committee|chairs the audit committee
independent_director|is an independent director
pro_tier|offers a Pro tier
anonymous|keeps survey responses anonymous
fully_remote|is fully remote
raised_series_b|has raised Series B funding
desktop_mode|supports desktop mode
""".strip().splitlines())

UNITS = {'GB':'GB','MB':'MB','TB':'TB','KB':'KB','mAh':'mAh','W':'W','Hz':'Hz','g':'g','kg':'kg','m':'m','mm':'mm','km':'km','in':'in'}
PLURALS = {'person':'people','country':'countries','company':'companies','century':'centuries','inch':'inches','watch':'watches'}
DURATION = {'ms':'ms','s':'seconds','min':'minutes','h':'hours','day':'days','week':'weeks','month':'months','year':'years'}

# These source annotations require narrative inference or disagree with the text.
# Keeping them out is safer than turning incidental mentions into new assertions.
EXCLUDED = {'p-36','p-46','p-99','p-289','p-299','p-313','p-867','p-874',
            'p-954','p-957','p-1064','p-1158','p-1175','p-1192','p-1207',
            'p-1277','p-1372','p-1384','p-1430','p-1487','p-1582','p-1587',
            'p-1546','p-1595','p-1621','p-1660','p-1681','p-1837',
            'p-119','p-174','p-204','p-338','p-516','p-1020','p-1087',
            'p-1142','p-1156','p-1290','p-1558','p-1585','p-1592','p-1607',
            'p-1727','p-1733'}

def number(n):
    if isinstance(n, int) or n == int(n):
        return f'{int(n):,}'
    return str(n)

def value(v):
    k, n = v['kind'], v.get('value')
    if k == 'date':
        parts = n.split('-')
        result = n if len(parts) == 1 else f'{calendar.month_name[int(parts[1])]} {parts[0]}'
        if len(parts) == 3:
            result = f'{int(parts[2])} {result}'
    elif k == 'version':
        result = f'v{n}'
    elif k == 'name':
        result = n
    elif k == 'money':
        symbol = {'EUR':'€','USD':'$','GBP':'£'}.get(v['currency'], v['currency']+' ')
        result = symbol + number(n)
        if 'per' in v:
            result += ' per ' + v['per']
    elif k == 'percent':
        result = number(n) + '%'
    elif k == 'duration':
        unit = DURATION[v['unit']]
        if n == 1 and unit not in {'ms'}:
            unit = unit.rstrip('s')
        result = f'{number(n)} {unit}'
    elif k == 'quantity':
        raw = v['unit']
        unit = UNITS.get(raw, raw if n == 1 else PLURALS.get(raw, raw+'s'))
        if raw in {'sdk','ide'}:
            unit = raw.upper() + ('' if n == 1 else 's')
        result = f'{number(n)} {unit}'
    else:
        raise ValueError(k)
    op = v.get('op', 'eq')
    if op != 'eq':
        prefixes = {'gt':'more than ','lt':'less than ','ge':'at least ','le':'at most '}
        if k == 'date':
            prefixes = {'gt':'after ','lt':'before ','ge':'no earlier than ','le':'no later than '}
        result = prefixes[op]+result
    return result

def boolean_clause(s, a, positive):
    predicate = BOOLS[a]
    if positive:
        return f'{s} {predicate}'
    if predicate.startswith('is '):
        predicate = 'is not ' + predicate[3:]
    elif predicate.startswith('has been '):
        predicate = 'has not been ' + predicate[9:]
    elif predicate.startswith('has '):
        predicate = 'has not ' + predicate[4:] if a in {'had_major_outage','raised_series_a','raised_series_b'} else 'does not have ' + predicate[4:]
    else:
        first, rest = predicate.split(' ', 1)
        base = {'provides':'provide','offers':'offer','supports':'support','accepts':'accept','includes':'include','keeps':'keep','chairs':'chair','enables':'enable','encrypts':'encrypt'}.get(first)
        if base is None:
            raise ValueError(predicate)
        predicate = 'does not ' + base + ' ' + rest
    return f'{s} {predicate}'

def render(form, style, source):
    clauses = []
    for index, f in enumerate(form):
        s, a, v = f['subject'], f['attribute'], f['value']
        src = source.casefold()
        if v['kind'] == 'boolean':
            clause = boolean_clause(s, a, v['value'])
            if v['value'] and a == 'memory_saver' and 'enabled' in src:
                clause = f'{s} has memory saver enabled'
            if v['value'] and a in {'ar_walking','ai_sidebar','memory_saver'} and 'by default' in src:
                features = {'ar_walking':'augmented-reality walking directions','ai_sidebar':'the AI sidebar','memory_saver':'memory saver'}
                clause = f"{s} enables {features[a]} by default"
            if a == 'satellite_messaging' and 'beta' in src:
                clause += ' in the beta'
            if a == 'single_sign_on' and 'for admins' in src:
                clause += ' for admins'
            clauses.append(clause)
            continue
        noun = NOUNS[a]
        val = value(v)
        # Retain qualifications present in the prose but absent from typed values.
        if a in {'standard_storage_price','cool_storage_price','archive_storage_price'} and ('terabyte' in src or 'tb' in src):
            val += ' per terabyte'
        if a == 'price_per_user' and s == 'Hooly Drive' and 'add-on' in src:
            noun = 'storage add-on price per user'
        if a == 'price' and re.search(r'per (user|person|head)',src):
            val += ' per user'
        if a == 'price' and 'recommended price' in src:
            noun = 'recommended price'
        if a == 'max_trade_in_credit' and 'when you order a hooly pixel fold' in src:
            val += ' when buying a Hooly Pixel Fold'
        if a == 'requests_per_minute' and 'per key' in src:
            noun += ' per key'
        if a == 'tab_sleep_delay' and ('memory saver' in src or 'saves memory' in src):
            noun += ' when memory saver is active'
        if a == 'time_to_hire':
            if 'average' in src:
                noun = 'average time to hire'
            if 'median' in src:
                noun = 'median time to hire'
            if 'from application to signature' in src:
                noun += ' from application to signature'
        if a == 'weight' and 'without the strap' in src:
            noun += ' without the strap'
        if a == 'battery_life':
            if 'typical' in src:
                noun = 'typical battery life'
            if 'always on' in src or 'always-on' in src:
                val += ' with the display set to always on'
        if a == 'stable_release' and any(word in src for word in ('planned','due','scheduled')):
            noun = 'planned stable release date'
        if a == 'date' and any(word in src for word in ('planned','set for','scheduled')):
            noun = 'scheduled event date'
        if a == 'active_users' and 'month' in src:
            noun = 'monthly active user total'
        if a == 'paying_customers' and 'paying' not in src:
            noun = 'customer base'
        if a == 'suggestion_acceptance_rate' and 'across all accounts in august' in src:
            val += ' as measured across all accounts in August'
        if a == 'text_limit' and 'pro' in src:
            noun = 'Pro tier text length limit'
        if a == 'max_cells' and ('file' in src or 'sheet' in src):
            noun += ' per file'
            if 'counting empty cells' in src:
                val += ', including empty cells in the used range'
        if a == 'concurrent_chats' and 'per user' in src:
            noun += ' per user'
        if a == 'saved_projects' and 'per account' in src:
            noun += ' per account'
        if a == 'mailbox_storage' and ('per user' in src or 'every inbox' in src):
            noun += ' per user'
        if a == 'max_participants' and 'per room' in src:
            noun += ' per room'
        if a == 'text_limit' and 'each request' in src:
            noun += ' per request'
        if a == 'livestream_viewers' and s == 'Hooly Meet':
            noun = 'livestream viewer capacity'
        if a == 'revenue_growth' and 'quarter on quarter' in src:
            noun = 'quarter-on-quarter revenue growth'
        if a == 'headcount' and 'quarter end' in src:
            noun = 'quarter-end employee headcount'
        if a == 'guidance' and "next quarter" in src:
            noun = 'next-quarter revenue guidance'
        if a == 'joined':
            prep = 'on' if len(v['value']) == 10 else 'in'
            clauses.append(f'{s} joined Hooly {prep} {val}')
        elif a == 'languages':
            clauses.append(f'{s} supports {val}')
        elif a in {'programming_languages','languages_served','languages_supported'}:
            clauses.append(f'{s} has {noun} of {val}')
        elif a == 'opening_keynote':
            clauses.append(f'{val} gives the opening keynote at {s}')
        elif a == 'run_by':
            clauses.append(f'{val} runs {s}')
        elif a == 'chairs':
            clauses.append(f'{s} chairs {val}')
        elif a == 'context_files' and 'request' in src:
            clauses.append(f'{s} can reference {val} in one request')
        elif style % 4 == 0:
            clauses.append(f"{s}'s {noun} is {val}")
        elif style % 4 == 1:
            clauses.append(f'The {noun} for {s} is {val}')
        elif style % 4 == 2:
            if v['kind'] == 'name':
                clauses.append(f'The {noun} of {s} is {val}')
            else:
                article = 'an' if noun[0].lower() in 'aeiou' else 'a'
                clauses.append(f'{s} has {article} {noun} of {val}')
        else:
            clauses.append(f'{noun[0].upper()+noun[1:]} for {s}: {val}')
    return '; '.join(clauses) + '.'

def main():
    rows = list(csv.DictReader((ROOT/'data/practice.csv').open()))
    normalize = lambda text: re.sub(r'[\W_]+', ' ', text.casefold()).strip()
    old_texts = {normalize(r['text']) for r in rows}
    pool = []
    for row in rows:
        if row['label'] != 'true' or row['id'] in EXCLUDED:
            continue
        form = json.loads(row['form'])
        try:
            # Vary the grammatical construction rather than adding superficial wrappers.
            style = int(hashlib.sha256(row['id'].encode()).hexdigest()[:8],16)
            for offset in range(4):
                text = render(form, style + offset, row['text'])
                normalized = normalize(text)
                if len(text) <= 300 and normalized not in old_texts:
                    break
            else:
                continue
        except (KeyError, ValueError):
            continue
        pool.append({'source_id':row['id'],'text':text,'date':row['date'],'label':'true','form':form,'transformation':'typed_assertions_rephrased_without_value_or_date_changes'})
    # Cover low-frequency subject/attribute pairs before filling from corpus order.
    coverage = collections.Counter((f['subject'],f['attribute']) for r in pool for f in r['form'])
    pool.sort(key=lambda r:(min(coverage[(f['subject'],f['attribute'])] for f in r['form']), int(r['source_id'][2:])))
    selected, used = [], set(old_texts)
    for row in pool:
        key = normalize(row['text'])
        if key in used:
            continue
        selected.append(row)
        used.add(key)
        if len(selected) == 900:
            break
    assert len(selected) == 900, len(selected)
    assert len({r['source_id'] for r in selected}) == 900
    by_id = {r['id']:r for r in rows}
    for r in selected:
        source = by_id[r['source_id']]
        assert source['date'] == r['date'] and source['label'] == r['label']
        assert json.loads(source['form']) == r['form']
    (HERE/'candidates.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected))
    print(json.dumps({'rows':len(selected),'source_rows':len({r['source_id'] for r in selected}),'subjects':len({f['subject'] for r in selected for f in r['form']}),'attributes':len({f['attribute'] for r in selected for f in r['form']}),'candidate_pool':len(pool)}))

if __name__ == '__main__':
    main()
