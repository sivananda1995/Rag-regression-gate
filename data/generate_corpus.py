"""Generate the synthetic support-knowledge-base corpus and its labeled golden set.

Why synthetic: a retrieval gate has to be demonstrated on a corpus whose relevance
labels are complete and correct. Public IR datasets with trustworthy labels are
either licence-encumbered or too large to commit, and hand-labeling a real corpus
inside one build is not credible. So the corpus is generated from a seeded template
grammar, and the labels come from the generator, which means they are exact.

The generator is deliberately adversarial, because a corpus that any retriever
scores 1.00 on cannot demonstrate a regression:
  * every document has near-duplicate siblings that differ only in platform or
    product, so lexical overlap alone does not identify the right one,
  * queries are written in user voice and pass through a synonym map, so the
    query rarely repeats the document's title wording,
  * a share of queries carry realistic typos.

Run: python data/generate_corpus.py --docs 420 --queries 140 --seed 20260812
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

PRODUCTS = [
    ("Atlas VPN Client", "atlas-vpn"),
    ("Beacon SSO Portal", "beacon-sso"),
    ("Cobalt Mail", "cobalt-mail"),
    ("Delta Drive", "delta-drive"),
    ("Echo Meetings", "echo-meetings"),
    ("Forge CI Runner", "forge-ci"),
    ("Granite Vault", "granite-vault"),
    ("Harbor Print Service", "harbor-print"),
    ("Ionis Helpdesk", "ionis-helpdesk"),
    ("Juno Time Tracker", "juno-time"),
    ("Kepler Analytics", "kepler-analytics"),
    ("Lumen Badge Reader", "lumen-badge"),
]

PLATFORMS = ["Windows 11", "macOS 15", "iOS 18", "Android 15", "Ubuntu 24.04", "ChromeOS"]

TASKS = [
    ("reset multi-factor authentication", "mfa", "enrollment", "MFA-4021",
     ["authenticator", "one-time passcode", "recovery codes"]),
    ("recover a locked account", "lockout", "identity", "IDP-1180",
     ["failed sign-in attempts", "unlock window", "security questions"]),
    ("restore a deleted shared folder", "restore", "storage", "STG-7742",
     ["retention window", "version history", "trash bin"]),
    ("fix certificate validation failures", "cert", "network", "TLS-5090",
     ["root certificate", "trust store", "expired chain"]),
    ("increase the upload size limit", "quota", "storage", "STG-3310",
     ["quota policy", "chunked upload", "administrator override"]),
    ("migrate settings to a replacement device", "migrate", "endpoint", "EP-2255",
     ["device pairing", "profile export", "handover token"]),
    ("stop duplicate calendar invitations", "duplicates", "messaging", "MSG-6641",
     ["sync conflict", "delegate access", "invitation loop"]),
    ("enable single sign-on for a new tenant", "sso", "identity", "IDP-9004",
     ["metadata exchange", "assertion signing", "claim mapping"]),
    ("clear a stuck print queue", "printqueue", "endpoint", "EP-4402",
     ["spooler service", "driver mismatch", "held job"]),
    ("resolve slow synchronisation", "slowsync", "storage", "STG-8815",
     ["delta sync", "bandwidth throttle", "large file scan"]),
    ("rotate an expiring service credential", "rotate", "security", "SEC-3121",
     ["secret version", "grace period", "dual-write window"]),
    ("recover a failed pipeline agent", "agent", "delivery", "CI-7003",
     ["runner heartbeat", "workspace cleanup", "queue backlog"]),
    ("correct a wrong time-zone on reports", "timezone", "analytics", "AN-5520",
     ["locale profile", "scheduled export", "daylight saving shift"]),
    ("re-provision a badge that stopped opening doors", "badge", "facilities", "FAC-1902",
     ["reader firmware", "credential template", "access group"]),
]

BODY = (
    "This article applies to {product} on {platform}. Users report {symptom} and the "
    "service log records error {code}. Confirm the {hint_a} first, because the most "
    "common cause is a stale {hint_b} left behind by the previous configuration. "
    "Open the administration console, select the affected account, and review the "
    "{hint_c} entry for the last twenty-four hours. If the entry is missing, the "
    "request never reached the {domain} service and the fault is upstream. "
    "To apply the fix, place the account in maintenance state, remove the stale "
    "record, then ask the user to sign in again from {platform} so a fresh record is "
    "written. Verify success by checking that error {code} no longer appears and that "
    "the {hint_a} shows the current timestamp. Escalate to the {domain} on-call rota "
    "if the error survives two clean attempts, and attach the console export."
)

SYMPTOMS = {
    "mfa": "the authenticator app never receives a challenge after a device change",
    "lockout": "sign-in is refused with no lockout notice and no self-service unlock",
    "restore": "a folder shared with a team vanished and version history looks empty",
    "cert": "the client refuses the connection and reports an untrusted issuer",
    "quota": "large uploads abort part way through with no error shown to the user",
    "migrate": "settings do not follow the user to a replacement device",
    "duplicates": "the same invitation arrives several times for one meeting",
    "sso": "the federated login loop returns to the sign-in page",
    "printqueue": "jobs stay queued and the queue cannot be cleared from the client",
    "slowsync": "synchronisation crawls for hours on files that changed by a few bytes",
    "rotate": "an integration begins failing authentication shortly after a rotation",
    "agent": "the build agent shows online but never claims a queued job",
    "timezone": "scheduled reports arrive stamped with the wrong local hour",
    "badge": "a badge reads at the turnstile but the door stays locked",
}

# Query paraphrases written in user voice. They deliberately avoid the title wording
# so that lexical retrieval has to work through overlap in the body and entities.
PARAPHRASES = {
    "mfa": [
        "new phone and my {product} codes never arrive, how do I start enrollment over",
        "{code} keeps showing when I try to sign in to {product} from {platform}",
        "lost my authenticator, need the six digit prompt back on {platform}",
    ],
    "lockout": [
        "{product} will not let me in and there is no unlock button on {platform}",
        "too many bad passwords on {product}, how long until I can try again",
        "account refused with {code} and self service unlock is missing",
    ],
    "restore": [
        "team folder disappeared from {product}, can we get it back",
        "shared directory gone and no earlier versions visible on {platform}",
        "{code} when opening a folder someone deleted last week",
    ],
    "cert": [
        "{product} says the issuer is not trusted on {platform}",
        "connection refused with {code}, looks like a trust problem",
        "untrusted certificate warning after the weekend on {platform}",
    ],
    "quota": [
        "big files stop part way up in {product} with no message",
        "{code} on a large attachment, is there a size cap on {platform}",
        "need a bigger upload allowance for {product}",
    ],
    "migrate": [
        "swapped laptops and none of my {product} preferences came across",
        "{code} while moving my profile to a replacement {platform} machine",
        "how do I carry settings to a new device for {product}",
    ],
    "duplicates": [
        "getting the same meeting invite four times from {product}",
        "{code} in the log and invitations keep repeating on {platform}",
        "calendar keeps duplicating one booking",
    ],
    "sso": [
        "federated sign in bounces back to the login screen for {product}",
        "{code} setting up a brand new tenant for single sign on",
        "login loop on {platform} after we turned on federation",
    ],
    "printqueue": [
        "print jobs stuck and I cannot clear them from {platform}",
        "{code} and the queue will not empty on {product}",
        "documents sit waiting forever at the printer",
    ],
    "slowsync": [
        "{product} takes hours to sync a tiny change on {platform}",
        "{code} appears while sync crawls through big files",
        "why is my sync so slow after a small edit",
    ],
    "rotate": [
        "integration broke right after we changed the secret for {product}",
        "{code} once the old credential expired on {platform}",
        "how do we swap an expiring key without downtime",
    ],
    "agent": [
        "build machine looks connected but never picks up work in {product}",
        "{code} and the runner sits idle with jobs waiting",
        "pipeline agent stopped taking jobs on {platform}",
    ],
    "timezone": [
        "scheduled report from {product} arrives with the wrong hour",
        "{code} and every export is off by an hour on {platform}",
        "reports show the wrong local time after the clocks changed",
    ],
    "badge": [
        "badge beeps at the door but it stays shut, {product} shows it active",
        "{code} on the reader and the credential looks fine",
        "door will not open even though the badge scans on {platform}",
    ],
}

TYPO_MAP = str.maketrans({"e": "e", "i": "i"})


def _inject_typo(text: str, rng: random.Random) -> str:
    words = text.split()
    if len(words) < 4:
        return text
    idx = rng.randrange(len(words))
    word = words[idx]
    if len(word) < 5:
        return text
    cut = rng.randrange(1, len(word) - 2)
    words[idx] = word[:cut] + word[cut + 1 :]
    return " ".join(words)


def build(docs_target: int, queries_target: int, seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    documents: list[dict] = []
    # Every (task, product, platform) triple is a plausible article. Sampling without
    # replacement across the full grid guarantees near-duplicate siblings exist.
    grid = [
        (task, product, platform)
        for task in TASKS
        for product in PRODUCTS
        for platform in PLATFORMS
    ]
    rng.shuffle(grid)
    chosen = grid[:docs_target]
    for i, (task, product, platform) in enumerate(chosen):
        title_action, key, domain, code, hints = task
        product_name, product_slug = product
        doc_id = f"kb-{i:04d}"
        documents.append(
            {
                "doc_id": doc_id,
                "title": f"How to {title_action} for {product_name} on {platform}",
                "text": BODY.format(
                    product=product_name,
                    platform=platform,
                    symptom=SYMPTOMS[key],
                    code=code,
                    hint_a=hints[0],
                    hint_b=hints[1],
                    hint_c=hints[2],
                    domain=domain,
                ),
                "task_key": key,
                "product": product_slug,
                "platform": platform,
                "error_code": code,
            }
        )

    queries: list[dict] = []
    pool = list(documents)
    rng.shuffle(pool)
    for i, doc in enumerate(pool[:queries_target]):
        key = doc["task_key"]
        template = rng.choice(PARAPHRASES[key])
        product_name = next(name for name, slug in PRODUCTS if slug == doc["product"])
        text = template.format(
            product=product_name, platform=doc["platform"], code=doc["error_code"]
        )
        # A query must name the product, otherwise dozens of articles about the same
        # task answer it equally well and the label set becomes arbitrary.
        if "{product}" not in template:
            text = f"{product_name}: {text}"

        # Relevance is derived from what the query actually constrains. A query that
        # names a platform has exactly one correct article; a query that names only
        # the product is answered by that product's article on any platform, and all
        # of them are labeled. Labeling only one would penalise a retriever for
        # returning a genuinely correct document.
        names_platform = "{platform}" in template
        relevant = [
            d["doc_id"]
            for d in documents
            if d["task_key"] == key
            and d["product"] == doc["product"]
            and (not names_platform or d["platform"] == doc["platform"])
        ]
        if rng.random() < 0.25:
            text = _inject_typo(text, rng)
        queries.append(
            {
                "query_id": f"q-{i:04d}",
                "text": text,
                "relevant_doc_ids": sorted(relevant),
                "task_key": key,
                "scope": "platform" if names_platform else "product",
            }
        )
    return documents, queries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", type=int, default=420)
    parser.add_argument("--queries", type=int, default=140)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--out-dir", default="data")
    args = parser.parse_args()

    documents, queries = build(args.docs, args.queries, args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "corpus.jsonl").open("w") as fh:
        for doc in documents:
            fh.write(json.dumps(doc) + "\n")
    with (out / "golden_queries.jsonl").open("w") as fh:
        for query in queries:
            fh.write(json.dumps(query) + "\n")
    print(f"wrote {len(documents)} documents and {len(queries)} queries to {out}/ (seed {args.seed})")


if __name__ == "__main__":
    main()
