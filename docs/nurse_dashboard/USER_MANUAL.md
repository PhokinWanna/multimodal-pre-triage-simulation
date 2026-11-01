# Nurse Dashboard — User Manual

The nurse dashboard is the staff side of the multimodal pre-triage kiosk. The kiosk talks with the walk-in patient, watches for danger signs and suggests a triage level; the dashboard shows that case to the nurse, who completes what the kiosk cannot collect and makes the final decision.

> **Research prototype.** The kiosk's level is a *suggestion*. The final triage decision is always made by a nurse. The system has not been clinically validated.

**Contents**

- [Part A — For nurses](#part-a--for-nurses)
  - [1. Sign in](#1-sign-in) · [2. The waiting queue](#2-the-waiting-queue) · [3. Reading a case](#3-reading-a-case) · [4. Completing the patient's details](#4-completing-the-patients-details) · [5. Making the decision](#5-making-the-decision) · [6. History and statistics](#6-history-and-statistics) · [7. Common situations](#7-common-situations)
- [Part B — For admins](#part-b--for-admins)
- [Part C — Installation and settings (technical)](#part-c--installation-and-settings-technical)
- [Part D — Known limitations](#part-d--known-limitations)

---

## Part A — For nurses

### 1. Sign in

Open the dashboard in the browser (usually `http://127.0.0.1:8000`) and sign in with the username and password your admin gave you.

![Sign-in page](img/01_login.png)

- After **5 wrong passwords** the account is locked for **5 minutes**.
- A sign-in lasts up to **12 hours** (one shift). Click **Sign out** (top right) when you leave the station.
- Forgot your password? Ask an admin to reset it.

### 2. The waiting queue

The **Queue** page lists every kiosk patient who is waiting for a nurse. It updates by itself — you do not need to reload. The green **Live** label shows the connection is working.

![Waiting queue](img/02_queue.png)

**Order:** most urgent level first, then the patient who has waited longest.

**Triage levels.** The kiosk computes an ESI level (1–5). The dashboard shows the matching MOPH ED Triage name. The number is always shown next to the colour.

| Level | Name (MOPH code) | Meaning |
| --- | --- | --- |
| **1** | Resuscitation (R) | Needs immediate life-saving care |
| **2** | Emergency (E) | High risk, severe pain or distress, new confusion |
| **3** | Urgency (U) | Stable, but likely needs two or more resources (lab, imaging, IV…) |
| **4** | Semi-urgency (SU) | Likely needs one resource |
| **5** | Non-urgency (NU) | No resources expected |
| **?** | Nurse to triage | The kiosk could not suggest a level — triage this patient yourself |

**Banners at the top**

- **Red banner** — a level 1 or 2 patient who has not been seen yet. Click **Open** to read the case or **Seen** to confirm you are with the patient.
- **Yellow banner** — "A patient at the kiosk needs a nurse". This appears when a patient **declined consent** or asked for help. No personal data is kept for these patients, so please go to the kiosk. Click **Attended** when done.

**Alert sound.** Browsers block sounds until you click once. Click **Turn on alert sound** at the start of the shift; it then beeps when a new red or yellow banner appears. The setting is remembered on that computer.

**Columns:** suggested level · waiting time · arrival time · chief complaint (CC) in the patient's own words · signs seen by the camera · status.

**Status:** **New** (nobody has looked yet) → **Seen** (a nurse is with the patient) → **Decided** (final level given; the patient leaves the queue).

### 3. Reading a case

Click the complaint or **Open**. The case page is laid out like a triage note.

![Case header](img/03_case_header.png)

The header shows the suggested level, the chief complaint, case number, arrival time, waiting time, language and status. Click **Mark as seen** when you meet the patient.

**Situation** — the complaint, the suggested level, how the kiosk session ended, and how strong the evidence is.

![Situation section](img/04_situation.png)

> The blue box is always there: **the kiosk cannot measure vital signs.** Its suggestion uses only what the patient said and what the camera saw. Check vital signs before you confirm the level.

Other notices you may see:

- **Backup rules were used** — the AI reasoning step did not finish in time, so simple rules made the suggestion. Review it with extra care.
- **Some kiosk data is incomplete** — the details are under **System details** at the bottom.

**Observed by the kiosk camera** — posture (holding chest, abdomen or head for 3 seconds or longer) and face colour (redness). The **photo** is only to help you find the right patient in the waiting area.

![Camera observations](img/05_observed.png)

Further down:

- **System assessment and suggested plan** — S/O/A/P text written by the kiosk. A suggestion, not a clinical decision.
- **Kiosk conversation** — everything the kiosk and patient said, in Thai or English.
- **System details** (folded) — technical information for staff checking data quality.

### 4. Completing the patient's details

The right-hand panel, **Completed by nurse**, holds what the kiosk does not collect.

![Completed by nurse](img/06_nurse_form.png)

1. **Identification** — the kiosk never asks for identity. Record the **HN** and the **patient name as on the ID card or passport** when you meet the patient.
2. **Vital signs** — BP, HR, RR, temperature, SpO₂ and pain score (0–10). Leave a box empty if not measured. Impossible values (for example HR 900) are rejected.
3. **Background** — allergies, current medications, past history (PMH) and a free note.
4. Click **Save**. Saving also marks the patient as **Seen**.

**Danger-zone warning.** If HR is above 100/min, RR above 20/min or SpO₂ below 92%, a red box appears and suggests considering **level 2** (ESI decision point D). The dashboard never changes the level by itself — you decide.

### 5. Making the decision

In **Nurse decision**:

1. **Choose the final level** (1–5). The kiosk's suggestion is marked *suggested*.
2. **Choose where the patient goes:** ER resuscitation room, ER treatment area, fast track, OPD queue, or other.
3. **Reason / note** — **required** if you choose a different level from the kiosk, or "Other".
4. Click **Confirm decision**.

If something is missing, the dashboard tells you what to fix:

![Decision with missing items](img/07_decision_error.png)

**Patient not identified yet?** You must record the HN or name first. In an emergency where that is not possible (unconscious, no documents), tick **Patient not yet identified — continue anyway**. This is recorded with the decision.

After saving, the patient leaves the queue and the decision is shown:

![Decision saved](img/08_decision_saved.png)

**Changing a decision.** Click **Change this decision**, choose again and confirm. The earlier decision is kept under **Earlier decisions**; nothing is erased. The kiosk's original suggestion is never overwritten.

### 6. History and statistics

**History** — search past cases by date, level, status, HN, name or case number. Shows the last 7 days unless you choose dates.

![History](img/09_history.png)

**Statistics** — compares the kiosk's suggestion with nurses' final decisions (last 30 days by default).

![Statistics](img/10_stats.png)

- **Kiosk level vs nurse decision** — rows are the kiosk's level, columns the nurse's. Green diagonal = agreement. **Red (below the diagonal)** = the kiosk was *less* urgent than the nurse (under-triage). **Blue (above)** = the kiosk was *more* urgent (over-triage).
- **Median time to first look / to decision** — minutes from the kiosk case arriving until a nurse marked it seen / decided.
- Data quality, how kiosk sessions ended, and where patients were sent.

Agreement with a nurse is not the same as clinical accuracy; these figures are for evaluating the prototype.

### 7. Common situations

| Situation | What to do |
| --- | --- |
| **"?" Nurse to triage** | The kiosk had no level. Triage as usual and choose the level yourself (no reason needed). |
| **"Patient cancelled at the kiosk"** | Check whether the patient is still waiting. If yes, triage them; if not, decide with a note such as "left before triage". |
| **Yellow banner: needs a nurse** | The patient declined consent or asked for help. Go to the kiosk; nothing about them is stored. |
| **Can't find the patient** | Use the kiosk photo and arrival time on the case page. |
| **Kiosk and nurse disagree** | Choose your level and write the reason; this is exactly what the review is for. |
| **Patient got worse after the decision** | Open the case from History, **Change this decision**, and give the new reason. |
| **The page says "Reconnecting…"** | The live link dropped. The queue still refreshes every 30 seconds; reload the page if it continues. |

---

## Part B — For admins

Admins see an extra **Admin** menu.

![Staff accounts](img/11_admin_users.png)

**Staff accounts**

- **Add an account** — full name, username, first password (8+ characters, give it in person), role. *Nurse* can use the queue, cases, decisions, history and statistics. *Admin* can also manage accounts, read the audit log and export data.
- **Reset password** — set a new password; the person is signed out everywhere.
- **Switch off / Switch on** — accounts are never deleted, so the audit trail stays complete. A switched-off account is signed out immediately. You cannot switch off your own account.

**Audit log** — every sign-in, failed sign-in, case opened, data saved, decision, export and automatic deletion, with who and when. Names and HN values are never written to the log.

![Audit log](img/12_audit.png)

**Export (CSV)** — on the Statistics page, **Download CSV (de-identified)** gives one row per case for research: levels, decision, timings, camera signs and vital signs. It contains **no** HN, name, conversation, notes or photo. Every export is recorded in the audit log. The file opens correctly in Excel with Thai text.

**Data protection settings** (shown on the accounts page) are set when the server starts — see Part C.

---

## Part C — Installation and settings (technical)

### Quick start (one computer, SQLite)

Requires Python 3.11 or newer (tested on 3.11 and 3.14).

```bash
python -m venv .venv-dashboard && source .venv-dashboard/bin/activate
pip install -r requirements-dashboard.txt
PYTHONPATH=src python -m nurse_dashboard create-admin
PYTHONPATH=src python -m uvicorn nurse_dashboard.app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. Start the kiosk as usual (`PYTHONPATH=src python -m triage_kiosk.main`); it already sends cases to this address. To try the dashboard without the kiosk:

```bash
python scripts/dashboard/send_demo_cases.py
```

### Updating the dashboard

After pulling a newer version, just restart it. On start-up the dashboard adds any new database columns by itself (the log shows "Database upgraded: added column(s) …"), so existing cases and accounts are kept. Only additive changes are automatic; if a change cannot be applied safely, start-up stops with a message explaining what to do.

### Account commands

```bash
PYTHONPATH=src python -m nurse_dashboard create-admin
PYTHONPATH=src python -m nurse_dashboard create-user [--admin]
PYTHONPATH=src python -m nurse_dashboard set-password USERNAME
PYTHONPATH=src python -m nurse_dashboard deactivate USERNAME   # or: activate
PYTHONPATH=src python -m nurse_dashboard list-users
PYTHONPATH=src python -m nurse_dashboard purge                 # delete data older than the retention period now
```

### Settings (environment variables)

| Variable | Default | Meaning |
| --- | --- | --- |
| `DASHBOARD_DATABASE_URL` | `sqlite:///data/dashboard/dashboard.db` | Database. PostgreSQL: `postgresql+psycopg://user:pass@host:5432/db` |
| `DASHBOARD_EVIDENCE_DIR` | `data/runtime/evidence` | Kiosk photo folder. Photos outside it are never shown. |
| `DASHBOARD_RETENTION_DAYS` | `30` | Cases, decisions, alerts and kiosk photos older than this are deleted automatically (checked at start-up and every 6 h). `0` = keep forever. |
| `DASHBOARD_PURGE_INTERVAL_HOURS` | `6` | How often the retention check runs. |
| `DASHBOARD_KIOSK_KEY` | *(empty)* | Shared key the kiosk must send. Empty = no check (only safe on `127.0.0.1`). |
| `DASHBOARD_SECRET_KEY` | *(auto, saved in `data/dashboard/secret_key`)* | Signs the login cookie. |
| `DASHBOARD_SESSION_HOURS` | `12` | Maximum sign-in length. |
| `DASHBOARD_COOKIE_SECURE` | `0` | Set `1` when served over HTTPS. |
| `DASHBOARD_SERVICE_POINT` | `ED` | `ED` (emergency triage) or `OPD` (outpatient screening point); shown in the top bar. |
| `DASHBOARD_TIMEZONE` | `Asia/Bangkok` | Time zone for displayed times. |

Kiosk side (`src/triage_kiosk/config.py`, environment variables):

| Variable | Default | Meaning |
| --- | --- | --- |
| `NURSE_DASHBOARD_ENDPOINT` | `http://127.0.0.1:8000/api/triage/submit_case` | Where the kiosk sends cases. |
| `NURSE_DASHBOARD_KEY` | *(empty)* | Must equal `DASHBOARD_KIOSK_KEY` when that is set. |

### Using a second computer at the nurse station (LAN)

By default only the kiosk computer can open the dashboard. Before allowing other computers:

1. Set a long random `DASHBOARD_KIOSK_KEY` on the dashboard and the same `NURSE_DASHBOARD_KEY` on the kiosk.
2. Serve the dashboard over **HTTPS** (for example behind Caddy or nginx with a hospital certificate) and set `DASHBOARD_COOKIE_SECURE=1`.
3. Only then listen on the network (`--host 0.0.0.0`), and restrict the port to the nurse-station network with the firewall.

### PostgreSQL with Docker

`docker-compose.yml` starts PostgreSQL 16 and the dashboard together:

```bash
cp .env.example .env          # then change every value in .env
docker compose up -d --build
docker compose exec dashboard python -m nurse_dashboard create-admin
```

> The Docker files were written and checked for syntax, but have not yet been built end-to-end (the build environment could not reach Docker Hub). The PostgreSQL code path itself is tested: the dashboard test suite passes on both SQLite and PostgreSQL 16, and GitHub Actions runs it on both.

### How it fits the kiosk

```
Kiosk → JSON spooled to data/runtime/spool_outbox → POST /api/triage/submit_case → dashboard database
        (kept until the dashboard answers 200/201; re-sent after a restart)
```

- The dashboard never rejects a well-formed case (one rejected case would block the kiosk's queue). Unexpected data is stored and flagged "Some kiosk data is incomplete".
- A re-sent case is recognised by its case number and not stored twice.
- A case without recorded consent keeps only its number, time and exit reason.
- Interactive API documentation: `http://127.0.0.1:8000/docs`.

### Tests

```bash
pip install -r requirements-test.txt
PYTHONPATH=src python -m pytest tests/dashboard tests/test_outbox.py
# also on PostgreSQL:
TEST_POSTGRES_URL=postgresql+psycopg://user:pass@localhost:5432/test PYTHONPATH=src python -m pytest tests/dashboard
```

---

## Part D — Known limitations

- **No vital signs from the kiosk.** The suggested level skips ESI decision point D; the nurse adds vital signs at the dashboard.
- **No identity from the kiosk.** Staff link each case to the HN at the desk (by design, for PDPA data minimisation).
- **Level colours** follow common Thai practice (red, pink, yellow, green, white) and still need confirming against the official MOPH ED Triage guideline. **Danger-zone values** (HR > 100, RR > 20, SpO₂ < 92%) are from the ESI Handbook (5th ed., 2023) and should be checked against its figure before clinical use.
- **Not connected to the hospital information system** (HIS); HL7/FHIR integration is future work.
- **Not yet evaluated with nurses.** A System Usability Scale (SUS) session is planned.
- The dashboard is a research prototype for decision support, not a medical device.
