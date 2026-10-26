# Test-case data

No conversation records, patient details, or face images are committed here.
The application creates local case records in `data/runtime/spool_outbox/` and
captured visual evidence in `data/runtime/evidence/` while it runs.

Those runtime files may contain sensitive information. Keep them local, do not
commit or share them, and remove them when they are no longer needed. The
runtime directories are excluded by `.gitignore`.
