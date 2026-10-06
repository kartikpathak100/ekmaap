# Jury Q&A preparation

Answers use only what the repository has actually tested. Where something has not been tested on real onions, the answer says so.

**1. What problem does EkMaap solve?**
Onion grading at procurement centres is done by eye, so two officers can grade the same crate differently and no record exists of what was seen. EkMaap measures every onion from one photo and issues a report that cannot be changed unnoticed.

**2. How does it work?**
One photo of the sample on a mat with a size reference. The app finds every onion (touching ones are split), measures it in millimetres, decides its condition, applies a versioned grading rule set and produces Grade A / URS / Reject percentages. The report is fingerprinted with SHA-256.

**3. How accurate is it?**
On synthetic photos drawn by our own generator, where true sizes are known, it found 420 of 420 onions with a mean size error of 0.26–0.77 mm across the test sets (`demo/test_results.json`). These are synthetic results. We have not yet measured accuracy on real onions; that needs a labelled real-photo set and is the next stage (`docs/training.md`).

**4. Why not train a model first?**
No labelled real-onion dataset exists for this problem. The prototype uses transparent rules that an officer can inspect, and the system is built to switch to trained models as officers confirm labels.

**5. What if the app is unsure about an onion?**
It is sent to the officer for a decision instead of being graded automatically. A wrong grade costs a farmer money; a flag costs seconds.

**6. How is tampering prevented?**
The report record and photo hashes are combined into a SHA-256 fingerprint shown as a QR code. Changing one number or one photo bit changes the fingerprint, and the public verify page then reports a mismatch. This shows a record is unchanged since issue; it does not prove the sample represents the lot.

**7. What are the grade size limits?**
The demo rule set uses 45 mm and 35 mm, taken from 2026 news reports. They are not from the official specification. They are stored as editable, versioned data so the department's official values can be loaded without a software release.

**8. Does it work offline or on cheap phones?**
The browser demo runs entirely on the phone in a single HTML file and needs no install. Real-world speed on low-end phones is covered by the phone test plan (`docs/test-plan.md`).

**9. How is size measured?**
Against a reference of known size in the photo: a printed 50 mm card in the demo, and an A4 mat with four ArUco markers in the full system, which also corrects phone tilt.

**10. What is the production design?**
FastAPI backend (45 endpoints, four roles), PostgreSQL with 15 append-only tables, file store keyed by SHA-256, CVAT labelling and a training/evaluation loop. The API workflow tests pass on PostgreSQL; the Docker image build and YOLO training have not been run yet.

**11. What about farmer privacy?**
Farmer details are refused without recorded consent, following the DPDP Act 2023 approach, and every action is written to an audit log.

**12. What are the known limits?**
Percentages are by count, not weight. Each onion is seen from one side, so hidden defects and internal rot are not visible. Real-onion accuracy is unmeasured. The browser demo has not been validated by officers.

**13. What happens in a dispute?**
The lot is re-opened, photos or reviews are added and a new report is issued that supersedes the old one. Both remain stored, so the old report can still be explained exactly.

**14. What will you do next?**
Interview officers, collect and label real onion photos on the printed mat, train and evaluate on photos the model has never seen, and replace the demo grade limits with the official specification.
