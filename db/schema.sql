-- EkMaap database schema (PostgreSQL 14+). GENERATED from backend/app/models.py - do not edit by hand.
-- Apply:  psql "$DATABASE_URL" -f db/schema.sql
CREATE EXTENSION IF NOT EXISTS pgcrypto;  -- gen_random_uuid() on PostgreSQL < 13

CREATE TABLE centres (
	id UUID DEFAULT gen_random_uuid() NOT NULL, 
	code VARCHAR(32) NOT NULL, 
	name VARCHAR(200) NOT NULL, 
	district VARCHAR(100), 
	state VARCHAR(100), 
	lat FLOAT, 
	lon FLOAT, 
	created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
	PRIMARY KEY (id), 
	UNIQUE (code)
);
COMMENT ON TABLE centres IS 'Procurement centres (NAFED / NCCF / APMC sites).';

CREATE TABLE farmers (
	id UUID DEFAULT gen_random_uuid() NOT NULL, 
	name VARCHAR(200) NOT NULL, 
	phone VARCHAR(20), 
	village VARCHAR(120), 
	district VARCHAR(100), 
	consent_at TIMESTAMP WITH TIME ZONE, 
	created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
	PRIMARY KEY (id)
);
COMMENT ON TABLE farmers IS 'Sellers. Minimal personal data; consent time recorded (DPDP Act 2023).';

CREATE TABLE users (
	id UUID DEFAULT gen_random_uuid() NOT NULL, 
	name VARCHAR(200) NOT NULL, 
	login VARCHAR(120) NOT NULL, 
	password_hash VARCHAR(200) NOT NULL, 
	role VARCHAR(20) NOT NULL, 
	centre_id UUID, 
	active BOOLEAN DEFAULT true NOT NULL, 
	created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
	PRIMARY KEY (id), 
	CONSTRAINT ck_role CHECK (role IN ('officer', 'supervisor', 'labeller', 'admin')), 
	UNIQUE (login), 
	FOREIGN KEY(centre_id) REFERENCES centres (id)
);
COMMENT ON TABLE users IS 'Officers, supervisors, labellers, admins. login = phone or email.';

CREATE TABLE audit_log (
	id BIGSERIAL NOT NULL, 
	actor_id UUID, 
	action VARCHAR(64) NOT NULL, 
	entity VARCHAR(32) NOT NULL, 
	entity_id VARCHAR(64), 
	detail JSONB, 
	at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(actor_id) REFERENCES users (id)
);
COMMENT ON TABLE audit_log IS 'Who did what, when. Append-only.';

CREATE TABLE lots (
	id UUID DEFAULT gen_random_uuid() NOT NULL, 
	lot_code VARCHAR(64) NOT NULL, 
	centre_id UUID NOT NULL, 
	officer_id UUID NOT NULL, 
	farmer_id UUID, 
	variety VARCHAR(100), 
	declared_weight_kg NUMERIC(12, 2), 
	status VARCHAR(20) DEFAULT 'open' NOT NULL, 
	lat FLOAT, 
	lon FLOAT, 
	created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
	PRIMARY KEY (id), 
	CONSTRAINT ck_status CHECK (status IN ('open', 'graded', 'reported', 'disputed', 'closed')), 
	UNIQUE (lot_code), 
	FOREIGN KEY(centre_id) REFERENCES centres (id), 
	FOREIGN KEY(officer_id) REFERENCES users (id), 
	FOREIGN KEY(farmer_id) REFERENCES farmers (id)
);
COMMENT ON TABLE lots IS 'A consignment brought to a centre for grading.';

CREATE TABLE model_versions (
	id UUID DEFAULT gen_random_uuid() NOT NULL, 
	kind VARCHAR(20) NOT NULL, 
	backend VARCHAR(20) NOT NULL, 
	version VARCHAR(64) NOT NULL, 
	file_sha256 VARCHAR(64), 
	storage_path TEXT, 
	metrics JSONB, 
	notes TEXT, 
	is_active BOOLEAN DEFAULT false NOT NULL, 
	created_by UUID, 
	created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
	PRIMARY KEY (id), 
	CONSTRAINT ck_kind CHECK (kind IN ('segmenter', 'classifier', 'size_calibration')), 
	CONSTRAINT ck_backend CHECK (backend IN ('classical', 'pixel_forest', 'yolo_seg', 'heuristic', 'sklearn', 'linear')), 
	UNIQUE (version), 
	FOREIGN KEY(created_by) REFERENCES users (id)
);
CREATE UNIQUE INDEX uq_model_versions_one_active_per_kind ON model_versions (kind) WHERE is_active;
COMMENT ON TABLE model_versions IS 'Trained model files and their test-split metrics. One active version per kind.';

CREATE TABLE rule_sets (
	id UUID DEFAULT gen_random_uuid() NOT NULL, 
	name VARCHAR(200) NOT NULL, 
	version VARCHAR(64) NOT NULL, 
	params JSONB NOT NULL, 
	source_note TEXT, 
	is_active BOOLEAN DEFAULT false NOT NULL, 
	created_by UUID, 
	created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
	PRIMARY KEY (id), 
	UNIQUE (version), 
	FOREIGN KEY(created_by) REFERENCES users (id)
);
CREATE UNIQUE INDEX uq_rule_sets_one_active ON rule_sets (is_active) WHERE is_active;
COMMENT ON TABLE rule_sets IS 'Versioned grading rules (size limits, condition -> grade). Never edited after use; make a new version.';
COMMENT ON COLUMN rule_sets.source_note IS 'Where the limits come from (circular no., date).';

CREATE TABLE evaluations (
	id UUID DEFAULT gen_random_uuid() NOT NULL, 
	segmenter_version_id UUID, 
	classifier_version_id UUID, 
	size_cal_version_id UUID, 
	rule_set_id UUID NOT NULL, 
	split VARCHAR(12) NOT NULL, 
	n_photos INTEGER NOT NULL, 
	metrics JSONB NOT NULL, 
	created_by UUID, 
	created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(segmenter_version_id) REFERENCES model_versions (id), 
	FOREIGN KEY(classifier_version_id) REFERENCES model_versions (id), 
	FOREIGN KEY(size_cal_version_id) REFERENCES model_versions (id), 
	FOREIGN KEY(rule_set_id) REFERENCES rule_sets (id), 
	FOREIGN KEY(created_by) REFERENCES users (id)
);
COMMENT ON TABLE evaluations IS 'Test-split results of a model combination. The only numbers to quote.';

CREATE TABLE photos (
	id UUID DEFAULT gen_random_uuid() NOT NULL, 
	lot_id UUID, 
	purpose VARCHAR(10) NOT NULL, 
	sha256 VARCHAR(64) NOT NULL, 
	storage_path TEXT NOT NULL, 
	original_name VARCHAR(255), 
	content_type VARCHAR(50), 
	width INTEGER NOT NULL, 
	height INTEGER NOT NULL, 
	split VARCHAR(12) DEFAULT 'unassigned' NOT NULL, 
	split_group VARCHAR(64), 
	uploaded_by UUID, 
	lat FLOAT, 
	lon FLOAT, 
	created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
	PRIMARY KEY (id), 
	CONSTRAINT ck_purpose CHECK (purpose IN ('lot', 'dataset')), 
	CONSTRAINT ck_split CHECK (split IN ('unassigned', 'train', 'val', 'test')), 
	FOREIGN KEY(lot_id) REFERENCES lots (id) ON DELETE CASCADE, 
	FOREIGN KEY(uploaded_by) REFERENCES users (id)
);
CREATE INDEX ix_photos_lot ON photos (lot_id);
CREATE UNIQUE INDEX uq_photos_dataset_sha ON photos (sha256) WHERE purpose = 'dataset';
COMMENT ON TABLE photos IS 'Every uploaded photo: of a lot (grading) or for the training dataset.';
COMMENT ON COLUMN photos.split_group IS 'e.g. capture day; splits keep a group together';

CREATE TABLE reports (
	id UUID DEFAULT gen_random_uuid() NOT NULL, 
	lot_id UUID NOT NULL, 
	record JSONB NOT NULL, 
	canonical TEXT NOT NULL, 
	record_sha256 VARCHAR(64) NOT NULL, 
	status VARCHAR(12) DEFAULT 'valid' NOT NULL, 
	supersedes_id UUID, 
	issued_by UUID NOT NULL, 
	issued_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
	PRIMARY KEY (id), 
	CONSTRAINT ck_status CHECK (status IN ('valid', 'superseded')), 
	FOREIGN KEY(lot_id) REFERENCES lots (id), 
	UNIQUE (record_sha256), 
	FOREIGN KEY(supersedes_id) REFERENCES reports (id), 
	FOREIGN KEY(issued_by) REFERENCES users (id)
);
CREATE INDEX ix_reports_lot_id ON reports (lot_id);
COMMENT ON TABLE reports IS 'Issued quality reports. record is exactly what was fingerprinted; never modified.';
COMMENT ON COLUMN reports.canonical IS 'The exact UTF-8 text that was hashed (sorted keys, no spaces).';

CREATE TABLE analyses (
	id UUID DEFAULT gen_random_uuid() NOT NULL, 
	photo_id UUID NOT NULL, 
	segmenter_version_id UUID, 
	classifier_version_id UUID, 
	size_cal_version_id UUID, 
	rule_set_id UUID NOT NULL, 
	scale_found BOOLEAN NOT NULL, 
	calibration JSONB, 
	summary JSONB NOT NULL, 
	timing_ms JSONB, 
	created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(photo_id) REFERENCES photos (id) ON DELETE CASCADE, 
	FOREIGN KEY(segmenter_version_id) REFERENCES model_versions (id), 
	FOREIGN KEY(classifier_version_id) REFERENCES model_versions (id), 
	FOREIGN KEY(size_cal_version_id) REFERENCES model_versions (id), 
	FOREIGN KEY(rule_set_id) REFERENCES rule_sets (id)
);
CREATE INDEX ix_analyses_photo ON analyses (photo_id, created_at);
COMMENT ON TABLE analyses IS 'One run of the pipeline on one photo, with the exact models and rule set used.';

CREATE TABLE annotations (
	id UUID DEFAULT gen_random_uuid() NOT NULL, 
	photo_id UUID NOT NULL, 
	polygon JSONB NOT NULL, 
	condition VARCHAR(10) NOT NULL, 
	ruler_mm FLOAT, 
	source VARCHAR(10) NOT NULL, 
	labeller_id UUID, 
	created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
	PRIMARY KEY (id), 
	CONSTRAINT ck_condition CHECK (condition IN ('sound', 'sprouted', 'damaged', 'rotten')), 
	CONSTRAINT ck_source CHECK (source IN ('manual', 'prelabel', 'import', 'review')), 
	FOREIGN KEY(photo_id) REFERENCES photos (id) ON DELETE CASCADE, 
	FOREIGN KEY(labeller_id) REFERENCES users (id)
);
CREATE INDEX ix_annotations_photo_id ON annotations (photo_id);
COMMENT ON TABLE annotations IS 'Ground-truth labels for TRAINING and EVALUATION (polygon + condition, optional ruler size).';

CREATE TABLE disputes (
	id UUID DEFAULT gen_random_uuid() NOT NULL, 
	report_id UUID NOT NULL, 
	raised_by_name VARCHAR(200) NOT NULL, 
	raised_by_role VARCHAR(10) NOT NULL, 
	reason TEXT NOT NULL, 
	status VARCHAR(10) DEFAULT 'open' NOT NULL, 
	resolution_note TEXT, 
	resolved_by UUID, 
	new_report_id UUID, 
	created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
	resolved_at TIMESTAMP WITH TIME ZONE, 
	PRIMARY KEY (id), 
	CONSTRAINT ck_raised_by_role CHECK (raised_by_role IN ('farmer', 'officer', 'trader', 'other')), 
	CONSTRAINT ck_status CHECK (status IN ('open', 'regraded', 'upheld', 'rejected')), 
	FOREIGN KEY(report_id) REFERENCES reports (id), 
	FOREIGN KEY(resolved_by) REFERENCES users (id), 
	FOREIGN KEY(new_report_id) REFERENCES reports (id)
);
CREATE INDEX ix_disputes_report_id ON disputes (report_id);
COMMENT ON TABLE disputes IS 'A challenge to an issued report and how it was resolved.';

CREATE TABLE detections (
	id UUID DEFAULT gen_random_uuid() NOT NULL, 
	analysis_id UUID NOT NULL, 
	idx INTEGER NOT NULL, 
	polygon JSONB NOT NULL, 
	centroid JSONB NOT NULL, 
	diameter_mm FLOAT, 
	width_mm FLOAT, 
	condition VARCHAR(10) NOT NULL, 
	proba JSONB NOT NULL, 
	grade VARCHAR(10) NOT NULL, 
	label VARCHAR(20) NOT NULL, 
	needs_review BOOLEAN NOT NULL, 
	reasons JSONB NOT NULL, 
	features JSONB NOT NULL, 
	PRIMARY KEY (id), 
	CONSTRAINT ck_condition CHECK (condition IN ('sound', 'sprouted', 'damaged', 'rotten')), 
	CONSTRAINT ck_grade CHECK (grade IN ('Grade A', 'URS', 'Reject', 'Review')), 
	UNIQUE (analysis_id, idx), 
	FOREIGN KEY(analysis_id) REFERENCES analyses (id) ON DELETE CASCADE
);
COMMENT ON TABLE detections IS 'One onion found by the pipeline. Polygon in original photo pixels.';

CREATE TABLE detection_reviews (
	id UUID DEFAULT gen_random_uuid() NOT NULL, 
	detection_id UUID NOT NULL, 
	reviewer_id UUID NOT NULL, 
	condition VARCHAR(10) NOT NULL, 
	diameter_mm FLOAT, 
	note TEXT, 
	created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
	PRIMARY KEY (id), 
	CONSTRAINT ck_condition CHECK (condition IN ('sound', 'sprouted', 'damaged', 'rotten')), 
	FOREIGN KEY(detection_id) REFERENCES detections (id) ON DELETE CASCADE, 
	FOREIGN KEY(reviewer_id) REFERENCES users (id)
);
CREATE INDEX ix_detection_reviews_detection_id ON detection_reviews (detection_id);
COMMENT ON TABLE detection_reviews IS 'Officer corrections at grading time. Latest review wins; history kept.';
COMMENT ON COLUMN detection_reviews.diameter_mm IS 'Set only if the officer measured it';

