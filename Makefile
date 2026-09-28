# Common tasks. D = labelled dataset folder (annotations.json, images/, ruler.csv), V = model version tag
D ?= data/real
V ?= v1
.PHONY: mat synthetic train evaluate baseline schema api-docs test

mat:            ; python scripts/make_mat.py --out ekmaap_mat.pdf
synthetic:      ; python scripts/make_synthetic.py --out data/synthetic --photos 30
train:
	python scripts/train_segmenter.py --coco $(D)/annotations.json --images $(D)/images --out models/seg_pf_$(V).joblib
	python scripts/train_classifier.py --coco $(D)/annotations.json --images $(D)/images --segmenter models/seg_pf_$(V).joblib --out models/clf_$(V).joblib
	python scripts/fit_size_calibration.py --coco $(D)/annotations.json --images $(D)/images --ruler $(D)/ruler.csv --segmenter models/seg_pf_$(V).joblib --out models/size_$(V).json
evaluate:
	python scripts/evaluate.py --coco $(D)/annotations.json --images $(D)/images --ruler $(D)/ruler.csv --segmenter models/seg_pf_$(V).joblib --classifier models/clf_$(V).joblib --size-cal models/size_$(V).json --out reports/eval_$(V)_test.json
baseline:
	python scripts/evaluate.py --coco $(D)/annotations.json --images $(D)/images --ruler $(D)/ruler.csv --out reports/eval_baseline_test.json
schema:         ; python -m backend.app.dump_schema
api-docs:       ; python scripts/dump_api.py
test:           ; python -m pytest -q
