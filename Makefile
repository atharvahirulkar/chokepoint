.PHONY: prefilter ingest graph features labels train eval serve dashboard all clean

prefilter:
	python -m pipeline.prefilter

ingest:
	python -m pipeline.ingest

graph:
	python -m pipeline.build_graph

features:
	python -m pipeline.features

labels:
	python -m pipeline.labels

train:
	python -m models.train

eval:
	python -m models.eval

serve:
	uvicorn api.main:app --reload --port 8000

dashboard:
	streamlit run dashboard/app.py

all: ingest graph features labels train eval

clean:
	rm -rf data/processed/* models/artifacts/* eval_report.json
