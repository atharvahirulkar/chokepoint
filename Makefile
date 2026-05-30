.PHONY: prefilter ingest graph features train eval serve dashboard all clean

prefilter:
	python -m pipeline.prefilter

ingest:
	python -m pipeline.ingest

graph:
	python pipeline/build_graph.py

features:
	python pipeline/features.py

train:
	python models/train.py

eval:
	python models/eval.py

serve:
	uvicorn api.main:app --reload --port 8000

dashboard:
	streamlit run dashboard/app.py

all: ingest graph features train eval

clean:
	rm -rf data/processed/* models/artifacts/*
