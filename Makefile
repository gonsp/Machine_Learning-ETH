# 	source activate ml_project

all: test

test:
	python run.py --config .config.yaml -X data/X_train.npy -y data/y_1.csv -a fit