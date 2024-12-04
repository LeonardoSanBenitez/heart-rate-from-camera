.ONESHELL: # https://stackoverflow.com/a/30590240
.SILENT: # https://stackoverflow.com/a/11015111

setup:
	poetry install --no-root

run-local:
	poetry run jupyter notebook
