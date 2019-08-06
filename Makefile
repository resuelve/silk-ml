test:
	python3 -m unittest discover

# Publish commands
publish: test publish.docs publish.package publish.pypi

publish.docs:
	sphinx-build -M html docs build

publish.package:
	python3 setup.py sdist bdist_wheel

publish.pypi:
	python3 -m twine upload dist/*

