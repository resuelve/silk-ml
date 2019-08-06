test:
	python3 -m unittest discover

# Publish commands
publish: test publish.docs publish.package publish.pypi

publish.docs:
	sphinx-build -M html sphinx_docs docs

publish.package:
	rm -rf build
	python3 setup.py sdist bdist_wheel

publish.pypi:
	python3 -m twine upload dist/*

