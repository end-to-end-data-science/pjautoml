all: clean

clean:
	find . -name "*.so" -o -name "*.pyc" -o -name "*.md5" -o -name "*.pyd" -o -name "*~" | xargs rm -f
	rm -rf .coverage.*
	rm -rf dist
	rm -rf build
	rm -rf docs/_build
	rm -rf docs/source/generated
	rm -rf docs/source/auto_examples
test:
	pytest tests/

test-cov:
	rm -rf coverage .coverage
	pytest --cov=pymfe/ tests/

code_check:
	flake8 pymfe | grep -v __init__
	pylint pymfe -d 'C0103, R0913, R0902, R0914, C0302, R0904, R0801, E1101'
	mypy pymfe --ignore-missing-imports

pypi: clean
	python3 setup.py sdist bdist_wheel
	twine upload dist/*
