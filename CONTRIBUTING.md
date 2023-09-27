# Contributor Guide

Thank you for your interest in improving this project.
This project is open-source under the [MIT license] and
welcomes contributions in the form of bug reports, feature requests, and pull requests.

Here is a list of important resources for contributors:

- [Source Code]
- [Documentation]
- [Issue Tracker]
- [Code of Conduct]

[mit license]: https://github.com/chainyo/tensorshare/blob/main/LICENSE
[source code]: https://github.com/chainyo/tensorshare
[documentation]: https://chainyo.github.io/tensorshare
[issue tracker]: https://github.com/chainyo/tensorshare/issues

## How to report a bug

Report bugs on the [Issue Tracker].

When filing an issue, make sure to answer these questions:

- Which operating system and Python version are you using?
- Which version of this project are you using?
- What did you do?
- What did you expect to see?
- What did you see instead?

The best way to get your bug fixed is to provide a test case,
and/or steps to reproduce the issue.

## How to request a feature

Request features on the [Issue Tracker].

## How to set up your development environment

You need Python 3.9+ and [Hatch].

It is recommended to use `pipx` to install these tools to make it system-wide available:

```console
$ pipx install hatch
```

Install the package with development requirements:

```console
$ hatch env create
$ hatch shell
```

[hatch]: https://hatch.pypa.io/latest/

## Quality checks

Run the quality checks before committing your change:

```console
$ hatch run quality:check  # linting and code formatting without fixing

$ hatch run quality:format  # linting and code formatting with fixing and preview
```

## How to test the project

Run the full test suite:

```console
$ hatch run tests:run
```


Unit tests are located in the _tests_ directory,
and are written using the [pytest] testing framework.

[pytest]: https://pytest.readthedocs.io/

## How to submit changes

Open a [pull request] to submit changes to this project.

Your pull request needs to meet the following guidelines for acceptance:

- The Nox test suite must pass without errors and warnings.
- Include unit tests. This project maintains 100% code coverage.
- If your changes add functionality, update the documentation accordingly.

Feel free to submit early, though—we can always iterate on this.

To run linting and code formatting checks before committing your change, you can install pre-commit as a Git hook by running the following command:

It is recommended to open an issue before starting work on anything.
This will allow a chance to talk it over with the owners and validate your approach.

[pull request]: https://github.com/chainyo/tensorshare/pulls

<!-- github-only -->

[code of conduct]: CODE_OF_CONDUCT.md