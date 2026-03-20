# How to contribute
Contributions are always welcome, this page gives a practical summary of the rules and development tools used for this library.

## Rules
The following rules are enforced for all developers of this library:

* Code has to be merged using a pull-request with atleast one review. Codebranch rules enforce this and will block any pushes to main.
* Don't add or change dependencies in the pyproject.toml without discussing with the team leads. As dependencies in python are brittle and might break things further down the line.
* It is fine to use copilot to generate summary in pull-request, but also add manually the why on some of the decisions made.
* Discuss any refactors before opening a pull-request. Big non-discussed refactor pull requests will be closed immediately.

## CI checks

We have the following checks:

* All python tests should pass

* All typecheck errors should be resolved

* Code should be linted using Ruff


## Developer commands

Our pixi config contains some developer convenience commands;

### Running unit-tests

```bash
pixi run test
```

### Running the linter

```bash
pixi run lint
```

### Running the type-checker

```bash
pixi run typecheck
```
