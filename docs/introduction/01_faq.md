# FAQ

## Where can I ask questions?

We use [GitHub issues](https://github.com/AIM-kennisplatformen/database-builder-libs/issues) for development related discussion. You should use them only if your question or issue is tightly related to the development of the library.

Before posting a question, please read this FAQ section since you might find the answer to your issue here as well.

## Is GPU/NPU supported?

For the embedding of documents this library uses docling which means that if your GPU/NPU is supported by docling. It should work with this library as well.

Any tools within this library using a LLM (such as summarization/text processing) is written with Ollama and OpenAI api compatibility. Meaning that the support of GPU/NPU for these type of tools is bound by whether Ollama supports your GPU/NPU or you have a LLM-inference runtime with OpenAI API compatibility.

## Why use Pixi not UV?
The library has some limited support for UV, meaning we try to keep it backward-compatible with UV. [Pixi](https://prefix.dev) is used as our main package manager. The reason for choosing Pixi over UV:

* Pixi uses both Conda and PyPI packages which makes it easier to install non-Python system dependencies (e.g. database drivers, scientific runtimes, OCR/PDF tooling, ML backends) in a reproducible way across platforms.
* Reproducibility: Pixi provides fully locked multi-platform environments (Linux/macOS/Windows) including native libraries, while uv only locks Python dependencies.
* CI stability: container-based services (TypeDB, Qdrant, embedding libraries, PDF tooling) often depend on compiled binaries. Conda packages prevent many platform-specific build failures that occur when compiling from PyPI wheels or source.
* Faster onboarding: developers do not need a preconfigured system Python, compilers, or OS packages — `pixi install` produces a complete working environment.
* Task runner integration: Pixi replaces Makefiles/tox/nox scripts by providing a consistent cross-platform task system (`pixi run test`, `pixi run lint`, etc.).
* Deterministic tooling versions: linters, type checkers, and test tools run in the same environment as the library, avoiding “works on my machine” issues.
* Future ML compatibility: vector databases and embedding pipelines frequently depend on native libraries (BLAS, CUDA, tokenizers). Pixi avoids binary compatibility issues when these are introduced.

UV remains supported for users that only need the pure-Python subset of the library, but Pixi is the reference environment for development, testing, and full functionality.