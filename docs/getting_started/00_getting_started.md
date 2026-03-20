# How to integrate in your project

To use this library in your project, you must add it as a dependency in your `pyproject.toml`.

Since the library is **not published on PyPI**, you have two options:

1. **Install directly from GitHub**
2. **Clone the repository locally and reference it from your project**

---

## Install directly from GitHub

This is the easiest way if you simply want to **use the library as a dependency**.

=== "UV"

    Add the dependency using:

    ```bash
    uv add "database-builder-libs @ git+https://github.com/AIM-kennisplatformen/database-builder-libs.git"
    ```

    This will automatically update your `pyproject.toml` and install the package.

=== "Pixi"
    Add the dependency with:

    ```bash
    pixi add "database-builder-libs @ git+https://github.com/AIM-kennisplatformen/database-builder-libs.git"
    ```

    Pixi will add the dependency under the `pypi-dependencies` section in your `pyproject.toml`.

## Install from a local clone (development)

If you want to **develop or test changes in the library**, clone the repository locally and reference the folder.

Clone the repository inside your project directory:

```bash
git clone https://github.com/AIM-kennisplatformen/database-builder-libs.git
```

Your project structure might look like this:

```
my-project/
├─ pyproject.toml
├─ database-builder-libs/
```


=== "UV"

    Add the local dependency to your `pyproject.toml`:

    ```toml
    [project]
    dependencies = [
        "database-builder-libs @ file:./database-builder-libs"
    ]
    ```

    For active development, you can install it in **editable mode**:

    ```bash
    uv pip install -e ./database-builder-libs
    ```


=== "Pixi"

    Add the following to your `pyproject.toml`:

    ```toml
    [tool.pixi.pypi-dependencies]
    database-builder-libs = { path = "./database-builder-libs", editable = true }
    ```

    Using `editable = true` ensures that changes to the local library are immediately reflected in your environment.


