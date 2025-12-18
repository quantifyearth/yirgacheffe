import mkdocs_gen_files

def changelog() -> None:
    with open("CHANGES.md", "r", encoding="UTF-8") as f:
        content = f.read()

    content = "# Changelog\n\n" + content

    with mkdocs_gen_files.open("changelog.md", "w", encoding="UTF-8") as f:
        f.write(content)

changelog()
