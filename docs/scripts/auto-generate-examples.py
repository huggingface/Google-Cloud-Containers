import os
import re


def process_readme_files():
    print("Processing README.md files from examples/gke and examples/cloud-run...")
    os.makedirs("docs/source/examples", exist_ok=True)

    for dir in ["gke", "cloud-run", "vertex-ai/notebooks"]:
        for root, _, files in os.walk(f"examples/{dir}"):
            for file in files:
                if file == "README.md" or file == "vertex-notebook.md":
                    process_file(root, file, dir)


def process_file(root, file, dir):
    dir_name = dir if not dir.__contains__("/") else dir.replace("/", "-")

    file_path = os.path.join(root, file)
    subdir = root.replace(f"examples/{dir}/", "")
    base = os.path.basename(subdir)

    if file_path == f"examples/{dir}/README.md":
        target = f"docs/source/examples/{dir_name}-index.mdx"
    else:
        target = f"docs/source/examples/{dir_name}-{base}.mdx"

    print(f"Processing {file_path} to {target}")
    with open(file_path, "r") as f:
        content = f.read()

    # For Juypter Notebooks, remove the comment i.e. `<!--` and the `--!>` but keep the metadata
    content = re.sub(r"<!-- (.*?) -->", r"\1", content, flags=re.DOTALL)

    # Replace image and link paths
    content = re.sub(
        r"\(\./(imgs|assets)/([^)]*\.png)\)",
        r"(https://raw.githubusercontent.com/huggingface/Google-Cloud-Containers/main/"
        + root
        + r"/\1/\2)",
        content,
    )
    content = re.sub(
        r"\(\.\./([^)]+)\)",
        r"(https://github.com/huggingface/Google-Cloud-Containers/tree/main/examples/"
        + dir
        + r"/\1)",
        content,
    )
    content = re.sub(
        r"\(\.\/([^)]+)\)",
        r"(https://github.com/huggingface/Google-Cloud-Containers/tree/main/"
        + root
        + r"/\1)",
        content,
    )

    def replacement(match):
        block_type = match.group(1)
        content = match.group(2)

        # Remove '> ' from the beginning of each line
        lines = [line[2:] for line in content.split("\n") if line.strip()]

        # Determine the Tip type
        tip_type = " warning" if block_type == "WARNING" else ""

        # Construct the new block
        new_block = f"<Tip{tip_type}>\n\n"
        new_block += "\n".join(lines)
        new_block += "\n\n</Tip>\n"

        return new_block

    # Regular expression to match the specified blocks
    pattern = r"> \[!(NOTE|WARNING)\]\n((?:>.*(?:\n|$))+)"

    # Perform the transformation
    content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

    # Remove any remaining '>' or '> ' at the beginning of lines
    content = re.sub(r"^>[ ]?", "", content, flags=re.MULTILINE)

    # Check for remaining relative paths
    if re.search(r"\(\.\./|\(\./", content):
        print("WARNING: Relative paths still exist in the processed file.")
        print(
            "The following lines contain relative paths, consider replacing those with GitHub URLs instead:"
        )
        for i, line in enumerate(content.split("\n"), 1):
            if re.search(r"\(\.\./|\(\./", line):
                print(f"{i}: {line}")
    else:
        print("No relative paths found in the processed file.")

    with open(target, "w") as f:
        f.write(content)


if __name__ == "__main__":
    process_readme_files()
