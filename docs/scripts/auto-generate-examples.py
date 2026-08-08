import os
import re


def process_readme_files():
    print("Processing README.md files from examples/gke and examples/cloud-run...")
    os.makedirs("docs/source/examples", exist_ok=True)

    for dir in ["gke", "cloud-run", "vertex-ai/notebooks"]:
        for root, _, files in os.walk(f"examples/{dir}"):
            for file in files:
                if file == "README.md" or file.__contains__("notebook.md"):
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
    # We only uncomment the metadata block to avoid uncommenting other HTML comments
    content = re.sub(r"<!--\s*(---.*?---)\s*-->", r"\1", content, flags=re.DOTALL)

    # Convert hfoptions comment markers to proper HF docs MDX components.
    # Notebooks use HTML comments so the markers are invisible in Jupyter but are
    # preserved verbatim by nbconvert, and then converted here to the MDX format
    # used by the HuggingFace documentation site, e.g.:
    #   <!-- hfoptions id="model" -->  →  <hfoptions id="model">
    #   <!-- hfoption id="FunctionGemma 270M IT" -->  →  <hfoption id="FunctionGemma 270M IT">
    #   <!-- /hfoption -->  →  </hfoption>
    #   <!-- /hfoptions -->  →  </hfoptions>
    # All hfoptions groups with the same id are linked in the UI: selecting one option
    # toggles every group with that id across the whole page simultaneously.
    content = re.sub(
        r"<!--\s*hfoptions\s+id=\"([^\"]+)\"\s*-->",
        r'<hfoptions id="\1">',
        content,
    )
    content = re.sub(
        r"<!--\s*hfoption\s+id=\"([^\"]+)\"\s*-->",
        r'<hfoption id="\1">',
        content,
    )
    content = re.sub(r"<!--\s*/hfoption\s*-->", r"</hfoption>", content)
    content = re.sub(r"<!--\s*/hfoptions\s*-->", r"</hfoptions>", content)

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

        # Determine the Tip type (NOTE and TIP both map to <Tip>, WARNING to <Tip warning>)
        tip_type = " warning" if block_type == "WARNING" else ""

        # Construct the new block
        new_block = f"<Tip{tip_type}>\n\n"
        new_block += "\n".join(lines)
        new_block += "\n\n</Tip>\n"

        return new_block

    # Regular expression to match the specified blocks
    pattern = r"> \[!(NOTE|TIP|WARNING)\]\n((?:>.*(?:\n|$))+)"

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

    # Calculate the example URL
    if file.__contains__("ipynb"):
        example_url = f"https://github.com/huggingface/Google-Cloud-Containers/tree/main/{root}/{file}"
        content += f"\n\n---\n<Tip>\n\n📍 Find the complete example on GitHub [here]({example_url})!\n\n</Tip>"

    with open(target, "w") as f:
        f.write(content)


if __name__ == "__main__":
    process_readme_files()
