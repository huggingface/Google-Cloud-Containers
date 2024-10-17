import os
import re
import subprocess


def current_git_branch():
    try:
        # First, try to get the branch name from GitHub Actions environment variables
        if "GITHUB_REF" in os.environ:
            branch = os.environ["GITHUB_REF"].split("/")[-1]
        else:
            # If not in GitHub Actions, use git command
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
            branch = result.stdout.strip()

        # Check if the branch is 'HEAD' (detached state in CI)
        if branch == "HEAD":
            # Try to get the branch name from GitHub Actions specific environment variable
            if "GITHUB_HEAD_REF" in os.environ:
                branch = os.environ["GITHUB_HEAD_REF"]
            elif "GITHUB_REF" in os.environ:
                branch = os.environ["GITHUB_REF"].split("/")[-1]
            else:
                branch = "Unable to determine branch name"

        return branch
    except:  # noqa: E722
        return "main"


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

    # Obtain the current checked out Git branch
    git_branch = current_git_branch()

    # For Juypter Notebooks, remove the comment i.e. `<!--` and the `--!>` but keep the metadata
    content = re.sub(r"<!-- (.*?) -->", r"\1", content, flags=re.DOTALL)

    # Replace image and link paths
    content = re.sub(
        r"\(\./(imgs|assets)/([^)]*\.png)\)",
        rf"(https://raw.githubusercontent.com/huggingface/Google-Cloud-Containers/{git_branch}/"
        + root
        + r"/\1/\2)",
        content,
    )
    content = re.sub(
        r"\(\.\./([^)]+)\)",
        rf"(https://github.com/huggingface/Google-Cloud-Containers/tree/{git_branch}/examples/"
        + dir
        + r"/\1)",
        content,
    )
    content = re.sub(
        r"\(\.\/([^)]+)\)",
        rf"(https://github.com/huggingface/Google-Cloud-Containers/tree/{git_branch}/"
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

    # Calculate the example URL
    example_url = f"https://github.com/huggingface/Google-Cloud-Containers/tree/{git_branch}/{root}"
    if file.__contains__("ipynb"):
        example_url += "/vertex-notebook.ipynb"

    # Add the final note
    content += f"\n---\n<Tip>\n\n📍 Find the complete example on GitHub [here]({example_url})!\n\n</Tip>"

    with open(target, "w") as f:
        f.write(content)


if __name__ == "__main__":
    process_readme_files()
