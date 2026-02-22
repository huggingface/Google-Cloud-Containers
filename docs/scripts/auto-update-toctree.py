import glob
import os
import re
import subprocess
from pathlib import Path


def get_git_date(file_path):
    if not file_path or not os.path.exists(file_path):
        return None
    try:
        date = (
            subprocess.check_output(
                ["git", "log", "-1", "--format=%ad", "--date=short", file_path],
                stderr=subprocess.STDOUT,
            )
            .decode("utf-8")
            .strip()
        )
        if not date:
            return None
        return date
    except Exception:
        return None


def get_original_file_path(mdx_file):
    stem = Path(mdx_file).stem

    if stem.endswith("-index"):
        d = stem.replace("-index", "")
        if d == "vertex-ai-notebooks":
            return "examples/vertex-ai/notebooks/README.md"
        if d == "vertex-ai":
            return "examples/vertex-ai/README.md"
        return f"examples/{d}/README.md"

    if stem.startswith("vertex-ai-notebooks-"):
        base = stem.replace("vertex-ai-notebooks-", "")
        # Try notebook first, then README
        nb_path = f"examples/vertex-ai/notebooks/{base}/notebook.ipynb"
        if os.path.exists(nb_path):
            return nb_path
        readme_path = f"examples/vertex-ai/notebooks/{base}/README.md"
        if os.path.exists(readme_path):
            return readme_path

    for d in ["gke", "cloud-run"]:
        if stem.startswith(f"{d}-"):
            base = stem.replace(f"{d}-", "")
            # Try notebook first, then README
            nb_path = f"examples/{d}/{base}/notebook.ipynb"
            if os.path.exists(nb_path):
                return nb_path
            readme_path = f"examples/{d}/{base}/README.md"
            if os.path.exists(readme_path):
                return readme_path
    return None


def update_toctree_yaml():
    output_file = "docs/source/_toctree.yml"
    dirs = ["vertex-ai", "gke", "cloud-run"]

    with open(output_file, "a") as f:
        f.write("# GENERATED CONTENT DO NOT EDIT!\n")
        f.write("- sections:\n")

        for dir in dirs:
            f.write("    - sections:\n")

            # Find and sort files
            files = sorted(glob.glob(f"docs/source/examples/{dir}-*.mdx"))
            files = [file for file in files if not file.endswith(f"{dir}-index.mdx")]

            # Dictionary to store files by type
            files_by_type = {}

            for file in files:
                with open(file, "r+") as mdx_file:
                    content = mdx_file.read()

                    # Match metadata, including potentially commented ones or with leading/trailing spaces
                    # We look for the first occurrence of the --- block
                    metadata_match = re.search(
                        r"^\s*(?:<!--\s*)?---\s*\n(.*?)\n---\s*(?:\s*-->)?",
                        content,
                        re.DOTALL | re.MULTILINE,
                    )

                    metadata = {}
                    if metadata_match:
                        metadata_str = metadata_match.group(1)
                        metadata = {
                            k.strip(): v.strip()
                            for k, v in re.findall(r"(\w+):\s*(.+)", metadata_str)
                        }

                        # Remove metadata block from content
                        # Handling potential leading whitespace and the optional <!-- --> wrapper
                        content = re.sub(
                            r"^\s*(?:<!--\s*)?---\s*\n.*?\n---\s*(?:\s*-->)?\s*\n",
                            "",
                            content,
                            flags=re.DOTALL | re.MULTILINE,
                        )
                        content = content.strip()

                        # Add "Written by" and "Last updated"
                        author = metadata.get("author")
                        original_file = get_original_file_path(file)
                        date = get_git_date(original_file)

                        extra_info = ""
                        if author and date:
                            extra_info = f"Written by {author} | Last updated {date}"
                        elif author:
                            extra_info = f"Written by {author}"
                        elif date:
                            extra_info = f"Last updated {date}"

                        if extra_info:
                            extra_info = f"_{extra_info}_"
                            # Match the first # Title line
                            match = re.search(r"^(# .+)$", content, re.MULTILINE)
                            if match:
                                title_line = match.group(1)
                                # Only replace the first occurrence of the title line
                                content = content.replace(
                                    title_line, f"{title_line}\n\n{extra_info}\n", 1
                                )

                        mdx_file.seek(0)
                        mdx_file.write(content)
                        mdx_file.truncate()

                if not all(key in metadata for key in ["title", "type"]):
                    print(f"WARNING: Metadata missing in {file}")
                    print("Ensure that the file contains the following metadata:")
                    print("title: <title>")
                    print("type: <type>")

                    # Remove the file from `docs/source/examples` if doesn't contain metadata
                    print(
                        "Removing the file as it won't be included in the _toctree.yml"
                    )
                    os.remove(file)

                    continue

                file_type = metadata["type"]
                if file_type not in files_by_type:
                    files_by_type[file_type] = []
                files_by_type[file_type].append((file, metadata))

            for file_type, file_list in files_by_type.items():
                f.write("        - sections:\n")
                for file, metadata in file_list:
                    base = Path(file).stem
                    title = metadata["title"]
                    f.write(f"            - local: examples/{base}\n")
                    f.write(f'              title: "{title}"\n')
                f.write("          isExpanded: false\n")
                f.write(f"          title: {file_type.capitalize()}\n")

            f.write("      isExpanded: true\n")

            if dir == "cloud-run":
                f.write(f"      local: examples/{dir}-index\n")
                f.write("      title: Cloud Run\n")
            elif dir == "vertex-ai":
                f.write("      title: Vertex AI\n")
            else:
                f.write(f"      local: examples/{dir}-index\n")
                f.write(f"      title: {dir.upper()}\n")

        f.write("  # local: examples/index\n")
        f.write("  title: Examples\n")
        f.write("# END GENERATED CONTENT\n")


if __name__ == "__main__":
    update_toctree_yaml()
