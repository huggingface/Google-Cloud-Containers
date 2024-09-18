import glob
import os
import re


from pathlib import Path


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
                    metadata_match = re.search(r"---(.*?)---", content, re.DOTALL)

                    metadata = {}
                    if metadata_match:
                        metadata_str = metadata_match.group(1)
                        metadata = dict(re.findall(r"(\w+):\s*(.+)", metadata_str))

                        # Remove metadata from content assuming it's the block on top
                        # surrounded by `---` including those too
                        content = re.sub(
                            r"^---\s*\n.*?\n---\s*\n",
                            "",
                            content,
                            flags=re.DOTALL | re.MULTILINE,
                        )
                        content = content.strip()

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
