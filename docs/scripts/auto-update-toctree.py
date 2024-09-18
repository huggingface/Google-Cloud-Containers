import glob
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
                with open(file, "r") as mdx_file:
                    content = mdx_file.read()
                    # Extract the metadata surrounded with `---`
                    metadata = re.search(r"---(.*?)---", content, re.DOTALL)
                    if metadata:
                        metadata = metadata.group(1)
                        metadata = dict(
                            line.split(": ", 1) for line in metadata.strip().split("\n")
                        )
                    else:
                        metadata = {}

                if not all(key in metadata for key in ["title", "type"]):
                    print(f"WARNING: Metadata missing in {file}")
                    print("Ensure that the file contains the following metadata:")
                    print("title: <title>")
                    print("type: <type>")
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

            f.write("      isExpanded: false\n")

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

    print("YAML tree structure appended to docs/source/_toctree.yml")


if __name__ == "__main__":
    update_toctree_yaml()
